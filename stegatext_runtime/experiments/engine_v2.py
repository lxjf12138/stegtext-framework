"""Explicit stopping/transport around the unchanged method and coder kernels."""
from __future__ import annotations

import hashlib
import hmac
import math
import time

import torch

from . import framing
from .config import canonical
from .stopping import contract, render, termination
from round4_2_common import CounterRNG
from stegtext.starter.adaptive import select_kernel
from stegtext.v2.core.types import MethodContext, VisibleTarget


def nonce(source, prompt, seed):
    return hashlib.sha256(canonical([source.manifest()["sha256"], prompt, seed]).encode()).hexdigest()


def shared_key(config, seed, carrier_nonce):
    from .engine import key_for
    return hmac.new(key_for(config, seed), b"stegatext-carrier-v2:"+carrier_nonce.encode(), hashlib.sha256).digest()


def public_termination(result):
    keys = ['terminal_reason', 'censored', 'framing_status', 'watchdog']
    if 'sentence_goal_reached' in result:
        keys += ['sentence_goal_reached','first_sentence_utf8_bytes','overshoot_utf8_bytes',
                 'completed_sentences','sentence_boundary_rule','last_action_preserved']
    if 'token_budget' in result:
        keys += ['token_budget']
    return {k:result[k] for k in keys}


def _xlog_sum(values):
    return math.fsum(float(value) * math.log2(float(value)) for value in values if float(value) > 0.0)


def _selected_record_diagnostics(plan, action, state, distributions=(), codec=None):
    """Return bounded metrics for the selected belief/action.

    Selective LAS plans expose aggregate masses and xlog sums.  We use those
    aggregates directly; a full frontier or vocabulary vector is never copied
    into the experiment record.  Exact realization baselines expose members,
    for which the selected member entropy is computed from the already loaded
    source probabilities.
    """
    opaque = plan.opaque if isinstance(plan.opaque, dict) else {}
    records = opaque.get("records", ())
    if records:
        record = records[int(action)]
        total = float(record.get("group_mass", opaque.get("masses", plan.probs.tolist())[int(action)]))
        exact = float(record.get("exact_mass", 0.0))
        strict = float(record.get("strict_mass", max(0.0, total - exact)))
        strict_xlog = float(record.get("strict_xlogx_sum", 0.0))
        exact_xlog = None
        lex = opaque.get("lex_masses")
        if lex is not None and "fresh_exact_interval" in record:
            lo, hi = record["fresh_exact_interval"]
            values = [float(value) for value in lex[int(lo):int(hi)] if float(value) > 0.0]
            frontier = opaque.get("frontier")
            if frontier is not None:
                rlo, rex = record["retained_interval"][0], record["retained_exact_end"]
                values.extend(float(value) for value in frontier.masses[int(rlo):int(rex)] if float(value) > 0.0)
            exact_xlog = _xlog_sum(values)
        if exact_xlog is None:
            # The aggregate is still useful when the optimizer does not expose
            # exact atom masses.  The missing value is explicit rather than an
            # invented identity entropy.
            exact_identity = None
            candidate_identity = None
        else:
            exact_identity = (-exact_xlog / exact + math.log2(exact)) if exact > 0 else 0.0
            candidate_identity = (-(exact_xlog + strict_xlog) / total + math.log2(total)) if total > 0 else 0.0
        strict_identity = ((-strict_xlog / strict + math.log2(strict)) if strict > 0 else 0.0)
        return dict(selected_group_key_hex=bytes(record.get("key", b"")).hex(),
                    selected_group_mass=total,
                    selected_group_exact_mass=exact,
                    selected_group_strict_mass=strict,
                    selected_group_strict_posterior_mass=strict / total if total > 0 else None,
                    exact_candidate_count=int(record.get("exact_count", 0)),
                    strict_candidate_count=int(record.get("strict_count", 0)),
                    candidate_identity_entropy_bits=candidate_identity,
                    exact_identity_entropy_bits=exact_identity,
                    strict_identity_entropy_bits=strict_identity,
                    posterior_entropy_bits=float(record.get("las_posterior_entropy_bits", 0.0)),
                    U_post=float(record.get("las_posterior_entropy_bits", 0.0)),
                    posterior_information_upper_bound_bits=float(record.get("las_posterior_entropy_bits", 0.0)),
                    exact_identity_information_loss_bits=(
                        exact_identity * exact / total if exact_xlog is not None and total > 0 else None),
                    record_metrics_source="planner_aggregate")
    members = opaque.get("members")
    if members is not None:
        selected_members = list(members[int(action)])
        dist = distributions[0] if distributions else opaque.get("dist")
        values = [] if dist is None else [float(dist.probs[int(token)].item()) for token in selected_members]
        mass = math.fsum(values)
        identity = -math.fsum((v / mass) * math.log2(v / mass) for v in values if v > 0) if mass > 0 else 0.0
        key = opaque.get("keys", [b""])[int(action)]
        return dict(selected_group_key_hex=bytes(key).hex(), selected_group_mass=mass,
                    selected_group_exact_mass=mass, selected_group_strict_mass=0.0,
                    selected_group_strict_posterior_mass=0.0, exact_candidate_count=len(values),
                    strict_candidate_count=0, candidate_identity_entropy_bits=identity,
                    exact_identity_entropy_bits=identity, strict_identity_entropy_bits=0.0,
                    posterior_entropy_bits=0.0, U_post=0.0,
                    posterior_information_upper_bound_bits=0.0,
                    exact_identity_information_loss_bits=identity,
                    record_metrics_source="exact_realization_members")
    return dict(selected_group_key_hex=None, selected_group_mass=float(plan.probs[int(action)].item()),
                selected_group_exact_mass=None, selected_group_strict_mass=None,
                selected_group_strict_posterior_mass=None, exact_candidate_count=None,
                strict_candidate_count=None, candidate_identity_entropy_bits=None,
                exact_identity_entropy_bits=None, strict_identity_entropy_bits=None,
                posterior_entropy_bits=None, U_post=None,
                posterior_information_upper_bound_bits=None, record_metrics_source="unavailable")


def instrumentation(plan, action, state, before_state=None, distributions=(), law_kind=None, codec=None):
    def retained_mass(value):
        rows = getattr(value, 'retained', ())
        masses = rows.masses if hasattr(rows, 'masses') else (r.mass for r in rows)
        return math.fsum(masses)
    p = plan.probs.detach().cpu().tolist()
    from .recording import distribution_trace, probability_summary, state_summary
    group_entropy = -math.fsum(x*math.log2(x) for x in p if x > 0)
    row = dict(law_kind=law_kind,
               **distribution_trace(p, action, raw_entropy=group_entropy),
               **probability_summary(p, selected=action, prefix="group"),
               group_count=len(p), retained_mass=retained_mass(state))
    row["group_entropy_bits"] = group_entropy
    row.update(positive_source_support=[int(torch.count_nonzero(d.probs).item()) for d in distributions],
               source_entropy_bits=[float((-d.probs[d.probs>0].double()*torch.log2(d.probs[d.probs>0].double())).sum().item()) for d in distributions],
               selected_group_probability=p[int(action)],
               selected_group_information_bits=-math.log2(p[int(action)]))
    row["source_distribution_summary"] = [
        probability_summary(d.probs, prefix=f"source_{index}") for index, d in enumerate(distributions)
    ]
    members = plan.opaque.get('members') if isinstance(plan.opaque, dict) else None
    if members is not None and distributions:
        raw = distributions[0].probs
        indices = torch.tensor(sorted({int(t) for group in members for t in group}), device=raw.device, dtype=torch.long)
        row['source_probability_mass_retained'] = float(raw[indices].double().sum().item()/raw.double().sum().item())
    elif before_state is not None and hasattr(before_state, 'expansion_mass'):
        row['source_probability_mass_retained'] = 1.
    # These are scalar diagnostics only.  They do not participate in planner
    # decisions and never retain a full-vocabulary vector.  For a filtered
    # candidate law q=p/m on retained support, the closed form is exact.  For
    # grouped LAS/SyncPool laws, the relevant source-group marginal is the
    # distribution-preserving contract and is therefore zero by construction;
    # final visible-text KL still requires a multi-sample estimate.
    if 'source_probability_mass_retained' in row:
        from .kl import restricted_distribution_metrics
        row.update(restricted_distribution_metrics(row['source_probability_mass_retained']))
    if law_kind == 'group_preserving':
        row.update(probability_space='visible_group_action',
                   group_kl_method_to_source_bits=0.0,
                   group_kl_source_to_method_bits=0.0,
                   group_tv_distance=0.0,
                   group_kl_status='ideal_marginal_contract',
                   final_visible_text_kl_status='requires_multi_sample_estimate')
    elif law_kind == 'filtered':
        row.update(probability_space='filtered_candidate_action',
                   group_kl_status='not_applicable',
                   final_visible_text_kl_status='requires_multi_sample_estimate')
    else:
        row.update(probability_space='unspecified_action',
                   group_kl_status='not_recorded',
                   final_visible_text_kl_status='requires_multi_sample_estimate')
    # Keep the legacy field as an after-advance alias; name both times explicitly.
    row['retained_mass_after'] = row['retained_mass']
    if before_state is not None:
        row['retained_mass_before'] = retained_mass(before_state)
    records = plan.opaque.get("records", []) if isinstance(plan.opaque, dict) else []
    if isinstance(plan.opaque, dict) and 'baseline_contract' in plan.opaque:
        row['baseline_contract'] = plan.opaque['baseline_contract']
    if records:
        selected = records[action]
        row.update(exact_count=selected["exact_count"], strict_count=selected["strict_count"],
                   multi_exact_groups=sum(r["exact_count"] > 1 for r in records),
                   U_post=selected["las_posterior_entropy_bits"])
    row.update(_selected_record_diagnostics(plan, action, before_state or state, distributions, codec))
    after = state_summary(state)
    row.update({f"{key}_after": value for key, value in after.items()})
    if before_state is not None:
        before = state_summary(before_state)
        row.update({f"{key}_before": value for key, value in before.items()})
        row["expansion_mass_delta"] = after["active_mass"] - before["active_mass"]
        row["retained_frontier_count_delta"] = after["retained_frontier_count"] - before["retained_frontier_count"]
        row["retained_frontier_mass_delta"] = after["retained_frontier_mass"] - before["retained_frontier_mass"]
    return row


@torch.inference_mode()
def planner(source, packet, *, key, stream=None, observer=None):
    from .engine import coder, goal, method
    from research_notes.full_validation import suite
    sending = stream is not None
    if not source.config.do_sample:
        raise ValueError("Planner requires a sampling source")
    variant, stop = packet["variant"], packet["stopping"]
    from stegatext_runtime.prompts import configure_prompt
    configure_prompt(source, packet.get('assistant_prefix'))
    token_mode = stop['mode'] == 'token_budget'
    if token_mode:
        from .token_budget import TokenBudgetSource
        source = TokenBudgetSource(source, stop['max_new_tokens'])
    if sending and packet.get('message_delivery') == 'prefix_stream' and (not stream or set(stream)-{'0','1'}):
        raise ValueError('Prefix-stream requires nonempty actual binary input')
    codec = source.backend.codec
    prompt = source.bind(packet["prompt"])
    reference_forwards_before = getattr(source.reference, "forwards", None)
    shared_sync = variant.get('implementation') == 'shared_planner_v1'
    m = method(source, 'fullsync' if shared_sync else variant["method"])
    state = m.start(prompt)
    if sending:
        target = None
    else:
        data = codec.text_to_bytes(packet["text"]) if packet["text"] is not None else bytes.fromhex(packet["visible_bytes_hex"])
        capped = packet.get('termination', {}).get('token_budget', {}).get('token_limit_reached', False)
        target = VisibleTarget(data+(codec.eos_bytes if packet["natural_eos"] or capped else b""))
    ctx = MethodContext(codec, prompt, "encode" if sending else "decode", target=target)
    c = coder(variant["coder"], ctx.mode, variant.get('coder_parameters'))
    selective = shared_sync or variant["method"] in ("las", "fullsync", "adaptive")
    tau = 0. if variant["method"] == "las" else math.inf if shared_sync or variant["method"] == "fullsync" else variant.get("tau")
    natural = stop["mode"] == "natural_eos_only"
    sentence_mode = stop['mode'] == 'sentence_boundary'
    watchdog = stop["watchdog_protocol_decisions"]
    limit = watchdog if natural or sentence_mode or token_mode else min(watchdog, stop["max_steps"]+stop["utf8_extension"])
    trace, pieces = [], []
    offset, reason = 0, "invalid_framing"
    planning_seconds, coder_seconds = 0., 0.
    max_required = 0
    model_query_count = 0
    protocol_decision_count = 0
    cache_snapshot_restores = 0
    for step in range(1, limit+1):
        if target is not None and target.done:
            reason = "packet_boundary"
            break
        ctx.step = step
        paths = ([state.base_path] if state.expansion_mass > 0 else []) if selective else m.queries(state)
        distributions = [source.distribution(path) for path in paths]
        model_query_count += len(paths)
        start = time.monotonic()
        plan = m.plan(state, distributions, ctx)
        planning_seconds += time.monotonic()-start
        start = time.monotonic()
        if sending:
            # iMEC inspects a real input window. DisCop checks each bit when
            # its selected tree path needs it; alphabet size is not bit demand.
            required = (None if variant['coder'] == 'discop_copy' else
                        c.block_size*c.window_size if hasattr(c, 'block_size') else len(plan.probs))
            if required is not None:
                max_required = max(max_required, offset+required)
            if required is not None and packet.get('message_delivery') == 'prefix_stream' and offset+required > len(stream):
                raise ValueError('Insufficient unpadded input for conservative coder lookahead')
            supplied = stream[offset:]
            if packet.get("message_delivery") == "finite_frame":
                supplied += "0"*(len(plan.probs)+1)
            out, action = c.encode(plan.probs.cpu(), supplied, rng=CounterRNG(key, "coder", step))
            if out != supplied[:len(out)]:
                raise ValueError("Encoder did not consume the supplied wire prefix")
            if required is None:
                max_required = max(max_required, offset+len(out))
        else:
            action = m.observe(state, plan, target, ctx)
            out = c.decode(plan.probs.cpu(), action, rng=CounterRNG(key, "coder", step))
        coder_seconds += time.monotonic()-start
        protocol_decision_count += 1
        kernel, advance = None, m.advance
        if selective:
            kernel = select_kernel(plan.opaque["records"][action]["las_posterior_entropy_bits"], tau)
            advance = m.advance if kernel == "las" else m.advance_fullsync
        before_state = state
        state = advance(state, plan, action, CounterRNG(key, "advance", step), ctx)
        if selective:
            mass = state.expansion_mass+math.fsum(state.retained.masses)
            if not ((state.terminal and mass == 0) or abs(mass-1) < 2e-6):
                raise ValueError("Invalid posterior mass")
        offset_before = offset
        resolved_before = len(before_state.resolved_bytes)
        offset += len(out)
        pieces.append(out)
        from .recording import scalar_mapping
        step_row = dict(step=step, action=int(action), bits=offset, consumed_bits=len(out),
                          consumed_bits_cumulative=offset, consumed_bits_before=offset_before,
                          kernel=kernel, retained=len(getattr(state, "retained", ())),
                          resolved_bytes=len(state.resolved_bytes),
                          visible_bytes_added=max(0, len(state.resolved_bytes)-resolved_before),
                          model_queries=len(paths),
                          planner_metrics=scalar_mapping(getattr(m, "last_metrics", {}), exclude_timing=True),
                          context_stats=scalar_mapping(ctx.stats),
                          **instrumentation(
                              plan, action, state, before_state, distributions,
                              ('filtered' if variant['method'] in
                                        ('basic_bytes_v1', 'mwis_bytes_v1', 'stepwise_chat_v1')
                                        else 'group_preserving' if (shared_sync or variant['method'] in
                                                                     ('las', 'fullsync', 'adaptive'))
                                        else None)))
        if variant['method'] == 'adaptive':
            step_row.update(adaptive_tau=variant.get('tau'),
                            adaptive_retained=(kernel == 'las'),
                            adaptive_gate_reason=('posterior_entropy_at_or_above_tau' if kernel == 'las'
                                                  else 'posterior_entropy_below_tau'))
        trace.append(step_row)
        if getattr(c, 'last_trace', None) is not None:
            from .recording import coder_trace
            trace[-1]['coder'] = coder_trace(c.last_trace)
            coder_diagnostic = dict(
                coder_name=variant['coder'], coder_mode='common_protocol',
                consumed_bits_this_step=len(out), consumed_bits_cumulative=offset,
                selected_action=int(action), action_probability=float(plan.probs[int(action)].item()),
                action_information_bits=(-math.log2(float(plan.probs[int(action)].item()))
                                        if float(plan.probs[int(action)].item()) > 0 else None),
            )
            trace[-1]['coder_diagnostics'] = coder_diagnostic
        if observer is not None:
            # An opt-in diagnostic sink receives existing objects after the
            # transition. It cannot supply an action or protocol randomness.
            observer(step=step, before=before_state, after=state, plan=plan,
                     distributions=distributions, action=int(action), consumed_bits=len(out), kernel=kernel)
        raw = state.resolved_bytes[:-len(codec.eos_bytes)] if state.terminal else state.resolved_bytes
        text, framing_status = render(codec, raw)
        if state.terminal:
            capped = token_mode and source.status(state)['token_limit_reached']
            reason = "token_limit" if capped else "natural_eos" if text is not None else "invalid_framing"
            break
        if sentence_mode:
            from .sentence import observation
            if observation(text, packet['language'])['sentence_goal_reached']:
                reason = 'sentence_goal'
                break
        if not natural and not sentence_mode and not token_mode and text is not None:
            if goal(text, stop, packet["language"], offset):
                reason = "text_goal"
                break
            if step >= stop["max_steps"]:
                reason = "bounded_limit"
                break
        if step >= watchdog:
            reason = "resource_watchdog"
            break
    raw = state.resolved_bytes[:-len(codec.eos_bytes)] if state.terminal else state.resolved_bytes
    text, framing_status = render(codec, raw)
    online = ''.join(pieces)
    tail = c.finalize_confirmed_prefix() if variant['coder'] == 'imec_prefix_exact' else ''
    delivered = online+tail
    expected_wire = stream if packet.get('message_delivery') != 'finite_frame' or not sending else stream+'0'*max(0,len(delivered)-len(stream))
    if sending and delivered != expected_wire[:len(delivered)]:
        raise ValueError('Terminal posterior prefix differs from supplied message')
    budget_status = source.status(state) if token_mode else None
    natural_eos = state.terminal and not (budget_status and budget_status['token_limit_reached'])
    from .recording import SINGLE_GENERATION_VERSION, canonical_digest
    result = dict(text=text, visible_bytes_hex=raw.hex(), bits=delivered, bits_count=len(delivered),
                  natural_eos=natural_eos, state_sha256=suite.digest(state), trace=trace, steps=len(trace),
                  stop_reason=reason, receiver_consumed=None if sending else target.done,
                  planning_seconds=planning_seconds, coder_seconds=coder_seconds,
                  **termination(reason, len(trace), watchdog, "protocol_decisions", framing_status),
                  recording_contract_version=SINGLE_GENERATION_VERSION,
                  trace_sha256=canonical_digest(trace), model_query_count=model_query_count,
                  protocol_decision_count=protocol_decision_count,
                  source_reference_forwards=(
                      getattr(source.reference, "forwards", None) - reference_forwards_before
                      if reference_forwards_before is not None and getattr(source.reference, "forwards", None) is not None else None),
                  total_visible_bytes=len(raw), visible_text_sha256=hashlib.sha256(raw).hexdigest(),
                  coder_name=variant['coder'], coder_mode=ctx.mode,
                  sampling_contract={name: getattr(source.config, name, None) for name in
                                     ('do_sample', 'temperature', 'top_k', 'top_p', 'typical_p',
                                      'repetition_penalty', 'min_p', 'num_beams')})
    if token_mode:
        result['token_budget'] = budget_status
    from stegatext_runtime.native_tokens import record_state
    result['native_token_sequence'] = record_state(
        source, prompt, state, natural_eos=natural_eos,
        synthetic_eos=bool(budget_status and budget_status['token_limit_reached']))
    result['native_sequence_sha256'] = canonical_digest(result['native_token_sequence'])
    result['native_output_token_count'] = result['native_token_sequence'].get('source_token_count')
    result['native_body_token_count'] = result['native_token_sequence'].get('body_token_count')
    result.update(online_committed_bits=offset, terminal_extra_prefix_bits=len(tail),
                  total_confirmed_bits=len(delivered), prefix_delivery_version=(
                      'terminal-lcp-v1' if variant['coder'] == 'imec_prefix_exact' else 'online-only-v1'),
                  nonempty_payload=bool(delivered))
    source_entropy = [float(value) for row in trace for value in row.get('source_entropy_bits', [])]
    group_entropies = [float(row['group_entropy_bits']) for row in trace if 'group_entropy_bits' in row]
    posterior_values = [float(row['posterior_entropy_bits']) for row in trace
                        if row.get('posterior_entropy_bits') is not None]
    result.update(source_entropy_total_bits=math.fsum(source_entropy),
                  source_entropy_per_decision=(math.fsum(source_entropy) / len(source_entropy)
                                               if source_entropy else None),
                  source_entropy_unavailable_steps=sum(
                      1 for row in trace if not row.get('source_entropy_bits')),
                  group_entropy_total_bits=math.fsum(group_entropies),
                  group_entropy_per_decision=(math.fsum(group_entropies) / len(group_entropies)
                                              if group_entropies else None),
                  posterior_information_total_bits=math.fsum(posterior_values),
                  posterior_information_per_decision=(math.fsum(posterior_values) / len(posterior_values)
                                                      if posterior_values else None),
                  strict_candidate_events=sum(1 for row in trace if (row.get('strict_candidate_count') or 0) > 0),
                  exact_duplicate_events=sum(1 for row in trace if (row.get('exact_candidate_count') or 0) > 1),
                  total_model_queries=model_query_count,
                  total_protocol_decisions=protocol_decision_count,
                  source_group_normalization_error=max(
                      [float(row.get('group_normalization_error', 0.0)) for row in trace] +
                      [float(summary.get(key, 0.0))
                       for row in trace for summary in row.get('source_distribution_summary', [])
                       for key in ('source_0_normalization_error', 'source_1_normalization_error')
                       if key in summary] + [0.0]),
                  filtered_path_kl_total_bits=math.fsum(
                      float(row['kl_method_to_source_bits']) for row in trace
                      if row.get('law_kind') == 'filtered' and row.get('kl_method_to_source_bits') is not None),
                  filtered_path_tv_total=math.fsum(
                      float(row['tv_distance']) for row in trace
                      if row.get('law_kind') == 'filtered' and row.get('tv_distance') is not None),
                  support_mismatch_steps=sum(1 for row in trace if row.get('support_mismatch') is True),
                  retained_frontier_max=max((int(row.get('retained_frontier_count_after', 0)) for row in trace), default=0),
                  candidate_identity_entropy_total_bits=math.fsum(
                      float(row['candidate_identity_entropy_bits']) for row in trace
                      if row.get('candidate_identity_entropy_bits') is not None),
                  strict_identity_entropy_total_bits=math.fsum(
                      float(row['strict_identity_entropy_bits']) for row in trace
                      if row.get('strict_identity_entropy_bits') is not None))
    if sending:
        result.update(message_input=dict(length_bits=len(stream), sha256=hashlib.sha256(stream.encode()).hexdigest(),
                      max_window_required_bits=max_required, message_exhausted=False,
                      padding_consumed=max(0, offset-len(stream)),
                      generator='SHAKE256:stegatext-experiment-message:v1:seed or explicit file',
                      random_domains='message SHAKE256; coder/advance HMAC CounterRNG by step',
                      lookahead_bound=('actual DisCop demand; each bit checked by coder' if variant['coder'] == 'discop_copy'
                                       else 'conservative max Huffman depth or exact iMEC window')),
                      sender_target_prefix_exact=True)
    else:
        result['receiver_prefix_exact'] = True  # support-confirmed; equality checked offline separately
    if hasattr(c, 'prefix_audit'):
        result['prefix_audit'] = c.prefix_audit()
        if delivered:
            result['prefix_audit']['zero_payload_reason'] = None
    if variant['coder'] == 'discop_copy':
        result['prefix_audit'] = dict(message_bit_order='sequential bits', first_unconfirmed_bit=offset,
            first_unconfirmed_values=[0,1], unresolved_head_support=[[0,'1/2'],[1,'1/2']],
            support_scope='first uninspected independent input bit', pending_blocks=None,
            known_noncontiguous_bits=0,
            zero_payload_reason='no_copy_distinguishing_node_on_selected_paths' if not offset else None)
    if 'prefix_audit' in result:
        audit = result['prefix_audit']
        result.update(first_unconfirmed_bit=audit.get('first_unconfirmed_bit'),
                      known_noncontiguous_bits=audit.get('known_noncontiguous_bits', 0),
                      zero_payload_reason=audit.get('zero_payload_reason'))
    if sentence_mode:
        from .sentence import observation
        result.update(observation(text, packet['language']))
    if hasattr(c, 'state'):
        result['coder_state'] = c.state()
    if not sending:
        if not target.done:
            raise ValueError("Receiver left unconsumed public bytes")
        if natural_eos != packet["natural_eos"]:
            raise ValueError("Receiver EOS differs from public termination")
        if public_termination(result) != packet["termination"]:
            raise ValueError("Receiver termination differs from public contract")
    return result


def send(source, case, config):
    from .engine import message_for
    from .randomness import case_nonce, protocol_descriptor
    variant, stop = case["variant"], config["stopping"]
    from stegatext_runtime.prompts import configure_prompt
    configure_prompt(source, case['prompt'].get('assistant_prefix'))
    prompt, seed = case["prompt"]["text"], case["seed"]
    if variant["kind"] == "ordinary":
        from .randomness import generate
        return None, generate(source, case, config, stopping=stop, language=case['prompt']['language'])
    carrier_nonce = case_nonce(config, case, source.manifest()) or nonce(source, prompt, seed)
    payload = message_for(config, seed, context=carrier_nonce)
    delivery = config["message"]["delivery"]
    wire = framing.encode(payload) if delivery == "finite_frame" else payload
    packet = dict(version="unified-experiment-v2", source=source.manifest(), prompt=prompt,
                  language=case["prompt"]["language"], seed=seed, nonce=carrier_nonce, variant=variant,
                  stopping=stop, engine_contract=contract(variant, stop), message_delivery=delivery)
    if 'assistant_prefix' in case['prompt']:
        packet['assistant_prefix'] = case['prompt']['assistant_prefix']
    descriptor = protocol_descriptor(config, case, source.manifest())
    if descriptor is not None:
        packet['experimental_randomness'] = descriptor
    key = shared_key(config, seed, carrier_nonce)
    result = planner(source, packet, key=key, stream=wire)
    if descriptor is not None:
        result['protocol_randomness'] = dict(descriptor, key_sha256=hashlib.sha256(key).hexdigest())
        result['message_randomness'] = dict(bits=len(payload), sha256=hashlib.sha256(payload.encode()).hexdigest(),
                                           domain='experimental-payload-v1', padded=False)
    packet.update(text=result["text"], natural_eos=result["natural_eos"], termination=public_termination(result))
    if result["text"] is None:
        packet["visible_bytes_hex"] = result["visible_bytes_hex"]
    return packet, result


def receive(source, packet, config):
    from .randomness import case_nonce, protocol_descriptor
    from stegatext_runtime.prompts import configure_prompt
    configure_prompt(source, packet.get('assistant_prefix'))
    if config["schema_version"] != 2 or packet["source"] != source.manifest():
        raise ValueError("Source or packet version differs from receiver contract")
    if packet["stopping"] != config["stopping"]:
        raise ValueError("Stopping differs from receiver configuration")
    if packet["engine_contract"] != contract(packet["variant"], packet["stopping"]):
        raise ValueError("Engine contract mismatch")
    if packet["message_delivery"] != config["message"]["delivery"]:
        raise ValueError("Message delivery contract mismatch")
    case = dict(seed=packet['seed'], prompt=dict(text=packet['prompt']), variant=packet['variant'])
    if 'assistant_prefix' in packet:
        case['prompt']['assistant_prefix'] = packet['assistant_prefix']
    expected = case_nonce(config, case, source.manifest()) or nonce(source, packet['prompt'], packet['seed'])
    descriptor = protocol_descriptor(config, case, source.manifest())
    if packet.get('experimental_randomness') != descriptor:
        raise ValueError('Experimental randomness contract mismatch')
    if packet["nonce"] != expected:
        raise ValueError("Carrier nonce mismatch")
    key = shared_key(config, packet['seed'], packet['nonce'])
    result = planner(source, packet, key=key)
    if descriptor is not None:
        result['protocol_randomness'] = dict(descriptor, key_sha256=hashlib.sha256(key).hexdigest())
    return result
