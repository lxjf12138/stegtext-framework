"""One source of prompt, generation settings and probabilities for both models."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from research_notes.inference_conformance.hf_reference import HFReference
from stegtext.v2.core.types import TokenDistribution
from stegtext.v2.model.hf_backend import HFBackend
from stegtext.v2.model.prompt import ChatPromptPolicy, PlainCompletionPromptPolicy

from .models import MODEL_IDS, prompt_mode
from .prompts import AssistantPrefillPromptPolicy

ROOT = Path(__file__).resolve().parents[1]


class ValidityMask(LogitsProcessor):
    def __init__(self, mask):
        self.mask = mask

    def __call__(self, input_ids, scores):
        return scores.masked_fill(~self.mask, -torch.inf)


class Source:
    def __init__(self, backend, spec, model_name, profile="checkpoint"):
        if profile not in ("checkpoint", "stego"):
            raise ValueError("profile must be checkpoint or stego")
        self.backend = backend
        self.model = backend.model
        self.tokenizer = backend.tokenizer
        self.device = backend.device
        self.spec = spec
        self.model_name = model_name
        self.profile = profile
        self.prompt_mode = prompt_mode(model_name)
        expected = ChatPromptPolicy if self.prompt_mode == "chat" else PlainCompletionPromptPolicy
        if not isinstance(backend.prompt_policy, expected):
            raise ValueError(f"{model_name} requires an explicit {self.prompt_mode} policy")
        if self.prompt_mode == "chat" and backend.prompt_policy.template_options != spec["template_options"]:
            raise ValueError("Chat options differ from the locked source protocol")
        self.config = copy.deepcopy(self.model.generation_config)
        self.overrides = {}
        if profile == "stego" and not self.config.do_sample:
            self.overrides["do_sample"] = True
            self.config.do_sample = True
        eos = self.config.eos_token_id
        self.eos_ids = frozenset(eos if isinstance(eos, list) else [eos])
        if self.eos_ids != backend.codec.eos_ids:
            raise ValueError("Codec EOS differs from checkpoint EOS")
        self.validity = None
        if profile == "stego":
            self.validity = torch.zeros(self.model.config.vocab_size, dtype=torch.bool, device=self.device)
            self.validity[list(self.tokenizer.get_vocab().values())] = True
            table = backend.codec.token_bytes_table
            self.validity[:len(table)] &= torch.tensor([bool(x) for x in table], device=self.device)
        self.reference = HFReference(backend, self.config, validity_mask=self.validity)
        self.prompt = None
        self.query_log = None  # Optional diagnostic hook; never needed by the receiver.

    @classmethod
    def load(cls, model_name, profile="checkpoint"):
        if model_name not in MODEL_IDS:
            raise ValueError(f"Unknown model {model_name!r}")
        lock = json.loads((ROOT/"environment/models.lock.json").read_text())
        spec = next(s for s in lock["models"] if s["repo_id"] == MODEL_IDS[model_name])
        mode = prompt_mode(model_name)
        if spec.get("prompt_mode", "plain_completion") != mode:
            raise ValueError("Model lock prompt protocol disagrees with model catalog")
        policy = ChatPromptPolicy(template_options=spec["template_options"]) if mode == "chat" else PlainCompletionPromptPolicy()
        backend = HFBackend.from_pretrained(
            str(ROOT/spec["local_path"]), device="cuda:0", physical_gpu=0,
            dtype="bfloat16", device_map={"": "cuda:0"}, cache_size=0,
            prompt_policy=policy,
            tokenizer_kwargs={"local_files_only": True},
            model_kwargs={"local_files_only": True, "attn_implementation": "sdpa"})
        backend.validate_tokenizer_contract()
        return cls(backend, spec, model_name, profile)

    def with_profile(self, profile):
        """Reuse model weights, but give each source its own KV/config state."""
        return Source(self.backend, self.spec, self.model_name, profile)

    def bind(self, prompt):
        self.prompt = self.backend.render_prompt(prompt)
        if not self.prompt.token_ids:
            raise ValueError("A nonempty tokenized prompt is required")
        self.reference.bind_prompt(self.prompt.token_ids)
        return self.prompt

    def configure_assistant_prefix(self, prefix=None):
        if self.prompt_mode != 'chat':
            if prefix is not None:
                raise ValueError('Assistant prefill requires an Instruct chat source')
            return
        options = self.spec['template_options']
        self.backend.prompt_policy = (ChatPromptPolicy(template_options=options) if prefix is None else
            AssistantPrefillPromptPolicy(prefix, template_options=options))

    def assert_sampling_contract(self):
        expected = getattr(self, 'expected_sampling', {})
        for label, config in [('source', self.config), ('reference', self.reference.config)]:
            if any(getattr(config, k) != v for k,v in expected.items()):
                raise ValueError(f'{label} sampling kernel differs from explicit experiment contract')

    def probabilities(self, token_ids):
        if getattr(self, 'expected_sampling', None):
            self.assert_sampling_contract()
        ids = tuple(token_ids)
        probabilities = self.reference.probabilities(ids)
        if self.query_log is not None:
            self.query_log(ids, probabilities)
        return probabilities

    def retokenize_text(self, prompt, text):
        """Receiver-view BPE on the public rendered prefix plus visible answer.

        This intentionally retains tokenization ambiguity; no sender history is
        supplied. A chat prefix already contains any required BOS token.
        """
        p = self.backend.render_prompt(prompt)
        if self.prompt_mode == "chat":
            policy = self.backend.prompt_policy
            prefix = (policy.render_text(self.tokenizer, prompt)
                      if isinstance(policy, AssistantPrefillPromptPolicy) else
                      self.tokenizer.apply_chat_template(
                          [{"role": "user", "content": str(prompt)}], tokenize=False,
                          add_generation_prompt=True, **policy.template_options))
            whole = self.tokenizer.encode(prefix + text, add_special_tokens=False)
        else:
            whole = self.tokenizer.encode(prompt + text, add_special_tokens=True)
        if whole[:len(p.token_ids)] != list(p.token_ids):
            raise ValueError("Retokenization changed the public prompt boundary")
        return whole[len(p.token_ids):]

    def distribution(self, path):
        probabilities = self.probabilities(path.token_ids())
        # Keep the full head through HF processing AND softmax. Removing only
        # zero padding afterwards cannot change the categorical law.
        n = self.backend.codec.vocab_size
        if torch.count_nonzero(probabilities[n:]).item():
            raise ValueError("Unrenderable head IDs have mass; use explicit stego profile")
        return TokenDistribution(probabilities[:n])

    def hf_processors(self):
        return LogitsProcessorList([] if self.validity is None else [ValidityMask(self.validity)])

    def manifest(self):
        configuration = self.config.to_dict()
        contract = {
            "version": 1, "model": self.model_name, "repo_id": self.spec["repo_id"],
            "revision": self.spec["revision"], "tokenizer_revision": self.spec["tokenizer_revision"],
            "profile": self.profile, "overrides": self.overrides,
            "asserted_sampling": getattr(self, "expected_sampling", None),
            "generation_config": configuration,
            "prompt_mode": "plain_completion", "add_special_tokens": True,
            "validity_mask": "byte_codec" if self.validity is not None else "none",
            "dtype": "bfloat16", "attention": "sdpa", "transformers": "4.57.6",
            "history_schedule": "fixed_prompt_prefill_then_single_token_replay",
            "hf_reference_sha256": hashlib.sha256((ROOT/"research_notes/inference_conformance/hf_reference.py").read_bytes()).hexdigest(),
        }
        if self.prompt_mode == "chat":
            contract.update(version=2, prompt_mode="chat", add_special_tokens=False,
                add_generation_prompt=True, message_roles=["user"],
                template_options=self.backend.prompt_policy.template_options,
                chat_template_sha256=hashlib.sha256(self.tokenizer.chat_template.encode()).hexdigest())
            if isinstance(self.backend.prompt_policy, AssistantPrefillPromptPolicy):
                contract.update(version=3, add_generation_prompt=False, continue_final_message=True,
                    message_roles=['user', 'assistant'],
                    assistant_prefix_sha256=hashlib.sha256(
                        self.backend.prompt_policy.assistant_prefix.encode()).hexdigest())
        encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        return {**contract, "sha256": hashlib.sha256(encoded).hexdigest()}

    @torch.inference_mode()
    def generate(self, prompt, *, max_new_tokens=None, seed, stopping=None, language='en', generator=None):
        if stopping is not None:
            watchdog = stopping["watchdog_generated_tokens"]
            if stopping['mode'] == 'token_budget':
                max_new_tokens = min(watchdog, stopping['max_new_tokens'])
            else:
                max_new_tokens = watchdog if stopping["mode"] in ('natural_eos_only','sentence_boundary') else min(watchdog, stopping["max_steps"])
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        p = self.bind(prompt)
        reference_forwards_before = getattr(self.reference, 'forwards', None)
        ids = list(p.token_ids)
        generated = []
        sampling_rows = []
        if generator is None:
            torch.manual_seed(seed)
        elif generator.initial_seed() != seed:
            raise ValueError('Generator seed does not match the recorded sampling seed')
        for _ in range(max_new_tokens):
            probs = self.probabilities(ids)
            from .experiments.recording import probability_summary, SINGLE_GENERATION_VERSION
            step_row = probability_summary(probs, prefix='source')
            step_row.update(step=len(sampling_rows) + 1,
                positive_source_support=step_row['source_support_count'],
                source_entropy_bits=step_row['source_entropy_bits'],
                probability_space='selected_sampler_reference',
                source_to_method_kl_bits=0.0, method_to_source_kl_bits=0.0,
                kl_method_to_source_bits=0.0, kl_source_to_method_bits=0.0,
                tv_distance=0.0, support_mismatch=False,
                finite_forward_kl=True, reverse_kl_infinite=False,
                final_visible_text_kl_status='requires_multi_sample_estimate')
            token = int(torch.multinomial(probs[None], num_samples=1, generator=generator)[0, 0]) if self.config.do_sample else int(probs.argmax())
            generated.append(token)
            step_row.update(token_id=token,
                            selected_token_probability=float(probs[token].item()),
                            selected_token_information_bits=(
                                -math.log2(float(probs[token].item())) if float(probs[token].item()) > 0 else None),
                            selected_is_argmax=bool(token == int(probs.argmax().item())),
                            sampling_randomness='local_generator' if generator is not None else 'global_torch')
            selected = float(probs[token].item())
            step_row['selected_token_log_probability'] = math.log(selected) if selected > 0 else None
            sampling_rows.append(step_row)
            ids.append(token)
            if token in self.eos_ids:
                break
            if stopping is not None and stopping['mode'] == 'sentence_boundary':
                from .experiments.stopping import render
                from .experiments.sentence import observation
                text, _ = render(self.backend.codec, self.backend.codec.tokens_to_bytes(generated))
                if observation(text, language)['sentence_goal_reached']:
                    break
        ended = generated[-1] in self.eos_ids
        result = {
            "source": self.manifest(), "prompt": prompt, "prompt_token_ids": list(p.token_ids),
            "token_ids": generated, "natural_eos": ended,
            "stop_reason": "eos" if ended else "max_new_tokens",
            "max_new_tokens": max_new_tokens, "seed": seed,
            "recording_contract_version": SINGLE_GENERATION_VERSION,
            "sampling_contract": {name: getattr(self.config, name, None) for name in
                                  ('do_sample', 'temperature', 'top_k', 'top_p', 'typical_p',
                                   'repetition_penalty', 'min_p', 'num_beams')},
            "text": self.tokenizer.decode(generated[:-1] if ended else generated,
                skip_special_tokens=False, clean_up_tokenization_spaces=False),
        }
        if stopping is not None:
            from .experiments.stopping import render, termination
            raw = self.backend.codec.tokens_to_bytes(generated[:-1] if ended else generated)
            text, framing = render(self.backend.codec, raw)
            reason = "natural_eos" if ended else "bounded_limit" if stopping["mode"] == "bounded_operational" and len(generated) >= stopping["max_steps"] else "resource_watchdog"
            if stopping['mode'] == 'token_budget':
                capped = not ended and len(generated) == stopping['max_new_tokens']
                if capped:
                    reason = 'token_limit'
                result['token_budget'] = dict(max_new_tokens=stopping['max_new_tokens'],
                    source_terminal=ended or capped, token_limit_reached=capped,
                    representative_generated_tokens=len(generated), unit='source_generated_tokens',
                    natural_eos_counts=True, internal_boundary_eos_counts=False)
            if ended and text is None:
                reason = "invalid_framing"
            if stopping['mode'] == 'sentence_boundary':
                from .experiments.sentence import observation
                info = observation(text, language)
                if not ended and info['sentence_goal_reached']:
                    reason = 'sentence_goal'
                result.update(info)
            result.update(text=text, visible_bytes_hex=raw.hex(), stop_reason=reason,
                          **termination(reason, len(generated), watchdog, "generated_tokens", framing))
            result['ordinary_sampling'] = sampling_rows
        else:
            result['ordinary_sampling'] = sampling_rows
        entropies = [float(row['source_entropy_bits']) for row in sampling_rows]
        body_rows = sampling_rows[:-1] if ended else sampling_rows
        result.update(source_entropy_total_bits=math.fsum(entropies),
                      source_entropy_per_generated_token=(math.fsum(entropies) / len(entropies)
                                                         if entropies else None),
                      source_entropy_body_total_bits=math.fsum(float(row['source_entropy_bits']) for row in body_rows),
                      source_entropy_body_per_token=(math.fsum(float(row['source_entropy_bits']) for row in body_rows) /
                                                    len(body_rows) if body_rows else None),
                      source_support_min=min((row['source_support_count'] for row in sampling_rows), default=0),
                      source_support_max=max((row['source_support_count'] for row in sampling_rows), default=0),
                      generated_token_count=len(generated), body_token_count=len(generated)-int(ended))
        from .native_tokens import record_sequence
        result['native_token_sequence'] = record_sequence(
            self, p.token_ids, generated, self.backend.codec.tokens_to_bytes(generated),
            natural_eos=ended, origin='runtime_sampled_sequence')
        from .experiments.recording import canonical_digest
        result['native_sequence_sha256'] = canonical_digest(result['native_token_sequence'])
        result['model_query_count'] = len(sampling_rows)
        result['protocol_decision_count'] = len(sampling_rows)
        result['source_reference_forwards'] = (
            getattr(self.reference, 'forwards', None) - reference_forwards_before
            if reference_forwards_before is not None and getattr(self.reference, 'forwards', None) is not None else None)
        return result
