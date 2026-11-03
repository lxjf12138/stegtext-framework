"""Record an actual final history, without tokenizing the visible text."""
from __future__ import annotations

import hashlib
import json

VERSION = 'unified-native-token-sequence-v1'


def record_sequence(source, prompt_ids, output_ids, wire, *, natural_eos,
                    origin, synthetic_eos=False):
    codec = source.backend.codec
    prompt_ids, wire_ids = list(map(int, prompt_ids)), list(map(int, output_ids))
    eos = sorted(codec.eos_ids)
    if synthetic_eos and (natural_eos or not wire_ids or wire_ids[-1] not in eos):
        raise ValueError('Invalid synthetic token-budget delimiter')
    ids = wire_ids[:-1] if synthetic_eos else wire_ids
    terminal = bool(ids and ids[-1] in eos)
    if terminal != bool(natural_eos) or any(t in eos for t in ids[:-1]):
        raise ValueError('Native history EOS differs from termination')
    realization = codec.tokens_to_bytes(wire_ids)
    verified = realization == wire
    contract = source.manifest()
    return dict(version=VERSION, status='verified' if verified else 'unavailable_incomplete_history',
                origin=origin, source_sha256=contract.get('sha256') or hashlib.sha256(
                    json.dumps(contract, sort_keys=True).encode()).hexdigest(), model=contract.get('model'),
                revision=contract.get('revision'), tokenizer_revision=contract.get('tokenizer_revision'),
                prompt_token_ids=prompt_ids, output_token_ids=ids,
                wire_output_token_ids=wire_ids, eos_token_ids=eos,
                natural_eos=bool(natural_eos), synthetic_boundary_eos=bool(synthetic_eos),
                body_token_count=len(ids)-int(terminal), source_token_count=len(ids),
                wire_sha256=hashlib.sha256(wire).hexdigest(),
                wire_reconstruction_verified=verified,
                scope='one realized final history; not a posterior marginal or retokenized visible text')


def record_state(source, prompt, state, *, natural_eos, synthetic_eos=False):
    path = state.base_path if hasattr(state, 'base_path') else state.path
    full, prefix = tuple(path.token_ids()), tuple(prompt.token_ids)
    if full[:len(prefix)] != prefix:
        raise ValueError('Final native history does not extend the actual prompt')
    return record_sequence(source, prefix, full[len(prefix):], state.resolved_bytes,
                           natural_eos=natural_eos, synthetic_eos=synthetic_eos,
                           origin='runtime_final_representative_history')
