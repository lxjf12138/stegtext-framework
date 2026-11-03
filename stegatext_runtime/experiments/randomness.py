"""Reproducible ordinary draws, separated from shared stego protocol keys.

Hash-derived PRNG streams avoid reusing the same categorical noise across
conditions. They do not constitute a mathematical proof of random independence.
"""
from __future__ import annotations

import hashlib

import torch

from .config import fingerprint

LEGACY = 'shared_replicate_v1'
INDEPENDENT = 'independent_conditions_v1'
POLICIES = {LEGACY, INDEPENDENT}
DOMAIN = 'stegatext:ordinary-sampling:independent-conditions:v1'
CASE_POLICY = 'independent_cases_v1'
CASE_DOMAIN = 'stegatext:all-methods:independent-cases:v1'


def independent_cases(config):
    return config.get('randomness', {}).get('stream_policy') == CASE_POLICY


def case_nonce(config, case, source_manifest):
    """Semantic case identity; labels and matrix slicing cannot change it."""
    if not independent_cases(config):
        return None
    spec = config['randomness']
    return fingerprint(dict(domain=CASE_DOMAIN, experiment_id=spec['experiment_id'],
        root_sha256=spec['sha256'], source_sha256=source_manifest['sha256'],
        prompt_sha256=hashlib.sha256(case['prompt']['text'].encode()).hexdigest(),
        stopping=config['stopping'], schema_version=config['schema_version'],
        replicate_seed=case['seed'], variant={k:v for k,v in case['variant'].items() if k != 'id'}))


def protocol_descriptor(config, case, source_manifest):
    value = case_nonce(config, case, source_manifest)
    return (dict(policy=CASE_POLICY, experiment_id=config['randomness']['experiment_id'],
                 condition_sha256=value, message_domain='experimental-payload-v1',
                 coder_domain='coder', representative_domain='advance') if value else None)


def policy(config):
    return config.get('inputs', {}).get('ordinary_seed_policy', LEGACY)


def descriptor(config, case, source_manifest):
    selected = policy(config)
    if selected not in POLICIES:
        raise ValueError('Unknown ordinary_seed_policy')
    if selected == LEGACY:
        return dict(policy=selected, replicate_seed=case['seed'], torch_seed=case['seed'],
                    generator='global_torch_legacy')
    prompt_hash = hashlib.sha256(case['prompt']['text'].encode()).hexdigest()
    condition = dict(domain=DOMAIN, source_sha256=source_manifest['sha256'],
                     prompt_sha256=prompt_hash, stopping=config['stopping'],
                     schema_version=config['schema_version'], replicate_seed=case['seed'])
    if independent_cases(config):
        condition.update(experiment_id=config['randomness']['experiment_id'],
                         experiment_root_sha256=config['randomness']['sha256'])
    digest = fingerprint(condition)
    return dict(policy=selected, domain=DOMAIN, replicate_seed=case['seed'],
                torch_seed=int(digest[:16], 16) & ((1 << 63)-1),
                condition_sha256=digest, source_sha256=source_manifest['sha256'],
                prompt_sha256=prompt_hash, generator='local_torch_generator')


def generate(source, case, config, **kwargs):
    """Keep the old API/RNG path exactly unless the new policy is explicit."""
    from stegatext_runtime.prompts import configure_prompt
    configure_prompt(source, case['prompt'].get('assistant_prefix'))
    if policy(config) == LEGACY:
        return source.generate(case['prompt']['text'], seed=case['seed'], **kwargs)
    stream = descriptor(config, case, source.manifest())
    seed = stream['torch_seed']
    generator = torch.Generator(device=getattr(source, 'device', 'cpu')).manual_seed(seed)
    result = source.generate(case['prompt']['text'], seed=seed, generator=generator, **kwargs)
    result['sampling_randomness'] = stream
    return result
