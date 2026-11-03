"""Independent byte-prefix oracles, upstream parity, and actual chat contracts."""
from fractions import Fraction
from itertools import product
from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import torch

from stegatext_runtime.experiments.baseline_adapters import (
    BytePrefixBaseline, ChatStepwise, basic_indices, byte_classes, mwis_indices,
)
from stegatext_runtime.experiments.engine import send, receive
from stegtext.v2.core.types import MethodContext, PathRef, PromptState, TokenDistribution, VisibleTarget
from tests.finite_source import FiniteSource, ReplayUniform


class Codec:
    eos_bytes = b"\x00\x00"
    def __init__(self, pieces, eos=()):
        self.pieces, self.eos = pieces, set(eos)
    def token_bytes(self, token):
        return self.pieces[token]
    def is_eos(self, token):
        return token in self.eos
    def next_token(self, *args):
        raise AssertionError("Prefix-free byte receiver must not retokenize")


def context(codec, mode="encode"):
    return MethodContext(codec, PromptState(PathRef.root(), (), b""), mode)


@pytest.mark.parametrize("strategy", ["basic", "mwis"])
def test_eos_and_exact_duplicate_masses_use_the_common_transport(strategy):
    codec = Codec({0: b"A", 1: b"A", 2: b"\0\0", 3: b"\0\0"}, eos=[2, 3])
    source = TokenDistribution(torch.tensor([.125, .125, .5, .25], dtype=torch.float64))
    method = BytePrefixBaseline(strategy)
    ctx = context(codec)
    plan = method.plan(method.start(ctx.prompt), [source], ctx)
    law = dict(zip(plan.opaque["keys"], plan.probs.tolist()))
    assert law == {b"\0\0": .75, b"A": .25}
    assert plan.opaque["baseline_contract"]["retained_mass"] == 1.


def test_basic_matches_fraction_strict_prefix_oracle_and_upstream_on_distinct_pieces():
    from research_notes.full_validation.suite import load_pools
    load_pools()
    from tokenization_consistency.pools import _pool_basic
    keys = [b"a", b"ab", b"abc", b"ac", b"b", b"bc", b"\0\0"]
    weights = [Fraction(n, 128) for n in [32, 16, 8, 4, 2, 1, 65]]
    codec = Codec(dict(enumerate(keys)), eos=[6])
    source = TokenDistribution(torch.tensor([float(p) for p in weights], dtype=torch.float64))
    ctx = context(codec)
    method = BytePrefixBaseline("basic")
    plan = method.plan(method.start(ctx.prompt), [source], ctx)
    valid = [i for i, a in enumerate(keys) if not any(a != b and b.startswith(a) for b in keys)]
    retained = sum(weights[i] for i in valid)
    oracle = {keys[i]: weights[i] / retained for i in valid}
    actual = dict(zip(plan.opaque["keys"], plan.probs.tolist()))
    assert actual.keys() == oracle.keys()
    assert all(abs(actual[k] - float(p)) < 1e-15 for k, p in oracle.items())
    class View:
        def batch_decode(self, sequences, **kwargs):
            return [keys[int(s[0])].decode("latin1") for s in sequences]
    original = _pool_basic(View(), [], source.probs, len(keys))
    assert set(original.token_ids.tolist()) == set(valid)


def test_optimized_mwis_is_identical_to_upstream_graph_and_level_dp():
    from research_notes.full_validation.suite import load_pools
    load_pools()
    from tokenization_consistency.pools import maximum_weight_independent_set
    rng = random.Random(9471)
    universe = [bytes(word) for n in range(1, 6) for word in product(b"ab", repeat=n)] + [b"\0\0"]
    for _ in range(120):
        keys = rng.sample(universe, rng.randint(1, 35))
        weights = [rng.randint(1, 100) / 128 for _ in keys]
        adjacency = [[j for j, b in enumerate(keys)
                      if i != j and (a.startswith(b) or b.startswith(a))]
                     for i, a in enumerate(keys)]
        assert mwis_indices(keys, weights) == maximum_weight_independent_set(adjacency, weights)


@pytest.mark.parametrize("strategy", ["basic", "mwis"])
def test_receiver_handles_a_suffix_inside_a_utf8_character_without_retokenization(strategy):
    codec = Codec({0: b"\xb8\xad", 1: b"z", 2: b"\0\0"}, eos=[2])
    source = TokenDistribution(torch.tensor([.5, .25, .25], dtype=torch.float64))
    ctx = context(codec, "decode")
    method = BytePrefixBaseline(strategy)
    state = method.start(ctx.prompt)
    plan = method.plan(state, [source], ctx)
    target = VisibleTarget(b"\xb8\xad\0\0")
    action = method.observe(state, plan, target, ctx)
    assert plan.opaque["keys"][action] == b"\xb8\xad"
    assert target.remaining == b"\0\0"


@pytest.mark.parametrize("name", ["basic_bytes_v1", "mwis_bytes_v1"])
@pytest.mark.parametrize("coder", ["discop_copy", "imec_prefix_exact"])
@pytest.mark.parametrize("fixture", [{}, {"multiple_eos": True}, {"chinese": True}])
def test_independent_byte_baseline_sender_receiver(tmp_path, name, coder, fixture):
    from tests.test_all_method_randomness import fixture as inputs
    cfg, case = inputs(tmp_path)
    case["variant"].update(method=name, coder=coder)
    if coder == "imec_prefix_exact":
        case["variant"]["coder_parameters"] = dict(block_size=6, window_size=3)
    packet, sender = send(FiniteSource(**fixture), case, cfg)
    receiver = receive(FiniteSource(**fixture), packet, cfg)
    assert sender["bits"] == receiver["bits"]
    assert sender["native_token_sequence"] == receiver["native_token_sequence"]
    assert sender["trace"] == receiver["trace"]


@pytest.fixture(scope="module", params=["qwen_instruct", "llama_instruct"])
def real_chat(request):
    import json
    from transformers import AutoTokenizer
    from stegtext.v2.model.tokenizer_codec import TokenizerCodec
    root = Path(__file__).resolve().parents[1]
    model = "Qwen3-4B-Instruct-2507" if request.param == "qwen_instruct" else "Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(root / "models" / model, local_files_only=True)
    eos = json.loads((root / "models" / model / "generation_config.json").read_text())["eos_token_id"]
    eos = [eos] if isinstance(eos, int) else eos
    codec = TokenizerCodec(tokenizer, eos)
    ids = tokenizer.apply_chat_template([dict(role="user", content="Describe a park." if request.param == "llama_instruct" else "描述公园。")],
                                       tokenize=True, add_generation_prompt=True)
    path = PathRef.root()
    for i in ids:
        path = path.append(i)
    prompt = PromptState(path, tuple(ids), b"")
    return SimpleNamespace(tokenizer=tokenizer), codec, prompt


def test_stepwise_calls_upstream_predicate_on_all_support_and_preserves_chat(real_chat):
    source, codec, prompt = real_chat
    method = ChatStepwise(source)
    state = method.start(prompt)
    candidates = list(range(96))
    raw = torch.zeros(codec.vocab_size, dtype=torch.float64)
    for rank, token in enumerate(candidates):
        raw[token] = (97 - rank) / 10000
    for token in codec.eos_ids:
        raw[token] = .1
    raw /= raw.sum()
    plan = method.plan(state, [TokenDistribution(raw)], MethodContext(codec, prompt, "encode"))
    contract = plan.opaque["baseline_contract"]
    assert contract["tested_data_candidates"] == 96
    assert not contract["expands_sampler_support"]
    kept = {i for group in plan.opaque["members"] for i in group}
    for token in candidates:
        combined = list(prompt.token_ids) + [token]
        text = source.tokenizer.decode(combined, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        expected = source.tokenizer(text, add_special_tokens=False)["input_ids"] == combined
        assert (token in kept) == expected
    assert kept & set(candidates[32:])
    assert codec.eos_ids <= kept


def test_stepwise_accepts_augmented_eos_without_deleting_it(real_chat):
    source, codec, prompt = real_chat
    method = ChatStepwise(source)
    raw = torch.zeros(codec.vocab_size, dtype=torch.float64)
    for i in codec.eos_ids:
        raw[i] = 1 / len(codec.eos_ids)
    plan = method.plan(method.start(prompt), [TokenDistribution(raw)], MethodContext(codec, prompt, "encode"))
    assert plan.probs.tolist() == [1.]
    assert set(plan.opaque["members"][0]) == set(codec.eos_ids)


@pytest.mark.parametrize("coder", ["discop_copy", "imec_prefix_exact"])
def test_shared_syncpool_is_fullsync_endpoint_with_independent_receiver(tmp_path, coder):
    from tests.test_all_method_randomness import fixture as inputs
    from stegatext_runtime.experiments.engine_v2 import planner
    cfg, case = inputs(tmp_path)
    variant = dict(id="syncpool", kind="planner", method="syncpool", coder=coder,
                   implementation="shared_planner_v1")
    if coder == "imec_prefix_exact":
        variant["coder_parameters"] = dict(block_size=6, window_size=3)
    packet = dict(prompt="", variant=variant, stopping=cfg["stopping"],
                  language="en", message_delivery="prefix_stream")
    bits, key = "01100101" * 1000, b"same-semantic-endpoint-1234567890"
    shared = planner(FiniteSource(), packet, key=key, stream=bits)
    endpoint = dict(packet, variant=dict(variant, method="fullsync"))
    endpoint["variant"].pop("implementation")
    reference = planner(FiniteSource(), endpoint, key=key, stream=bits)
    assert shared["trace"] == reference["trace"]
    assert shared["bits"] == reference["bits"]
    packet.update(text=shared["text"], natural_eos=shared["natural_eos"],
                  termination={k: shared[k] for k in ["terminal_reason", "censored", "framing_status", "watchdog", "token_budget"]})
    receiver = planner(FiniteSource(), packet, key=key)
    assert receiver["bits"] == shared["bits"]
    assert receiver["native_token_sequence"] == shared["native_token_sequence"]


def test_direct_coder_correctness_does_not_fix_tokenization_ambiguity():
    from stegatext_runtime.discop import DisCopSession
    coder = DisCopSession()
    probabilities = (.5, .5)  # actions realize "a" and "ab"; cover can emit "a" then "b".
    consumed, action = coder.encode(probabilities, "0", rng=ReplayUniform([Fraction(1, 4)]))
    assert coder.decode(probabilities, action, rng=ReplayUniform([Fraction(1, 4)])) == consumed
    assert coder.decode(probabilities, 1 - action, rng=ReplayUniform([Fraction(1, 4)])) != consumed
