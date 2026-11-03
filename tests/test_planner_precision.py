"""Keep reachable positive LAS groups when source atoms arrive as float32."""
from fractions import Fraction as F
from types import SimpleNamespace

import pytest
import torch

from stegatext_runtime.discop import DisCopSession
from stegatext_runtime.experiments.engine import method
from stegatext_runtime.imec import coupling
from stegtext.v2.core.types import MethodContext, TokenDistribution
from stegtext.v2.model.toy_backend import ToyBackend, ToyBackendConfig
from stegtext.v2.model.tokenizer_codec import EOS_STEGA
from tests.finite_source import ReplayUniform


@pytest.fixture
def reachable_small_group():
    backend = ToyBackend(ToyBackendConfig(
        {0: b"x", 1: b"xc", 2: b"a", 3: b"b", 4: EOS_STEGA}, {}, eos_ids=[4]))
    backend.codec_index = backend.index
    planner = method(SimpleNamespace(backend=backend), "las")
    ctx = MethodContext(backend.codec, backend.render_prompt(""), "encode")
    state = planner.start(ctx.prompt)
    first = TokenDistribution(torch.tensor([1e-20, 1., 0., 0., 0.], dtype=torch.float32))
    plan = planner.plan(state, [first], ctx)
    assert [r["key"] for r in plan.opaque["records"]] == [b"x"]
    state = planner.advance(state, plan, 0, ReplayUniform([F(1, 4)]), ctx)
    assert state.base_path.token_ids() == (0,)
    assert state.expansion_mass > 0
    second = TokenDistribution(torch.tensor([0., 0., 1e-30, 1., 0.], dtype=torch.float32))
    plan = planner.plan(state, [second], ctx)
    # Same supplied atom values; only their container precision differs.
    wide = planner.plan(state, [TokenDistribution(second.probs.double())], ctx)
    keys = [r["key"] for r in plan.opaque["records"]]
    assert keys == [b"a", b"b", b"c"]
    return plan, wide, keys.index(b"a"), state.expansion_mass * float(second.probs[2])


def test_float32_source_keeps_positive_group_after_normalization(reachable_small_group):
    plan, wide, action, expected_mass = reachable_small_group
    assert plan.opaque["records"][action]["group_mass"] == expected_mass
    assert expected_mass == pytest.approx(1e-50, rel=1e-6, abs=0)
    assert float(plan.probs[action]) > 0
    assert bool((plan.probs > 0).all())
    assert float(plan.probs.sum()) == pytest.approx(1.)
    torch.testing.assert_close(plan.probs.double(), wide.probs, rtol=0, atol=0)


def test_discop_can_encode_and_decode_the_small_group(reachable_small_group):
    plan, _, action, _ = reachable_small_group
    sender, receiver = DisCopSession(), DisCopSession()
    # Force the tiny branch at both tree nodes; this checks reachability and
    # recovery, not equality between a finite RNG grid and ideal probabilities.
    bits, selected = sender.encode(plan.probs.cpu(), "00", rng=ReplayUniform([F(0), F(0)]))
    assert selected == action
    assert sender.last_trace["selected_probability"] > 0
    assert receiver.decode(plan.probs.cpu(), selected, rng=ReplayUniform([F(0), F(0)])) == bits


@pytest.mark.parametrize("strategy", ["greedy", "prefix"])
def test_imec_coupling_keeps_the_small_group_marginal(reachable_small_group, strategy):
    plan, _, action, _ = reachable_small_group
    p, q, rows, columns = coupling([1] * 64, plan.probs.cpu(), strategy)
    assert q[action] > 0
    supplied = tuple(F(float(v)) for v in plan.probs)
    assert q[action] == supplied[action] / sum(supplied)
    assert sum(columns[action].values()) == q[action]
    assert all(sum(row.values()) == mass for row, mass in zip(rows, p))
    assert all(sum(col.values()) == mass for col, mass in zip(columns, q))
