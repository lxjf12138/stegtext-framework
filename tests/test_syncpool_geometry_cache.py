"""Support changes must invalidate cached pools without caching their masses."""
from types import SimpleNamespace

import pytest
import torch

from stegtext.v2.core.types import TokenDistribution
from stegtext.v2.methods.syncpool import SyncPoolMethod
from stegtext.v2.model.toy_backend import ToyBackend, ToyBackendConfig
from stegtext.v2.model.tokenizer_codec import EOS_STEGA


def context(tokens=None):
    backend = ToyBackend(ToyBackendConfig(
        tokens or {0: b'a', 1: b'ab', 2: b'ac', 3: b'a', 4: b'b', 5: EOS_STEGA}, {}, eos_ids=[5]))
    return SimpleNamespace(codec=backend.codec)


def plan(method, ctx, weights, *, device='cpu', dtype=torch.float64):
    return method.plan(None, [TokenDistribution(torch.tensor(weights, device=device, dtype=dtype))], ctx)


def test_same_support_reuses_geometry_but_recomputes_masses(monkeypatch):
    ctx, method = context(), SyncPoolMethod()
    first = plan(method, ctx, [1, 2, 3, 4, 5, 6])
    def unexpected_lookup(token):
        raise AssertionError('A warm geometry should not decode token bytes again')
    monkeypatch.setattr(ctx.codec, 'token_bytes', unexpected_lookup)
    second = plan(method, ctx, [2, 3, 4, 5, 6, 7])
    assert first.opaque['members'] == second.opaque['members']
    masses = {frozenset(ids): float(p) for ids, p in zip(second.opaque['members'], second.probs)}
    assert masses[frozenset([0, 1, 2, 3])] == pytest.approx(14 / 27)
    assert not torch.equal(first.probs, second.probs)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_support_changes_eviction_and_dtype_match_fresh_plans(device):
    ctx, method = context(), SyncPoolMethod()
    # Vary support past the cache capacity, then revisit evicted entries.
    for mask in [*range(1, 20), *range(1, 20)]:
        weights = [float(bool(mask & (1 << i))) for i in range(5)] + [1.]
        for dtype in [torch.float32, torch.float64]:
            actual = plan(method, ctx, weights, device=device, dtype=dtype)
            fresh = plan(SyncPoolMethod(), ctx, weights, device=device, dtype=dtype)
            assert actual.opaque['keys'] == fresh.opaque['keys']
            assert actual.opaque['members'] == fresh.opaque['members']
            assert torch.equal(actual.probs, fresh.probs)
        assert len(method._geometry) <= 8
    split = plan(method, ctx, [0, 1, 1, 0, 1, 1], device=device)
    joined = plan(method, ctx, [1e-50, 1, 1, 0, 1, 1], device=device)
    assert len(split.probs) == 4 and len(joined.probs) == 3


def test_codec_and_device_changes_do_not_reuse_old_geometry():
    method, ctx = SyncPoolMethod(), context()
    first = plan(method, ctx, [1]*6)
    replacement = context({0: b'x', 1: b'y', 2: b'z', 3: b'w', 4: b'v', 5: EOS_STEGA})
    changed = plan(method, replacement, [1]*6)
    assert len(first.probs) == 3 and len(changed.probs) == 6
    if torch.cuda.is_available():
        gpu = plan(method, replacement, [1]*6, device='cuda')
        cpu = plan(method, replacement, [1]*6)
        assert gpu.probs.device.type == 'cuda' and cpu.probs.device.type == 'cpu'
        torch.testing.assert_close(gpu.probs.cpu(), cpu.probs)
