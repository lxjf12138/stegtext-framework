"""Scientific controls and equivalence checks for the experiment framework."""
import copy
import functools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from stegatext_runtime.experiments.config import ROOT, cases, resolve
from stegatext_runtime.experiments.engine import clone, key_for, message_for, planner
from stegatext_runtime.experiments.runner import cached, differences, read, write
from stegatext_runtime.protocol import encode as active_encode
from stegtext.v2.core.types import TokenDistribution
from stegtext.v2.model.toy_backend import ToyBackend, ToyBackendConfig


class ToySource:
    def __init__(self):
        self.backend = ToyBackend(ToyBackendConfig({0: b"a", 1: b"ab", 2: b"b"}, {}))
        self.backend.codec_index = self.backend.index
        self.backend.codec.text_to_bytes = lambda text: text.encode()
        self.config = SimpleNamespace(do_sample=True)
    def bind(self, prompt): return self.backend.render_prompt(prompt)
    def distribution(self, path):
        # Different latent histories induce different predictions.
        p = [.4, .2, .4] if len(path.token_ids()) % 2 == 0 else [.2, .5, .3]
        return TokenDistribution(torch.tensor(p, dtype=torch.float64))
    def manifest(self): return {"scope": "toy-exact"}


@pytest.mark.parametrize("policy,tau", [("las", 0.), ("fullsync", float("inf")), ("adaptive", .5)])
@pytest.mark.parametrize("bits", ["0"*4096, "1"*4096, "00001101011001"*300])
def test_shared_planner_matches_active_protocol_and_receiver(policy, tau, bits):
    key = b"toy-shared-key"*2
    old_packet, expected = active_encode(ToySource(), "", bits, key=key, policy=policy, tau=tau, max_decisions=12)
    variant = dict(id=policy, kind="planner", method=policy, coder="discop_copy")
    if policy == "adaptive": variant["tau"] = tau
    packet = dict(prompt="", variant=variant, stopping=dict(max_steps=12, utf8_extension=32), language="en")
    sent = planner(ToySource(), packet, key=key, stream=bits)
    packet.update(text=sent["text"], natural_eos=sent["natural_eos"])
    received = planner(ToySource(), packet, key=key)
    for field in ("bits", "text", "state_sha256"):
        assert sent[field] == expected[field] == received[field]
    assert sent["trace"] == received["trace"]
    assert [r["action"] for r in sent["trace"]] == [r["action"] for r in expected["trace"]]


def test_configuration_inheritance_keeps_controls_and_pairing():
    base = resolve(ROOT/"configs/experiments/smoke.toml")
    changed = resolve(ROOT/"configs/experiments/temperature_ablation.toml")
    assert differences(base, changed) == [
        dict(field="name", before="shared_smoke", after="temperature_ablation"),
        dict(field="source.sampling.temperature", before=None, after=.9),
    ]
    group = list(cases(base))
    assert len(group) == 16
    assert all(key_for(base, c["seed"]) == key_for(changed, c["seed"]) for c in group)
    assert all(message_for(base, c["seed"]) == message_for(changed, c["seed"]) for c in group)
    assert key_for(base, 11) != key_for(base, 29)
    assert message_for(base, 11) != message_for(base, 29)


def test_corrupt_record_and_wrong_resume_contract_rejected(tmp_path):
    path = tmp_path/"record.json"
    write(path, dict(status="ok", case_id="a", execution_sha256="one"))
    assert cached(path, dict(case_id="a", execution_sha256="one"))
    with pytest.raises(ValueError, match="Resume contract"):
        cached(path, dict(case_id="a", execution_sha256="two"))
    raw = json.loads(path.read_text()); raw["status"] = "error"; path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="Corrupted"):
        read(path)


def test_config_typo_and_inheritance_cycle_rejected(tmp_path):
    path = tmp_path/"bad.toml"
    path.write_text('extends = "'+str(ROOT/"configs/experiments/smoke.toml")+'"\n[source.sampling]\ntop_pp=0.9\n')
    with pytest.raises(ValueError, match="Unknown sampling"):
        resolve(path)
    path.write_text('extends = "bad.toml"\n')
    with pytest.raises(ValueError, match="inheritance cycle"):
        resolve(path)


def test_decorated_kernel_binding_isolated_without_monkeypatch():
    # Torch decorators wrap closures. Cloning the wrapper would silently keep
    # the old global binding, which would violate shared-message controls.
    namespace = {"payload": lambda: "original", "torch": torch}
    exec('@torch.inference_mode()\ndef wrapped():\n return payload()\n', namespace)
    changed = clone(namespace["wrapped"], payload=lambda: "configured")
    assert changed() == "configured"
    assert namespace["wrapped"]() == "original"
