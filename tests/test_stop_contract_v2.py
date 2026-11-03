import json
import subprocess
import sys

import pytest

from stegatext_runtime.experiments.config import ROOT, resolve
from stegatext_runtime.experiments.engine import planner, receive, send
from stegatext_runtime.experiments.engine_v2 import public_termination
from stegatext_runtime.experiments.stopping import normalize
from tests.finite_source import FiniteSource


def packet(method="las", **stopping):
    return dict(prompt="", language="en", message_delivery="prefix_stream",
                variant=dict(id=method, kind="planner", method=method, coder="discop_copy"),
                stopping=normalize(dict(mode="natural_eos_only", watchdog_protocol_decisions=16, **stopping)))


def roundtrip(source, p, bits="01"*2048):
    s = planner(source, p, key=b"finite-key", stream=bits)
    public = dict(p, text=s["text"], natural_eos=s["natural_eos"], termination=public_termination(s))
    if s["text"] is None:
        public["visible_bytes_hex"] = s["visible_bytes_hex"]
    r = planner(source, public, key=b"finite-key")
    assert s["bits"] == r["bits"]
    assert s["trace"] == r["trace"]
    assert s["state_sha256"] == r["state_sha256"]
    assert r["receiver_consumed"]
    return s, public


@pytest.mark.parametrize("name", ["syncpool", "las", "adaptive"])
def test_natural_eos_roundtrip(name):
    p = packet(name)
    if name == "adaptive":
        p["variant"]["tau"] = .5
    s, _ = roundtrip(FiniteSource(multiple_eos=True), p)
    assert s["terminal_reason"] == "natural_eos"
    assert s["natural_eos"] and not s["censored"]


def test_ordinary_and_planner_watchdogs_are_censored():
    stop = normalize(dict(mode="natural_eos_only", watchdog_generated_tokens=2, watchdog_protocol_decisions=3))
    ordinary = FiniteSource(endless=True).generate("", seed=11, stopping=stop)
    assert ordinary["censored"] and ordinary["terminal_reason"] == "resource_watchdog"
    p = packet()
    p["stopping"] = stop
    s, _ = roundtrip(FiniteSource(endless=True), p)
    assert s["censored"] and s["steps"] == 3 and not s["natural_eos"]
    assert ordinary["watchdog"]["unit"] != s["watchdog"]["unit"]


def test_utf8_boundary_is_preserved_without_repair():
    p = packet()
    p["stopping"]["watchdog_protocol_decisions"] = 1
    s, public = roundtrip(FiniteSource(chinese=True), p)
    assert s["text"] is None and public["visible_bytes_hex"] == "e4"
    assert s["censored"] and s["framing_status"] == "invalid_utf8"
    s, _ = roundtrip(FiniteSource(chinese=True), packet())
    assert s["text"] == "中" and s["terminal_reason"] == "natural_eos"


def test_bounded_utf8_extension_and_budget():
    p = packet()
    p["stopping"] = normalize(dict(mode="bounded_operational", max_steps=1, utf8_extension=4))
    s, _ = roundtrip(FiniteSource(chinese=True), p)
    assert s["text"] == "中" and s["steps"] == 2
    assert s["terminal_reason"] == "bounded_limit" and not s["censored"]


def test_goals_only_in_bounded_and_natural_fallback_does_not_truncate():
    with pytest.raises(ValueError, match="only allowed"):
        normalize(dict(mode="natural_eos_only", sentences=1))
    p = packet(max_steps=1, utf8_extension=0)
    s, _ = roundtrip(FiniteSource(), p)
    assert s["steps"] > 1 and s["natural_eos"]


def test_bounded_sentence_goal_is_a_distinct_normal_stop():
    source = FiniteSource(endless=True)
    source.backend.codec._bytes[0] = b"An example. "
    p = packet("syncpool")
    p["stopping"] = normalize(dict(mode="bounded_operational", max_steps=8, sentences=2))
    s, _ = roundtrip(source, p)
    assert s["steps"] == 2 and s["terminal_reason"] == "text_goal"
    assert not s["natural_eos"] and not s["censored"]


def test_ordinary_bounded_and_eos_on_watchdog_boundary():
    stop = normalize(dict(mode="bounded_operational", max_steps=2))
    s = FiniteSource(endless=True).generate("", seed=11, stopping=stop)
    assert s["terminal_reason"] == "bounded_limit" and not s["censored"]
    source = FiniteSource()
    source.transitions[()] = [(3, 1)]
    s = source.generate("", seed=11, stopping=normalize(dict(mode="natural_eos_only", watchdog_generated_tokens=1)))
    assert s["terminal_reason"] == "natural_eos" and not s["censored"]


def test_receiver_rejects_unconsumed_text_and_false_termination():
    source = FiniteSource()
    _, public = roundtrip(source, packet())
    public["text"] += "x"
    with pytest.raises((ValueError, RuntimeError)):
        planner(source, public, key=b"finite-key")
    _, public = roundtrip(source, packet())
    public["termination"]["terminal_reason"] = "resource_watchdog"
    with pytest.raises(ValueError, match="termination"):
        planner(source, public, key=b"finite-key")


def test_v1_packet_decode_compatibility():
    cfg = dict(schema_version=1, message=dict(mode="fixture", bits=4096, pattern="random"), randomness=dict(mode="fixture"),
               stopping=dict(max_steps=2, utf8_extension=32))
    case = dict(prompt=dict(text="", language="en"), seed=11,
                variant=dict(id="las", kind="planner", method="las", coder="discop_copy"))
    public, s = send(FiniteSource(), case, cfg)
    assert public["version"] == "unified-experiment-v1"
    r = receive(FiniteSource(), public, cfg)
    assert s["bits"] == r["bits"] and s["trace"] == r["trace"]


def test_v2_receiver_in_separate_process_without_sender_trace():
    cfg = dict(schema_version=2, message=dict(mode="fixture", bits=4096, pattern="random", delivery="prefix_stream"),
               randomness=dict(mode="fixture"), stopping=normalize(dict(mode="natural_eos_only")))
    case = dict(prompt=dict(text="", language="en"), seed=11,
                variant=dict(id="las", kind="planner", method="las", coder="discop_copy"))
    public, s = send(FiniteSource(), case, cfg)
    assert not ({"trace", "bits", "token_ids", "state_sha256"} & public.keys())
    script = "import json,sys; from tests.finite_source import FiniteSource; from stegatext_runtime.experiments.engine import receive; p,c=json.load(sys.stdin); print(json.dumps(receive(FiniteSource(),p,c)))"
    completed = subprocess.run([sys.executable, "-B", "-c", script], input=json.dumps([public, cfg]), text=True, capture_output=True, cwd=ROOT, check=True)
    r = json.loads(completed.stdout)
    assert r["receiver_consumed"] and r["bits"] == s["bits"] and r["state_sha256"] == s["state_sha256"]
    public["version"] = "unified-experiment-v1"
    with pytest.raises(ValueError, match="v2 packet"):
        receive(FiniteSource(), public, cfg)


def test_instrumentation_does_not_change_actions_randomness_or_model_queries(monkeypatch):
    from stegatext_runtime.experiments import engine_v2
    p = packet()
    source = FiniteSource()
    first = planner(source, p, key=b"finite-key", stream="01"*2048)
    queries = source.reference.forwards
    monkeypatch.setattr(engine_v2, "instrumentation", lambda *args: {})
    source = FiniteSource()
    second = planner(source, p, key=b"finite-key", stream="01"*2048)
    assert source.reference.forwards == queries
    assert all(first[k] == second[k] for k in ("text", "bits", "state_sha256", "steps"))


def test_nonce_and_rng_domains_are_separate():
    from stegatext_runtime.experiments.engine_v2 import nonce, shared_key
    from round4_2_common import CounterRNG
    cfg = dict(randomness=dict(mode="fixture"))
    n = nonce(FiniteSource(), "one", 11)
    key = shared_key(cfg, 11, n)
    assert key == shared_key(cfg, 11, n)
    assert key != shared_key(cfg, 11, nonce(FiniteSource(), "two", 11))
    assert CounterRNG(key, "coder", 1).random() != CounterRNG(key, "advance", 1).random()


def test_v2_config_and_invalid_native_protocol_rejected(tmp_path):
    text = (ROOT/"configs/experiments/sprint_smoke_v2.toml").read_text()
    path = tmp_path/"v2.toml"
    path.write_text(text)
    cfg = resolve(path)
    assert cfg["schema_version"] == 2 and "max_steps" not in cfg["stopping"]
    path.write_text(text.replace('kind = "planner"', 'kind = "upstream"'))
    with pytest.raises(ValueError, match="ordinary/planner"):
        resolve(path)
