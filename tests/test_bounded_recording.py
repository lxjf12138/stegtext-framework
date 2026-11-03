"""Independent legacy-format, reconstruction and memory regressions."""
from fractions import Fraction
import hashlib
import json
import tracemalloc

import pytest

from stegatext_runtime.experiments.config import canonical, fingerprint
from stegatext_runtime.experiments.recording import canonical_digest, coder_trace
from stegatext_runtime.experiments.runner import read, write


def test_streamed_digest_and_file_bytes_match_legacy_format(tmp_path):
    record = dict(unicode="中文🙂", nested=[dict(z=-0., a=1e-200), None, True, (1, 2)],
                  p=[.3, 1/3, 1e30])
    assert canonical_digest(record) == fingerprint(record)
    expected = json.dumps(dict(record, record_sha256=fingerprint(record)),
                          ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = tmp_path / "record.json"
    write(path, record)
    assert path.read_bytes() == expected.encode()
    path.write_text(expected + " "*20000)
    assert read(path) == json.loads(json.dumps(record))


def test_hash_and_write_do_not_allocate_a_complete_json_string(tmp_path):
    value = dict(large=["x"*128]*100000)
    tracemalloc.start()
    digest = canonical_digest(value)
    _, peak_hash = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert digest == hashlib.sha256(canonical(value).encode()).hexdigest()
    tracemalloc.start()
    write(tmp_path / "large.json", value)
    _, peak_write = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak_hash < 1_000_000
    assert peak_write < 1_000_000


def test_partial_write_preserves_previous_record_and_removes_temporary(tmp_path, monkeypatch):
    path = tmp_path / "record.json"
    write(path, dict(old=True))
    before = path.read_bytes()
    def failed_dump(value, handle, **kwargs):
        handle.write('{"partial":')
        raise OSError("injected write interruption")
    monkeypatch.setattr(json, "dump", failed_dump)
    with pytest.raises(OSError, match="interruption"):
        write(path, dict(new=True))
    assert path.read_bytes() == before
    assert not path.with_suffix(".json.tmp").exists()


def test_truncated_gigabyte_record_is_rejected_before_parsing(tmp_path, monkeypatch):
    path = tmp_path / "truncated.json"
    with path.open("wb") as handle:
        handle.seek(4*1024**3)
        handle.write(b'"partial Fraction 123/9')
    def forbidden(*args, **kwargs):
        raise AssertionError("Must not allocate or parse the corrupt giant record")
    monkeypatch.setattr(json, "load", forbidden)
    with pytest.raises(ValueError, match="Incomplete record"):
        read(path)


def test_small_and_non_imec_traces_are_unchanged():
    small = dict(group_probabilities=["1/2", "1/2"], coupling_edges=[[0, 0, "1/2"]])
    assert coder_trace(small) is small
    other = dict(group_probabilities=[1.]*5000)
    assert coder_trace(other) is other


def test_large_imec_summary_matches_exact_marginals_and_independent_payload(tmp_path):
    from stegatext_runtime.imec import coupling
    from stegatext_runtime.experiments.engine import send, receive
    from tests.test_all_method_randomness import fixture
    from tests.test_discop_message_demand import UniformWords
    config, case = fixture(tmp_path)
    case["variant"].update(method="basic_bytes_v1", coder="imec_prefix_exact",
                           coder_parameters=dict(block_size=6, window_size=3))
    packet, sent = send(UniformWords(8192), case, config)
    received = receive(UniformWords(8192), packet, config)
    assert sent["bits_count"] == 6
    assert received["bits"] == sent["bits"]
    assert received["trace"] == sent["trace"]
    first = sent["trace"][0]
    trace = first["coder"]
    assert "coupling_edges" not in trace and "group_probabilities" not in trace
    prior = [Fraction(0)]*64
    for i, value in trace["posterior_before"]:
        prior[i] = Fraction(value)
    assert "group_probabilities" not in first
    p, q, rows, columns = coupling(prior, [1/8192]*8192)
    assert all(sum(row.values()) == mass for row, mass in zip(rows, p))
    assert all(sum(col.values()) == mass for col, mass in zip(columns, q))
    storage = trace["storage"]
    assert storage["version"] == 'summary-imec-trace-v2'
    assert storage["group_count"] == storage["coupling_edge_count"] == 8192
    assert storage["probability_head"] == [[j, str(q[j])] for j in range(8)]
    assert storage["selected_group_probability"] == str(q[first["action"]])
    assert first['group_distribution']['entropy_bits_normalized'] == 13.
    assert len(json.dumps(sent)) < 25000
    write(tmp_path / "sender.json", sent)
    assert read(tmp_path / "sender.json")["trace"] == sent["trace"]
    # Distribution summaries remain auditable after delivery compression too.
    from tools.controlled_comparison.replay import compact_record
    write(tmp_path / "envelope.json", dict(result=sent))
    envelope = json.loads((tmp_path / "envelope.json").read_text())
    compact = compact_record(envelope)
    assert compact == envelope
