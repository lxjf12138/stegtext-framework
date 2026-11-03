"""Regression tests for bounded per-generation diagnostics.

The assertions intentionally inspect one generated carrier and one ordinary
sample.  They verify that the new fields are present and reconstructible while
the action stream and independent receiver contract remain unchanged.
"""
import json

from stegatext_runtime.experiments.engine_v2 import planner, public_termination
from stegatext_runtime.experiments.stopping import normalize
from tests.finite_source import FiniteSource


def packet(method="las", coder="discop_copy", limit=4):
    variant = dict(id=method, kind="planner", method=method, coder=coder)
    if method == "adaptive":
        variant["tau"] = 0.5
    if coder == "imec_prefix_exact":
        variant["coder_parameters"] = dict(block_size=6, window_size=3)
    return dict(prompt="", language="en", message_delivery="prefix_stream", variant=variant,
                stopping=normalize(dict(mode="token_budget", max_new_tokens=limit,
                                        watchdog_protocol_decisions=64)))


def test_planner_records_single_generation_source_posterior_and_payload_metrics():
    result = planner(FiniteSource(), packet(), key=b"recording", stream="01" * 2048)
    assert result["recording_contract_version"] == "single-generation-recording-v2"
    assert result["trace_sha256"]
    assert result["source_entropy_total_bits"] >= 0
    assert result["group_entropy_total_bits"] >= 0
    assert result["total_protocol_decisions"] == result["steps"]
    assert result["native_output_token_count"] >= 1
    row = result["trace"][0]
    for key in ("source_distribution_summary", "group_entropy_bits", "selected_group_probability",
                "posterior_entropy_bits", "candidate_identity_entropy_bits",
                "active_mass_before", "active_mass_after", "retained_frontier_count_after",
                "source_probability_mass_retained", "kl_method_to_source_bits", "tv_distance",
                "consumed_bits", "visible_bytes_added", "coder_diagnostics"):
        assert key in row
    assert row["coder_diagnostics"]["consumed_bits_this_step"] == row["consumed_bits"]
    # The diagnostics are bounded summaries; no full frontier/probability dump
    # is allowed to reappear in a serialized single-case record.
    assert len(json.dumps(result, ensure_ascii=False)) < 100_000


def test_ordinary_generation_records_native_path_and_each_sampler_step():
    source = FiniteSource()
    stop = normalize(dict(mode="token_budget", max_new_tokens=3,
                          watchdog_generated_tokens=16, watchdog_protocol_decisions=16))
    result = source.generate("", seed=17, stopping=stop)
    rows = result["ordinary_sampling"]
    assert result["recording_contract_version"] == "single-generation-recording-v2"
    assert len(rows) == result["generated_token_count"]
    assert result["native_token_sequence"]["output_token_ids"] == result["token_ids"]
    assert result["native_token_sequence"]["wire_reconstruction_verified"]
    assert all(row["source_support_count"] > 0 for row in rows)
    assert all("selected_token_probability" in row and "source_entropy_bits" in row for row in rows)
    assert result["source_entropy_total_bits"] == sum(row["source_entropy_bits"] for row in rows)


def test_imec_step_records_coupling_contract_without_full_matrix():
    result = planner(FiniteSource(), packet(coder="imec_prefix_exact", limit=1),
                     key=b"imec-recording", stream="01" * 2048)
    coder = result["trace"][0]["coder"]
    assert coder["exact_marginals"]
    assert coder["row_normalization_error"] == 0.0
    assert coder["column_normalization_error"] == 0.0
    assert coder["coupling_mass_error"] == 0.0
    assert "coupling_edges" not in coder or len(coder["coupling_edges"]) <= 4096
    assert result["bits_count"] == result["online_committed_bits"] + result["terminal_extra_prefix_bits"]


def test_sender_and_receiver_have_identical_reconstructible_trace():
    sender = planner(FiniteSource(), packet("adaptive"), key=b"pair", stream="01" * 2048)
    public = dict(packet("adaptive"), text=sender["text"], natural_eos=sender["natural_eos"],
                  termination=public_termination(sender))
    receiver = planner(FiniteSource(), public, key=b"pair")
    assert sender["trace"] == receiver["trace"]
    assert sender["trace_sha256"] == receiver["trace_sha256"]

