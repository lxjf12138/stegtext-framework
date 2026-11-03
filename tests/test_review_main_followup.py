"""Prevent premature GPU work, seed reuse, and result-dependent replication."""
from copy import deepcopy
import math

from stegatext_runtime.experiments import runner
from tools.review_main_followup_20261025 import parent_finished, select_targets, finish_report


def cell(variant, *, k=256, ppl=5., bpb=.4, entropy=100.):
    return dict(model="llama_instruct", top_k=k, variant=variant, planned=32,
                ppl=ppl, byte_delivery_bpb=bpb, group_entropy_bits=entropy)


def test_waits_for_entire_main_and_guard_not_a_completed_batch(tmp_path):
    runner.write(tmp_path / "batches/batch_01/COMPLETE.json", {})
    runner.write(tmp_path / "RESOURCES.json", dict(status="running"))
    assert not parent_finished(tmp_path)
    runner.write(tmp_path / "COMPLETE.json", {})
    assert not parent_finished(tmp_path)
    runner.write(tmp_path / "RESOURCES.json", dict(status="failed"))
    assert not parent_finished(tmp_path)
    runner.write(tmp_path / "RESOURCES.json", dict(status="finished"))
    assert parent_finished(tmp_path)


def test_ppl_screen_is_two_sided_and_does_not_force_filtered_baselines_to_match():
    cells = [cell("ordinary"), cell("las_imec"), cell("syncpool_imec"),
             cell("adaptive_0500_imec", ppl=4.), cell("adaptive_1000_imec", ppl=6.),
             cell("basic_imec", ppl=100.), cell("mwis_imec", ppl=2.)]
    before = deepcopy(cells)
    targets = select_targets(cells)
    assert cells == before
    assert len(targets) == 1
    assert set(targets[0]["variants"]) == {"ordinary", "las_imec", "syncpool_imec", "adaptive_0500_imec", "adaptive_1000_imec"}
    reasons = [r for r in targets[0]["reasons"] if r["kind"] == "ppl_distribution_diagnostic"]
    assert {round(r["ratio"], 2) for r in reasons} == {.8, 1.2}


def test_capacity_and_reversals_are_labelled_empirical_and_pair_both_sides():
    cells = [cell("ordinary"), cell("las_imec"), cell("syncpool_imec", bpb=.3),
             cell("adaptive_0250_imec", bpb=.30), cell("adaptive_0500_imec", bpb=.34),
             cell("adaptive_4000_imec", bpb=.10)]
    target = select_targets(cells)[0]
    assert "adaptive_4000_imec" not in target["variants"]
    assert {"adaptive_0250_imec", "adaptive_0500_imec"} <= set(target["variants"])
    assert {r["kind"] for r in target["reasons"]} == {"empirical_capacity_retention", "empirical_threshold_reversal"}


def test_no_targets_for_small_differences_or_expected_high_tau_capacity_loss():
    cells = [cell("ordinary"), cell("las_imec", ppl=5.1), cell("syncpool_imec", ppl=4.9),
             cell("adaptive_0500_imec", bpb=.38), cell("adaptive_4000_imec", bpb=.1)]
    assert select_targets(cells) == []


def test_k_reversal_repeats_original_neighbor_conditions():
    cells = [cell(v, k=k, bpb=b) for k, b in [(128, .4), (256, .3)]
             for v in ["ordinary", "las_imec", "syncpool_imec"]]
    targets = select_targets(cells)
    assert {t["top_k"] for t in targets} == {128, 256}
    assert all(any(r["kind"] == "empirical_k_reversal" for r in t["reasons"]) for t in targets)


def test_confirmation_report_keeps_extreme_unrenderable_and_failed_scores(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.review_main_followup_20261025.analyze", lambda _: None)
    rounds = [tmp_path / "repeat_01", tmp_path / "repeat_02"]
    for index, out in enumerate(rounds):
        rows = []
        for prompt in ["a", "b"]:
            for variant in ["ordinary", "las_imec", "syncpool_imec"]:
                extreme = index == 1 and prompt == "a" and variant == "las_imec"
                failed_score = index == 1 and prompt == "b" and variant == "syncpool_imec"
                ordinary = variant == "ordinary"
                rows.append(dict(model="llama_instruct", top_k=256, variant=variant,
                    prompt_id=prompt, seed=470101 + index, method="ordinary" if ordinary else "las",
                    sender_status="ok", receiver_status=None if ordinary else "ok", exact=not ordinary,
                    nll_total=None if failed_score else 10 * math.log(1000 if extreme else 5),
                    evaluator_tokens=0 if failed_score else 10, bytes=20, body_tokens=10,
                    strict_verified_bits=None if ordinary else 8,
                    text_renderable=not extreme, strict_text_delivery_bits=0 if extreme or ordinary else 8))
        runner.write(out / "analysis/progress_audit/CASES.json", dict(rows=rows))
    finish_report(tmp_path, rounds)
    report = runner.read(tmp_path / "analysis/SUMMARY.json")
    saved = runner.read(tmp_path / "analysis/CASES.json")["rows"]
    assert len(saved) == report["generated"] == 12
    assert report["main_data_included"] is False
    las = next(c for c in report["cells"] if c["variant"] == "las_imec")
    assert las["unrenderable_outputs"] == 1 and las["bytes"] == 80
    assert las["byte_delivery_bpb"] == .4 and las["valid_text_delivery_bpb"] == .3
    sp = next(c for c in report["intervals"] if c["variant"] == "syncpool_imec")
    assert sp["status"] == "unavailable_retained_failures"
