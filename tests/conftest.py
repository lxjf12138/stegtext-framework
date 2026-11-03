"""Skip checks that read experiment records not distributed with this repository.

These tests replay the prepared main-experiment configurations or recorded
runs. They run in a workspace that contains those records and are skipped,
with the missing path as the reason, otherwise.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PREPARATION = "research_notes/baseline_experiment_design_20261022/main_table_uniform_preparation_20261023/configs"
RECORDS = {
    "tests/test_imec_terminal_prefix.py::test_historical_64_case_terminal_prefix_vector":
        "reports/sentence_imec_20261018/runs.json",
    "tests/test_review_main_runner.py::test_ten_document_batches_cover_all_conditions_without_crossing_boundaries":
        PREPARATION,
    "tests/test_review_main_runner.py::test_all_k_use_identical_inputs_and_reject_a_smaller_full_subset":
        PREPARATION,
    "tests/test_review_main_runner.py::test_slicing_real_main_configuration_preserves_cases_and_random_streams":
        PREPARATION,
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        required = RECORDS.get(item.nodeid)
        if required and not (ROOT/required).exists():
            item.add_marker(pytest.mark.skip(reason=f"requires experiment records: {required}"))
