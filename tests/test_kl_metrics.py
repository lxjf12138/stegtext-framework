import math

import pytest

from stegatext_runtime.experiments.kl import discrete_metrics, restricted_distribution_metrics


def test_restricted_filter_closed_form_matches_definition():
    value = restricted_distribution_metrics(0.75)
    assert value["source_probability_mass_retained"] == pytest.approx(0.75)
    assert value["kl_method_to_source_bits"] == pytest.approx(-math.log2(0.75))
    assert value["kl_source_to_method_bits"] is None
    assert value["tv_distance"] == pytest.approx(0.25)
    assert value["support_mismatch"] is True


def test_no_filter_has_zero_both_kl_orientations():
    value = restricted_distribution_metrics(1.0)
    assert value["kl_method_to_source_bits"] == pytest.approx(0.0)
    assert value["kl_source_to_method_bits"] == pytest.approx(0.0)
    assert value["tv_distance"] == pytest.approx(0.0)
    assert value["support_mismatch"] is False


def test_common_space_metrics_detect_support_mismatch():
    value = discrete_metrics([0.25, 0.75], [1.0, 0.0])
    assert value["kl_method_to_source_bits"] == pytest.approx(2.0)
    assert value["kl_source_to_method_bits"] is None
    assert value["tv_distance"] == pytest.approx(0.75)


def test_common_space_metrics_are_zero_for_identical_law():
    value = discrete_metrics([0.2, 0.3, 0.5], [0.2, 0.3, 0.5])
    assert value["kl_method_to_source_bits"] == pytest.approx(0.0, abs=1e-14)
    assert value["kl_source_to_method_bits"] == pytest.approx(0.0, abs=1e-14)
    assert value["tv_distance"] == pytest.approx(0.0, abs=1e-14)

