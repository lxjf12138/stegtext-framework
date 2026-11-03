import pytest

from stegatext_runtime.experiments import framing
from stegatext_runtime.experiments.delivery import aggregate, measure


def test_prefix_equality_does_not_mean_full_delivery():
    r = measure("01"*100, "0101", "0101", replay_exact=True)
    assert r["confirmed_prefix_bits"] == 4 and r["requested_message_bits"] == 200
    assert not r["full_message_delivered"] and r["frame_verified"] is None


def test_wrong_shared_prefix_is_checked_against_requested_message():
    r = measure("00110", "00000", "00000", replay_exact=True)
    assert r["confirmed_prefix_bits"] == 2 and not r["sender_prefix_valid"]
    assert measure("00110", "00110", "00011")["confirmed_prefix_bits"] == 2


@pytest.mark.parametrize("payload", ["", "0", "10101", "1"*64])
def test_frame_complete_early_eos_and_corruption(payload):
    bits = framing.encode(payload)
    r = measure(payload, bits+"000", bits+"000", delivery="finite_frame", replay_exact=True)
    assert r["full_message_delivered"] and r["frame_verified"]
    assert r["confirmed_prefix_bits"] == len(payload) and r["padding_bits_consumed"] == 3
    r = measure(payload, bits[:-1], bits[:-1], delivery="finite_frame", replay_exact=True)
    assert not r["full_message_delivered"] and not r["frame_verified"]
    wrong = bits[:-1]+str(1-int(bits[-1]))
    assert framing.inspect(wrong) is None


def test_no_receiver_means_zero_confirmation_and_side_information_counted():
    r = measure("0101", "0101", None, auxiliary_bits=33)
    assert r["confirmed_prefix_bits"] == 0 and r["auxiliary_bits"] == 33
    assert measure("", "", "")["confirmed_prefix_bits"] == 0


def test_bpb_is_ratio_of_sums_and_includes_failures_and_zero_bits():
    base = dict(model="toy", variant=dict(id="las", kind="planner"), auxiliary_bits=0,
                full_message_delivered=False, censored=False, exact=True)
    rows = [dict(base, confirmed_prefix_bits=10, output=dict(text_utf8_bytes=10)),
            dict(base, confirmed_prefix_bits=0, output=dict(text_utf8_bytes=90), exact=False)]
    r = aggregate(rows)[0]
    assert r["reliable_bpb"] == .1 and r["cases"] == 2 and r["failed_cases"] == 1


def test_finite_frame_actual_engine_continues_to_eos_and_excludes_padding():
    from tests.finite_source import BitCarrierSource
    from stegatext_runtime.experiments.engine import message_for, receive, send
    from stegatext_runtime.experiments.stopping import normalize
    cfg = dict(schema_version=2, message=dict(mode="fixture", bits=8, pattern="random", delivery="finite_frame"),
               randomness=dict(mode="fixture"), stopping=normalize(dict(mode="natural_eos_only")))
    case = dict(prompt=dict(text="", language="en"), seed=11,
                variant=dict(id="las", kind="planner", method="las", coder="discop_copy"))
    packet, s = send(BitCarrierSource(), case, cfg)
    r = receive(BitCarrierSource(), packet, cfg)
    assert s["natural_eos"] and r["receiver_consumed"] and s["trace"] == r["trace"]
    metrics = measure(message_for(cfg, 11), s["bits"], r["bits"], delivery="finite_frame", replay_exact=True)
    assert metrics["full_message_delivered"] and metrics["confirmed_prefix_bits"] == 8
    assert metrics["consumed_wire_bits"] == 128 and metrics["padding_bits_consumed"] == 40
