import pytest

from tools.tradeoff.analyze import comparison, cost, summary, lower_envelope


def row(bits, nbytes, seconds, exact=True):
    return dict(verified_bits=bits if exact else 0, bytes=nbytes,
                sender_seconds=seconds, receiver_seconds=seconds, sender_forwards=10,
                sender_queries=5, exact=exact, error=None if exact else 'failure',
                censored=False, terminal_reason='natural_eos', las_kernel_steps=0,
                fullsync_kernel_steps=0, positive_u_steps=0, decisions=5,
                strict_events=0, group_entropy_sum=0)


def test_ratio_of_sums_includes_failed_and_zero_bit_carriers():
    s = summary([row(10, 100, 2), row(20, 900, 3), row(999, 100, 5, exact=False)])
    assert s['bits'] == 30
    assert s['bytes'] == 1100
    assert s['sender_bps'] == 3
    assert s['bpb'] == 30/1100
    assert s['errors'] == s['zero_payload'] == 1


def test_break_even_solves_actual_cost_and_has_seconds_per_byte_units():
    fast = [row(10, 100, 1)]
    compact = [row(20, 100, 4)]
    point = comparison(compact, fast)
    threshold = point['sender_break_even_seconds_per_byte']
    assert threshold == pytest.approx(.02)
    assert cost(summary(compact), threshold, 'sender') == pytest.approx(cost(summary(fast), threshold, 'sender'))
    assert cost(summary(compact), .01, 'sender') > cost(summary(fast), .01, 'sender')
    assert cost(summary(compact), .03, 'sender') < cost(summary(fast), .03, 'sender')


def test_unknown_text_length_does_not_silently_drop_failed_case():
    assert summary([row(10, 100, 1), row(0, None, 5, exact=False)])['bpb'] is None


def test_affine_envelope_finds_intermediate_method_and_merges_irrelevant_cuts():
    candidates={m:dict(sender_seconds_per_bit=a,bytes_per_bit=b) for m,a,b in
                [('fast',1.,10.),('adaptive',2.,5.),('compact',4.,1.)]}
    result=lower_envelope(candidates,'sender')
    assert [r['methods'] for r in result]==[['fast'],['adaptive'],['compact']]
    assert result[0]['lambda_to']==pytest.approx(.2)
    assert result[1]['lambda_from']==pytest.approx(.2)
    assert result[1]['lambda_to']==pytest.approx(.5)
    assert result[2]['lambda_to'] is None
