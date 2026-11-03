"""Offline receiver-vs-request metrics, separate from receiver authentication."""
from . import framing


def common_prefix(*values):
    if not values:
        return 0
    return next((i for i, bits in enumerate(zip(*values)) if len(set(bits)) != 1), min(map(len, values)))


def measure(requested, sent, received, *, delivery="prefix_stream", replay_exact=False, auxiliary_bits=0):
    wire = framing.encode(requested) if delivery == "finite_frame" else requested
    if delivery == "finite_frame":
        wire += "0"*max(0, max(len(sent), len(received or ""))-len(wire))
    confirmed_wire = common_prefix(wire, sent, received or "")
    consumed_wire = len(sent)
    header = framing.HEADER_BITS if delivery == "finite_frame" else 0
    consumed = max(0, min(len(requested), consumed_wire-header))
    confirmed = max(0, min(len(requested), confirmed_wire-header))
    decoded = framing.inspect(received) if delivery == "finite_frame" and received is not None else None
    strict = bool(replay_exact and received is not None and received == sent and sent == wire[:len(sent)] and len(sent) <= len(wire))
    return dict(prefix_exact=strict, nonempty_payload=bool(strict and consumed),
                requested_message_bits=len(requested), consumed_prefix_bits=consumed,
                confirmed_prefix_bits=confirmed, consumed_wire_bits=consumed_wire,
                confirmed_wire_prefix_bits=confirmed_wire,
                full_message_delivered=bool(delivery == "finite_frame" and decoded is not None and decoded == requested and replay_exact),
                frame_verified=(decoded is not None) if delivery == "finite_frame" else None,
                frame_overhead_bits=framing.OVERHEAD_BITS if delivery == "finite_frame" else 0,
                padding_bits_consumed=max(0, consumed_wire-len(requested)-framing.OVERHEAD_BITS) if delivery == "finite_frame" else 0,
                auxiliary_bits=auxiliary_bits, message_delivery=delivery,
                confirmation_scope="Offline exact common-prefix check, not receiver authentication",
                sender_prefix_valid=sent == wire[:len(sent)] and len(sent) <= len(wire),
                strict_prefix_verified=strict, strict_verified_payload_bits=consumed if strict else 0)


def aggregate(rows):
    groups = {}
    for row in rows:
        if row["variant"]["kind"] == "ordinary":
            continue
        groups.setdefault((row["model"], row["variant"]["id"]), []).append(row)
    result = []
    for (model, variant), group in sorted(groups.items()):
        lengths = [(r.get("output") or {}).get("text_utf8_bytes") for r in group]
        denominator = sum(lengths) if all(x is not None for x in lengths) else None
        bits = sum(r["confirmed_prefix_bits"] for r in group)
        strict_bits = sum(r.get('strict_verified_payload_bits',r['confirmed_prefix_bits'] if r['exact'] is True else 0) for r in group)
        tokens = [(r.get('output') or {}).get('evaluation_tokens') for r in group]
        token_denominator = sum(tokens) if all(t is not None for t in tokens) else None
        auxiliary = sum(r["auxiliary_bits"] for r in group)
        seconds = sum((r.get("performance") or {}).get("wall_seconds", 0) for r in group)
        receiver_seconds = sum((r.get("receiver_performance") or {}).get("wall_seconds", 0) for r in group)
        result.append(dict(model=model, variant=variant, cases=len(group),
                           confirmed_prefix_bits=bits, output_utf8_bytes=denominator,
                           reliable_bpb=strict_bits/denominator if denominator else None,
                           strict_verified_payload_bits=strict_bits,
                           reliable_bpt=strict_bits/token_denominator if token_denominator else None,
                           output_evaluation_tokens=token_denominator,
                           zero_payload_fraction=sum(r['confirmed_prefix_bits'] == 0 for r in group)/len(group),
                           common_prefix_bpb_legacy=bits/denominator if denominator else None,
                           auxiliary_bits=auxiliary,
                           sender_wall_seconds=seconds, receiver_wall_seconds=receiver_seconds,
                           confirmed_bits_per_second=strict_bits/seconds if seconds else None,
                           end_to_end_confirmed_bits_per_second=strict_bits/(seconds+receiver_seconds) if seconds+receiver_seconds else None,
                           sender_model_forwards=sum((r.get("performance") or {}).get("model_forwards", 0) for r in group),
                           auxiliary_adjusted_bpb=max(0, strict_bits-auxiliary)/denominator if denominator else None,
                           full_messages_delivered=sum(r["full_message_delivered"] for r in group),
                           zero_payload_cases=sum(r["confirmed_prefix_bits"] == 0 for r in group),
                           censored_cases=sum(r["censored"] is True for r in group),
                           failed_cases=sum(r["exact"] is False for r in group),
                           invalid_or_missing_text_cases=sum(x is None for x in lengths)))
    return result
