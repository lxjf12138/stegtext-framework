"""Versioned text receiver for native codecs, including chat and EOS identity.

Planner methods operate on terminal byte equivalence classes instead. Native
token codecs need the terminal token identity when several EOS IDs are valid;
that identity is explicit public side information, never a hidden token trace.
"""
import math

from research_notes.full_validation import systems


def attach(source, packet, result):
    if source.prompt_mode != "chat":
        return packet
    terminal = result["tokens"][-1] if packet["natural_eos"] else None
    eos = sorted(source.eos_ids)
    if terminal is not None and terminal not in eos:
        raise ValueError("Natural EOS packet has no terminal token")
    return dict(packet, text_transport={
        "version": "rendered-prefix-eos-v1",
        "eos_index": eos.index(terminal) if terminal is not None else None,
        "eos_index_bits": math.ceil(math.log2(len(eos))) if terminal is not None else 0,
    })


def text_tokens(source, packet):
    if "text_transport" not in packet:
        if source.prompt_mode == "chat":
            raise ValueError("Chat native packet is missing its text transport contract")
        return systems.text_tokens(source, packet)
    info = packet["text_transport"]
    if info["version"] != "rendered-prefix-eos-v1":
        raise ValueError("Unknown text transport version")
    tokens = source.retokenize_text(packet["prompt"], packet["text"])
    eos = sorted(source.eos_ids)
    index = info["eos_index"]
    expected_bits = math.ceil(math.log2(len(eos))) if packet["natural_eos"] else 0
    if info["eos_index_bits"] != expected_bits:
        raise ValueError("Incorrect terminal side-information accounting")
    if packet["natural_eos"]:
        if type(index) is not int or not 0 <= index < len(eos):
            raise ValueError("Invalid terminal EOS index")
        tokens.append(eos[index])
    elif index is not None:
        raise ValueError("EOS index on an unterminated packet")
    return tokens
