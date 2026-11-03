"""Bounded carriers around original Meteor and reviewed ANStega cores.

This is an explicitly new delivery protocol: fresh public prompt per carrier,
fixed token budget with UTF-8 completion, public ANS terminal state. It makes
no claim of being the same reference channel as one unbounded article.
"""
from __future__ import annotations

import argparse
import json
import os

import bitarray
import torch

from stegatext_runtime import Source
from research_notes.full_validation.suite import ROOT, PROMPT, ZH_PROMPT, bits_for, fixture_key, write
from research_notes.full_validation.systems import BoundSource, Medium, upstream, text_tokens
from research_notes.full_validation.transport import PAYLOAD, frame, unframe
from research_notes.problem_resolution.paper_adapters import anstega_paper_namespace


OUT = ROOT / "runs/problem_resolution_v1"
HERE = ROOT / "research_notes/problem_resolution"


class CarrierBoundary(Exception):
    pass


class BoundedSource(BoundSource):
    def __init__(self, source, prompt, token_budget=128):
        super().__init__(source, prompt, cap=token_budget+33)
        self.token_budget = token_budget

    def distribution(self, ids):
        generated = list(ids[len(self.prompt.token_ids):])
        if generated and generated[-1] in self.source.eos_ids:
            raise CarrierBoundary("natural_eos")
        if len(generated) >= self.token_budget:
            codec = self.source.backend.codec
            try:
                codec.bytes_to_text(codec.tokens_to_bytes(generated))
            except Exception:
                if len(generated) >= self.token_budget+32:
                    raise
            else:
                raise CarrierBoundary("token_budget_utf8_boundary")
        return super().distribution(ids)


@torch.inference_mode()
def send_carrier(source, system, prompt, stream, key, seq):
    bound = BoundedSource(source, prompt)
    seed = 101+seq
    stop = None
    aux = dict(token_budget=128, seed=seed, precision=32)
    if system.startswith("anstega_"):
        ns = anstega_paper_namespace(bound, mask_key=key if system == "anstega_keyed" else None)
        supplied = "1"+stream
        records = []
        original = ns["encode_step"]

        def step(*args):
            result = original(*args)
            records.append(result[:3])
            return result

        ns["encode_step"] = step
        try:
            ns["encode_anstega"](None, torch.tensor([bound.prompt.token_ids]), supplied, 161,
                                 device="cpu", precision=32, ans_t=16, seed=seed)
        except CarrierBoundary as exc:
            stop = str(exc)
        if not records:
            raise ValueError("No ANS token was produced")
        tokens = [r[0] for r in records]
        state = records[-1][1]
        raw_bits = supplied[:37]+"".join(r[2] for r in records)
        expected = raw_bits[1:]
        assert 2**36 <= state < 2**37
        aux.update(ans_t=16, final_state=state, final_state_bits=state.bit_length(), public_initial_prefix="1",
                   mask_rng="hmac-sha256-bitstream-v1" if system == "anstega_keyed" else "python-mt19937-fixture")
    else:
        ns = upstream("meteor_borrowed", bound)
        medium = Medium(bound)
        progress = []
        original = ns["num_same_from_beg"]

        def common(*args):
            n = original(*args)
            progress.append(n)
            return n

        ns["num_same_from_beg"] = common
        cfg = dict(finish_sent=False, precision=32, is_sort=False, seed=seed, input_key=key,
                   sample_seed_prefix=b"bounded-meteor-v1", input_nonce=seq.to_bytes(8, "big"))
        try:
            ns["METEOREncoder"](medium, **cfg).encode(bitarray.bitarray(stream), context=prompt)
        except CarrierBoundary as exc:
            stop = str(exc)
        tokens = medium.output
        assert len(tokens) == len(progress)
        expected = stream[:sum(progress)]
        aux.update(input_nonce=seq, finish_sent=False, is_sort=False, final_state_bits=0)
    if stop is None:
        raise ValueError("Expected a declared carrier boundary before supplied padding ran out")
    eos = bool(tokens and tokens[-1] in source.eos_ids)
    assert not any(t in source.eos_ids for t in tokens[:-1])
    codec = source.backend.codec
    text = codec.bytes_to_text(codec.tokens_to_bytes(tokens[:-1] if eos else tokens))
    packet = dict(version="bounded-upstream-v1", system=system, prompt=prompt, seq=seq,
                  source=source.manifest(), text=text, natural_eos=eos, auxiliary=aux,
                  confirmed_bits=len(expected), stop_reason=stop)
    assert stream.startswith(expected)
    return packet, dict(bits=expected, tokens=tokens, bits_count=len(expected), pid=os.getpid())


@torch.inference_mode()
def receive_carrier(source, packet, key):
    tokens = text_tokens(source, packet)
    bound = BoundSource(source, packet["prompt"])
    aux = packet["auxiliary"]
    seed = aux["seed"]
    if packet["system"].startswith("anstega_"):
        expected_rng = "hmac-sha256-bitstream-v1" if packet["system"] == "anstega_keyed" else "python-mt19937-fixture"
        if aux.get("mask_rng", "python-mt19937-fixture") != expected_rng:
            raise ValueError("ANS mask RNG contract mismatch")
        ns = anstega_paper_namespace(bound, mask_key=key if packet["system"] == "anstega_keyed" else None)
        parts = ns["decode_anstega"](None, torch.tensor([bound.prompt.token_ids]), tokens.copy(),
                                     aux["final_state"], device="cpu", precision=32, ans_t=16, seed=seed)
        raw = "".join(reversed(parts))
        if not raw.startswith(aux["public_initial_prefix"]):
            raise ValueError("ANS public initial prefix mismatch")
        bits = raw[1:]
    else:
        ns = upstream("meteor_borrowed", bound)
        progress = []
        original = ns["num_same_from_beg"]

        def common(*args):
            n = original(*args)
            progress.append(n)
            return n

        ns["num_same_from_beg"] = common
        cfg = dict(finish_sent=False, precision=32, is_sort=False, seed=seed, input_key=key,
                   sample_seed_prefix=b"bounded-meteor-v1", input_nonce=aux["input_nonce"].to_bytes(8, "big"))
        result, _ = ns["METEORDecoder"](Medium(bound), **cfg).decode(tokens.copy(), context=packet["prompt"])
        # The original decoder emits 32 bits from the last interval's lower
        # endpoint. Only its common prefix is determined by the received token.
        bits = result.to01()[:sum(progress)]
    if len(bits) != packet["confirmed_bits"]:
        raise ValueError("Confirmed prefix length mismatch")
    return dict(bits=bits, bits_count=len(bits), pid=os.getpid())


def stage(model, which):
    source = Source.load(model, "stego")
    for system in ("anstega_reviewed", "meteor_borrowed", "anstega_keyed"):
        for language, prompt in (("en", PROMPT), ("zh", ZH_PROMPT)):
            directory = OUT/model/"bounded"/(system+"__"+language)
            destination = directory/(which+".json")
            if destination.exists():
                continue
            accumulated = ""
            rows = []
            if which == "sender":
                names = []
                for seq in range(16):
                    name = system+"__"+language+f"__{seq}"
                    key = fixture_key(model, "bounded:"+name)
                    stream = frame(PAYLOAD)[len(accumulated):]+bits_for(key)
                    packet, result = send_carrier(source, system, prompt, stream, key, seq)
                    write(directory/"public"/(name+".json"), packet)
                    write(directory/"private"/(name+".json"), result)
                    names.append(name)
                    accumulated += result["bits"]
                    rows.append(dict(name=name, bits=result["bits_count"], eos=packet["natural_eos"], stop=packet["stop_reason"], final_state_bits=packet["auxiliary"]["final_state_bits"]))
                    if unframe(accumulated)["status"] == "complete":
                        break
                write(directory/"public/manifest.json", dict(model=model, system=system, language=language, packets=names))
            else:
                manifest = json.loads((directory/"public/manifest.json").read_text())
                for name in manifest["packets"]:
                    key = fixture_key(model, "bounded:"+name)
                    packet = json.loads((directory/"public"/(name+".json")).read_text())
                    assert packet["source"] == source.manifest() and packet["prompt"] == prompt
                    result = receive_carrier(source, packet, key)
                    write(directory/"decoded"/(name+".json"), result)
                    accumulated += result["bits"]
                    rows.append(dict(name=name, bits=result["bits_count"]))
            result = dict(model=model, system=system, language=language, frame=unframe(accumulated), packets=rows, pid=os.getpid())
            write(destination, result)
            print(json.dumps({k:v for k,v in result.items() if k!='packets'}), flush=True)


def summary():
    rows = []
    for model in ("qwen", "llama"):
        for directory in sorted((OUT/model/"bounded").iterdir()):
            s = json.loads((directory/"sender.json").read_text())
            r = json.loads((directory/"receiver.json").read_text())
            exact = []
            for p in s["packets"]:
                a = json.loads((directory/"private"/(p["name"]+".json")).read_text())
                b = json.loads((directory/"decoded"/(p["name"]+".json")).read_text())
                exact.append(a["bits"] == b["bits"] and a["pid"] != b["pid"])
            rows.append(dict(model=model, system=s["system"], language=s["language"], packets=len(exact),
                             exact_packets=sum(exact), frame=r["frame"],
                             public_final_state_bits=sum(p["final_state_bits"] for p in s["packets"]),
                             public_initial_prefix_bits=len(exact) if s['system'].startswith('anstega_') else 0))
    write(HERE/"bounded_results.json", dict(rows=rows))
    print(json.dumps(rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("qwen", "llama"))
    p.add_argument("--stage", choices=("sender", "receiver", "summary"), required=True)
    a = p.parse_args()
    summary() if a.stage == "summary" else stage(a.model, a.stage)
