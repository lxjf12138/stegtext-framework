"""Public-packet reset/framing/error tests and matched ordinary controls."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
import traceback

import torch

from stegatext_runtime import Source
from round4_2_common import CounterRNG
from research_notes.full_validation.suite import (ROOT,HERE,OUT,PROMPT,SCENARIOS,write,log,
    fixture_key,bits_for,run,rendered,shape,goal)

POLICIES=("syncpool","las","fullsync","adaptive","syncpool_official_discop")
PAYLOAD=b"Library: 10:00."


def frame(payload):
    return f"{len(payload):032b}"+"".join(f"{b:08b}" for b in payload)+"".join(f"{b:08b}" for b in hashlib.sha256(payload).digest()[:16])


def unframe(bits):
    if len(bits)<32: return {"status":"incomplete_header"}
    length=int(bits[:32],2)
    needed=32+length*8+128
    if length>4096: return {"status":"invalid_length","declared_bytes":length}
    if len(bits)<needed: return {"status":"incomplete","needed_bits":needed,"available_bits":len(bits)}
    payload=bytes(int(bits[i:i+8],2) for i in range(32,32+length*8,8))
    expected=frame(payload)
    if bits[:needed]!=expected: return {"status":"checksum_mismatch"}
    return {"status":"complete","payload_hex":payload.hex(),"frame_bits":needed,"padding_bits":len(bits)-needed}


def envelope(source,method,prompt,spec,seq,previous):
    official=method=="syncpool_official_discop"
    return dict(version="validation-transport-v1",source=source.manifest(),method="syncpool" if official else method,coder="discop_upstream" if official else "discop",
                prompt=prompt,scenario_spec=spec,seq=seq,previous_text_sha256=previous)


def sender(model):
    source=Source.load(model,"stego")
    from research_notes.full_validation.cache import accelerate
    accelerate(source)
    for policy in POLICIES:
        for scenario in ("reset_single","reset_multi","framed_cross_packet","framed_fresh_carriers"):
            series=policy+"__"+scenario
            directory=OUT/model/"transport"/series
            if (directory/"sender.json").exists(): continue
            prompt=PROMPT
            accumulated=""
            names=[]
            records=[]
            previous=None
            framed=scenario.startswith("framed_")
            fresh=scenario=="framed_fresh_carriers"
            for seq in range(16 if fresh else 8 if framed else 3):
                name=f"{series}__{seq}"
                key=fixture_key(model,name)
                if framed:
                    spec={"cap":64}
                    stream=frame(PAYLOAD)[len(accumulated):]+bits_for(key)
                else:
                    spec={"cap":128,"sentences":1 if scenario=="reset_single" else 3}
                    stream=bits_for(key)
                packet=envelope(source,policy,prompt,spec,seq,previous)
                try:
                    result=run(source,packet,key=key,stream=stream)
                    packet.update(text=result["text"],natural_eos=result["natural_eos"])
                    write(directory/"public"/(name+".json"),packet)
                    write(directory/"private"/(name+".json"),result)
                    names.append(name)
                    records.append({k:result[k] for k in ("steps","bits_count","scenario_goal","natural_eos","shape")})
                    accumulated+=result["bits"]
                    prompt=PROMPT if fresh else prompt+result["text"]+"\n\n"
                    previous=hashlib.sha256(packet["text"].encode()).hexdigest()
                    if framed and unframe(accumulated)["status"]=="complete": break
                except Exception as exc:
                    records.append({"error":f"{type(exc).__name__}: {exc}","traceback":traceback.format_exc()})
                    break
            manifest=dict(series=series,model=model,policy=policy,scenario=scenario,packets=names,
                          prompt_schedule="fresh_independent_carrier" if fresh else "append_previous_public_text")
            write(directory/"public/manifest.json",manifest)
            result={**manifest,"records":records,"pid":os.getpid(),"frame":unframe(accumulated) if framed else None}
            write(directory/"sender.json",result)
            log(stage="transport_sender",model=model,series=series,packets=len(names),frame=result["frame"])


def assemble(manifest,packets,decoded):
    if len(packets)!=len(manifest["packets"]): return {"status":"missing_packet"}
    previous=None
    for seq,packet in enumerate(packets):
        if packet["seq"]!=seq: return {"status":"out_of_order"}
        if packet["previous_text_sha256"]!=previous: return {"status":"chain_mismatch"}
        previous=hashlib.sha256(packet["text"].encode()).hexdigest()
    return unframe("".join(decoded))


def receiver(model):
    source=Source.load(model,"stego")
    from research_notes.full_validation.cache import accelerate
    accelerate(source)
    for policy in POLICIES:
        for scenario in ("reset_single","reset_multi","framed_cross_packet","framed_fresh_carriers"):
            series=policy+"__"+scenario
            directory=OUT/model/"transport"/series
            if (directory/"receiver.json").exists(): continue
            public=directory/"public"
            if not (public/"manifest.json").exists(): continue
            manifest=json.loads((public/"manifest.json").read_text())
            packets=[]
            results=[]
            for name in manifest["packets"]:
                packet=json.loads((public/(name+".json")).read_text())
                packets.append(packet)
                try:
                    if packet["source"]!=source.manifest(): raise ValueError("Source contract mismatch")
                    result=run(source,packet,key=fixture_key(model,name))
                    write(directory/"decoded"/(name+".json"),result)
                    results.append({"status":"ok","name":name,"bits":result["bits"],"pid":os.getpid()})
                except Exception as exc:
                    results.append({"status":"error","name":name,"error":f"{type(exc).__name__}: {exc}"})
            all_ok=all(r["status"]=="ok" for r in results) and bool(results)
            framed=scenario.startswith("framed_")
            transport_frame=assemble(manifest,packets,[r["bits"] for r in results]) if all_ok and framed else None
            faults={}
            if all_ok and framed and transport_frame["status"]=="complete":
                decoded=[r["bits"] for r in results]
                faults["drop_last"]=assemble(manifest,packets[:-1],decoded[:-1])
                faults["reverse_order"]=assemble(manifest,list(reversed(packets)),list(reversed(decoded)))
                corrupted="".join(decoded)
                if len(corrupted)>40:
                    corrupted=corrupted[:40]+str(1-int(corrupted[40]))+corrupted[41:]
                faults["flip_payload_bit"]=unframe(corrupted)
                first=packets[0]
                name=manifest["packets"][0]
                for fault in ("wrong_key","truncate_text","edit_text"):
                    packet=dict(first)
                    key=fixture_key(model,name)
                    if fault=="wrong_key": key=hashlib.sha256(key+b"wrong").digest()
                    if fault=="truncate_text": packet.update(text=first["text"][:len(first["text"])//2],natural_eos=False)
                    if fault=="edit_text": packet.update(text=" INVALID"+first["text"],natural_eos=False)
                    try:
                        result=run(source,packet,key=key)
                        bits=[result["bits"]]+decoded[1:]
                        faults[fault]=unframe("".join(bits))
                    except Exception as exc:
                        faults[fault]={"status":"receiver_rejected","error":f"{type(exc).__name__}: {exc}"}
            result={**manifest,"results":results,"all_packets_decoded":all_ok,"frame":transport_frame,"faults":faults,"pid":os.getpid()}
            write(directory/"receiver.json",result)
            log(stage="transport_receiver",model=model,series=series,all_decoded=all_ok,frame=transport_frame)


@torch.inference_mode()
def ordinary(model):
    source=Source.load(model,"stego")
    for scene,spec in SCENARIOS.items():
        for seed in (11,29):
            path=OUT/model/"ordinary"/(scene+f"__{seed}.json")
            if path.exists(): continue
            p=source.bind(spec["prompt"])
            ids=list(p.token_ids)
            rng=CounterRNG(fixture_key(model,f"ordinary:{scene}:{seed}"),"ordinary",0)
            generated=[]
            start=time.monotonic()
            stop="budget"
            for step in range(spec["cap"]+32):
                probs=source.probabilities(ids).double().cpu()
                cdf=probs.cumsum(0)
                token=int(torch.searchsorted(cdf,rng.random()*float(cdf[-1])))
                generated.append(token)
                ids.append(token)
                eos=token in source.eos_ids
                try:
                    text=source.backend.codec.bytes_to_text(source.backend.codec.tokens_to_bytes(generated[:-1] if eos else generated))
                except Exception: continue
                if eos or goal(text,spec) or step+1>=spec["cap"]:
                    stop="eos" if eos else "text_boundary" if goal(text,spec) else "budget"
                    break
            result=dict(model=model,scene=scene,seed=seed,prompt=spec["prompt"],text=text,token_ids=generated,
                        natural_eos=eos,stop_reason=stop,scenario_goal=goal(text,spec),shape=shape(text),source=source.manifest(),wall_seconds=time.monotonic()-start)
            write(path,result)
            log(stage="ordinary",model=model,scene=scene,seed=seed,tokens=len(generated),eos=eos)


def summary():
    rows=[]
    for model in ("qwen","llama"):
        for policy in POLICIES:
            for scenario in ("reset_single","reset_multi","framed_cross_packet","framed_fresh_carriers"):
                series=policy+"__"+scenario
                directory=OUT/model/"transport"/series
                if not (directory/"sender.json").exists() or not (directory/"receiver.json").exists():
                    rows.append(dict(model=model,policy=policy,scenario=scenario,packets=0,exact_packets=0,all_exact=False,frame=None,faults={},status="missing"))
                    continue
                s=json.loads((directory/"sender.json").read_text())
                r=json.loads((directory/"receiver.json").read_text())
                exact=[]
                for name in s["packets"]:
                    a=json.loads((directory/"private"/(name+".json")).read_text())
                    path=directory/"decoded"/(name+".json")
                    b=json.loads(path.read_text()) if path.exists() else {}
                    exact.append(all(a[k]==b.get(k) for k in ("bits","text","state_sha256","trace")) and a["pid"]!=b.get("pid"))
                rows.append(dict(model=model,policy=policy,scenario=scenario,packets=len(exact),exact_packets=sum(exact),
                                 all_exact=bool(exact) and all(exact),frame=r["frame"],faults=r["faults"],
                                 text_goals_met=sum(t.get("scenario_goal",False) for t in s["records"])))
    write(HERE/"transport_results.json",{"rows":rows})
    log(series=len(rows),exact_packets=sum(r["exact_packets"] for r in rows),packets=sum(r["packets"] for r in rows))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model",choices=("qwen","llama"))
    p.add_argument("--stage",choices=("sender","receiver","ordinary","summary"),required=True)
    a=p.parse_args()
    if a.stage=="summary": summary()
    else: globals()[a.stage](a.model)
