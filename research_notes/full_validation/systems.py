"""Pinned complete upstream coding loops on the audited real LM source.

The legacy model frontend alone is replaced. Token-channel and text-channel
receivers are separate stages; auxiliary information is explicitly public.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import hmac
import importlib.util
import json
import math
from math import ceil, log2
import os
from pathlib import Path
import random
import sys
import time
import traceback

import bitarray
import numpy as np
import scipy
from scipy.stats import entropy
import torch

from stegatext_runtime import Source
from research_notes.full_validation.suite import ROOT,HERE,OUT,PROMPT,ZH_PROMPT,bits_for,fixture_key,write,log,shape
from research_notes.full_validation.upstream import extract


class BoundSource:
    """No sampler override: positive Source law, CPU64 normalization for kernels."""
    def __init__(self,source,prompt,cap=512):
        self.source=source
        self.prompt=source.bind(prompt)
        self.cap=cap
        self.calls=0
        self.normalization_tv_max=0.

    def distribution(self,ids):
        self.calls+=1
        if self.calls>self.cap:
            raise RuntimeError("512-query diagnostic watchdog reached")
        if len(ids)>len(self.prompt.token_ids) and ids[-1] in self.source.eos_ids:
            raise RuntimeError("Upstream loop attempted inference beyond natural EOS")
        p=self.source.probabilities(ids).double().cpu()
        ids=torch.nonzero(p>0).flatten()
        probs=p[ids]
        norm=probs/probs.sum()
        self.normalization_tv_max=max(self.normalization_tv_max,float((norm-probs).abs().sum()/2))
        order=torch.argsort(norm,descending=True,stable=True)
        return norm[order],ids[order]

    def get_probs_past(self,model,prev,past,device='cpu',top_p=1.):
        suffix=tuple(int(t) for t in prev.flatten().tolist())
        ids=suffix if past is None else tuple(past)+suffix
        p,actions=self.distribution(ids)
        return p,actions,ids


class Medium:
    def __init__(self,bound):
        self.bound=bound
        self.enc=bound.source.tokenizer
        self.device="cpu"
        self.probs_top_k=50

    def encode_context(self,context):
        return list(self.bound.prompt.token_ids)

    def reset(self,context=None):
        self.output=[]
        self.ids=list(self.bound.prompt.token_ids)
        return self.step(None)

    def step(self,action):
        if action is not None:
            token=int(action)
            self.output.append(token)
            self.ids.append(token)
        p,ids=self.bound.distribution(self.ids)
        self.action_labels=ids
        values=p.numpy().astype(np.longdouble)
        # The original Medium returns a NumPy entropy scalar. Its diagnostics
        # intentionally allow 0/0 -> NaN at a zero-entropy step. Converting it
        # to Python float would turn that statistics-only division into an
        # artificial ZeroDivisionError, before the coding loop can continue.
        h=np.longdouble(entropy(values.astype(float),base=2))
        info={"indices":ids.numpy(),"log_probs":np.log(values),"log_probs_T=1":np.log(values),
              "medium_entropy":h,"medium_entropy_raw":h,"end_of_line":False}
        return values,info

    def get_output(self,clean_up=False): return self.output.copy()
    def humanify(self,output): return self.enc.decode(output,skip_special_tokens=False,clean_up_tokenization_spaces=False)


def namespace(bound):
    return dict(torch=torch,th=torch,np=np,random=random,time=time,entropy=entropy,
                ceil=ceil,log2=log2,math=math,hashlib=hashlib,hmac=hmac,bitarray=bitarray,
                defaultdict=defaultdict,deepcopy=deepcopy,scipy=scipy,
                get_probs_past=bound.get_probs_past)


def upstream(name,bound):
    ns=namespace(bound)
    if name=="sparsamp":
        base=ROOT/"external_baselines/SparSamp/Artifact new/Basic Test"
        extract(base/"utils.py",ns,{"func_mrn","get_lower_upper_bound","dec2bin"})
        extract(base/"sparsamp.py",ns)
    elif name.startswith("anstega"):
        base=ROOT/"external_baselines/ANStega/Artifact_ANStega"
        extract(base/"utils.py",ns,{"generate_random_mask_bitstring","xor_low_bits"})
        extract(base/"anstega_1031.py",ns)
    elif name in ("imec_official","meteor_borrowed"):
        base=ROOT/"external_baselines/iMEC/src"
        extract(base/"utils.py",ns,{"kl2","entropy2","DRBG","num_same_from_beg","bin_sort","bits2int","int2bits","is_sent_finish"})
        extract(base/"mec.py",ns)
        extract(base/"medium.py",ns,{"ActionLabelException"})
        extract(base/("imec.py" if name=="imec_official" else "meteor.py"),ns)
    return ns


class RRCStepper:
    def __init__(self,bound,rrc): self.bound,self.rrc=bound,rrc
    def prefill(self,context): self.ids=list(context.flatten().tolist())
    def step(self,token): self.ids.append(int(token))
    def distribution(self,k):
        p,ids=self.bound.distribution(self.ids)
        return self.rrc.StepDistribution(p,ids,min(k,len(p)))


def load_rrc():
    spec=importlib.util.spec_from_file_location("validation_rrc",ROOT/"external_baselines/RRC/rrc_core.py")
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def arimec(bound,merge,secret=None,tokens=None,seed=0):
    sys.path.insert(0,str(ROOT/"external_baselines/ARIMEC"))
    from mec import ARIMEC
    class Bits:
        branching_factor=2
        def is_terminal(self,prefix): return len(prefix)==16
        def conditional(self,prefix): return np.array([.5,.5])
        def evaluate(self,z): return -16*math.log(2) if len(z)==16 else -math.inf
        def sample(self):
            z=np.random.randint(0,2,16).tolist()
            return z,self.evaluate(z)
    class Language:
        branching_factor=bound.source.backend.codec.vocab_size
        def __init__(self):
            self.paths={():list(bound.prompt.token_ids)}
            self.tables={}
        def table(self,prefix):
            key=tuple(prefix)
            if key not in self.tables:
                if key not in self.paths:
                    _,ids=self.table(prefix[:-1])
                    self.paths[key]=self.paths[tuple(prefix[:-1])]+[int(ids[prefix[-1]])]
                p,ids=bound.distribution(self.paths[key])
                self.tables[key]=(p.numpy(),ids.tolist())
            return self.tables[key]
        def conditional(self,prefix): return self.table(prefix)[0]
        def is_terminal(self,prefix):
            if not prefix: return False
            self.table(prefix[:-1])
            last=self.tables[tuple(prefix[:-1])][1][prefix[-1]]
            return len(prefix)>=64 or last in bound.source.eos_ids
        def evaluate(self,z):
            return sum(math.log(self.conditional(z[:i])[a]) for i,a in enumerate(z))
        def sample(self):
            y=[]
            while not self.is_terminal(y): y.append(int(np.random.choice(len(self.conditional(y)),p=self.conditional(y))))
            return y,self.evaluate(y)
    nu=Language()
    algorithm=ARIMEC(Bits(),nu,merge=merge)
    np.random.seed(seed)
    if secret is not None:
        y,logp=algorithm.sample_y_given_x([int(c) for c in secret])
        generated=[nu.table(y[:i])[1][a] for i,a in enumerate(y)]
        return generated,{"conditional_log_probability":float(logp)}
    y=[]
    for token in tokens:
        _,ids=nu.table(y)
        if token not in ids: raise ValueError("Retokenized token outside current support")
        y.append(ids.index(token))
    x,confidence=algorithm.estimate_x_given_y(y)
    return "".join(map(str,x)),{"map_log_probability":float(confidence)}


def system_cases():
    for name in ("sparsamp","anstega","anstega_normalized","imec_official","meteor_borrowed","rrc_verified","rrc_paper_stop","arimec","arimec_merge"):
        for seed in (11,29):
            yield dict(case=f"{name}__{seed}",system=name,seed=seed,prompt=PROMPT if seed==11 else ZH_PROMPT)


@torch.inference_mode()
def send(source,item,key):
    b=BoundSource(source,item["prompt"])
    name=item["system"]
    seed=item["seed"]
    supplied=bits_for(key)
    if name=="anstega_normalized":
        supplied="1"+supplied  # Public normalization bit; count it as overhead.
    context=torch.tensor([b.prompt.token_ids])
    aux={}
    if name.startswith("rrc_"):
        rrc=load_rrc()
        rrc.set_precision(64)
        result=rrc.embed_message(RRCStepper(b,rrc),context,supplied[:64],random.Random(seed),64,max_tokens=256,verify=name=="rrc_verified")
        tokens=result["token_ids"]
        expected=supplied[:64]
        aux={"message_bits":64,"verify":name=="rrc_verified","retry_count":result["extra_tokens"]}
    elif name.startswith("arimec"):
        expected=supplied[:16]
        tokens,stats=arimec(b,name=="arimec_merge",secret=expected,seed=seed)
        aux={"message_bits":16,"max_output_tokens":64,**stats}
    else:
        ns=upstream(name,b)
        if name=="sparsamp":
            tokens,chunks,*_=ns["encode_spar"](None,list(b.prompt.token_ids),supplied,48,device="cpu",block_size=16,random_seed=seed)
            expected="".join(chunks)
            aux={"block_size":16,"minimum_output_tokens":48}
        elif name.startswith("anstega"):
            tokens,final_state,chunks,*_=ns["encode_anstega"](None,context,supplied,64,device="cpu",precision=32,ans_t=16,seed=seed)
            expected="".join(chunks)
            initial=int(supplied[:37],2)
            aux={"final_state":final_state,"final_state_bits":final_state.bit_length(),"precision":32,"ans_t":16,
                 "initial_state_normalized":2**36<=initial<2**37,"public_initial_prefix_bits":1 if name=="anstega_normalized" else 0}
        else:
            medium=Medium(b)
            if name=="imec_official":
                enc=ns["IMECEncoder"](medium,block_size=4,seed=seed,clean_up_output=False)
                _,tokens,stats=enc.encode(bitarray.bitarray(supplied[:64]),context=item["prompt"])
                expected=supplied[:64]
                aux={"block_size":4,"n_chunks":16,"message_bits":64}
            else:
                cfg=dict(finish_sent=False,precision=32,is_sort=False,seed=seed,input_key=key,
                         sample_seed_prefix=b"validation-meteor",input_nonce=seed.to_bytes(8,"big"))
                enc=ns["METEOREncoder"](medium,**cfg)
                _,tokens,stats=enc.encode(bitarray.bitarray(supplied[:64]),context=item["prompt"])
                expected=supplied[:64]
                aux={"message_bits":64,"precision":32,"finish_sent":False,"is_sort":False}
    eos=bool(tokens and tokens[-1] in source.eos_ids)
    if any(t in source.eos_ids for t in tokens[:-1]): raise ValueError("Output contains internal EOS")
    codec=source.backend.codec
    data=codec.tokens_to_bytes(tokens[:-1] if eos else tokens)
    text_error=None
    try:
        text=codec.bytes_to_text(data)
    except Exception as exc:
        text=None
        text_error=f"{type(exc).__name__}: {exc}"
    text_packet={**item,"source":source.manifest(),"text":text,"natural_eos":eos,"auxiliary":aux}
    return text_packet,{"status":"ok","tokens":tokens,"bits":expected,"bits_count":len(expected),"shape":shape(text or ""),"text_error":text_error,
                        "text":text,"queries":b.calls,"normalization_tv_max":b.normalization_tv_max,"pid":os.getpid()}


def text_tokens(source,packet):
    # Standard receiver-view retokenization, including the public prompt boundary.
    p=source.backend.render_prompt(packet["prompt"])
    whole=source.tokenizer.encode(packet["prompt"]+packet["text"],add_special_tokens=True)
    if whole[:len(p.token_ids)]!=list(p.token_ids):
        raise ValueError("Retokenization changed the public prompt boundary")
    generated=whole[len(p.token_ids):]
    if packet["natural_eos"]: generated.append(min(source.eos_ids))
    return generated


@torch.inference_mode()
def receive(source,packet,tokens,key):
    b=BoundSource(source,packet["prompt"])
    name=packet["system"]
    seed=packet["seed"]
    aux=packet["auxiliary"]
    context=torch.tensor([b.prompt.token_ids])
    info={}
    if name.startswith("rrc_"):
        rrc=load_rrc()
        rrc.set_precision(aux["message_bits"])
        recovered=rrc.extract_message(RRCStepper(b,rrc),context,tokens,aux["message_bits"],random.Random(seed),64)
    elif name.startswith("arimec"):
        recovered,info=arimec(b,name=="arimec_merge",tokens=tokens,seed=seed)
    else:
        ns=upstream(name,b)
        if name=="sparsamp":
            recovered="".join(ns["decode_spar"](None,tokens,list(b.prompt.token_ids),device="cpu",block_size=16,random_seed=seed))
        elif name.startswith("anstega"):
            chunks=ns["decode_anstega"](None,context,tokens.copy(),aux["final_state"],device="cpu",precision=32,ans_t=16,seed=seed)
            recovered="".join(reversed(chunks))
        elif name=="imec_official":
            dec=ns["IMECDecoder"](Medium(b),block_size=4,n_chunks=16)
            result,_=dec.decode(tokens,context=packet["prompt"])
            recovered=result.to01()
        else:
            cfg=dict(finish_sent=False,precision=32,is_sort=False,seed=seed,input_key=key,
                     sample_seed_prefix=b"validation-meteor",input_nonce=seed.to_bytes(8,"big"))
            dec=ns["METEORDecoder"](Medium(b),**cfg)
            result,_=dec.decode(tokens.copy(),context=packet["prompt"],private_message_bitlen=64)
            recovered=result.to01()
    return dict(status="ok",bits=recovered,bits_count=len(recovered),queries=b.calls,pid=os.getpid(),**info)


def stage(model,which):
    source=Source.load(model,"stego")
    for item in system_cases():
        name=item["case"]
        dest=OUT/model/"systems"/which/(name+".json")
        if dest.exists(): continue
        key=fixture_key(model,"systems:"+name)
        start=time.monotonic()
        try:
            if which=="sender":
                packet,result=send(source,item,key)
                if packet["text"] is not None:
                    write(OUT/model/"systems/public_text"/(name+".json"),packet)
                write(OUT/model/"systems/public_token"/(name+".json"),{**packet,"token_ids":result["tokens"]})
            else:
                channel="public_token" if which=="receiver_token" else "public_text"
                path=OUT/model/"systems"/channel/(name+".json")
                if not path.exists(): continue
                packet=json.loads(path.read_text())
                if packet["source"]!=source.manifest(): raise ValueError("Source contract mismatch")
                tokens=packet["token_ids"] if which=="receiver_token" else text_tokens(source,packet)
                result=receive(source,packet,tokens,key)
        except Exception as exc:
            result={"status":"error","error":f"{type(exc).__name__}: {exc}","traceback":traceback.format_exc(),"pid":os.getpid()}
        result={**item,**result,"wall_seconds":time.monotonic()-start}
        write(dest,result)
        log(model=model,stage=which,case=name,**{k:result[k] for k in ("status","error","bits_count","wall_seconds") if k in result})


def summary():
    rows=[]
    for model in ("qwen","llama"):
        for item in system_cases():
            name=item["case"]
            records={stage:json.loads(p.read_text()) if (p:=OUT/model/"systems"/stage/(name+".json")).exists() else {} for stage in ("sender","receiver_token","receiver_text")}
            s=records["sender"]
            row={"model":model,**item,"sender_status":s.get("status","missing")}
            for channel in ("token","text"):
                r=records["receiver_"+channel]
                row[channel+"_status"]=r.get("status","unavailable_text" if channel=="text" and s.get("text_error") else "missing")
                row[channel+"_roundtrip"]=s.get("status")==r.get("status")=="ok" and s["bits"]==r["bits"] and s["pid"]!=r["pid"]
                row[channel+"_error"]=r.get("error",s.get("error") or (s.get("text_error") if channel=="text" else None) or ("" if row[channel+"_roundtrip"] else "bit mismatch or missing result"))
                if s.get("status")==r.get("status")=="ok" and s["bits"]!=r["bits"]:
                    a,b=s["bits"],r["bits"]
                    at=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
                    row[channel+"_error"]=f"bit mismatch: expected {len(a)} bits, recovered {len(b)}; first difference at {at}"
            row.update({k:s[k] for k in ("bits_count","queries","shape","normalization_tv_max","wall_seconds") if k in s})
            rows.append(row)
    write(HERE/"systems_results.json",{"rows":rows})
    log(systems=len(rows),token_roundtrips=sum(r["token_roundtrip"] for r in rows),text_roundtrips=sum(r["text_roundtrip"] for r in rows))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model",choices=("qwen","llama"))
    p.add_argument("--stage",choices=("sender","receiver_token","receiver_text","summary"),required=True)
    a=p.parse_args()
    if a.stage=="summary": summary()
    else: stage(a.model,a.stage)
