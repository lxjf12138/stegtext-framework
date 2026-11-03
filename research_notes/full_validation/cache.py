"""Lossless CPU snapshots of canonical KV states for branching test workloads.

This changes storage/recomputation only. Every saved prefix was evaluated by the
same fixed prompt prefill followed by single-token forwards. Copying BF16 cache
tensors preserves their bits. A restored prefix followed by the same forwards
therefore reproduces canonical replay by induction. The first 16 restorations
per process are additionally compared, using torch.equal, with an independent
unaccelerated HFReference on the complete requested history.
"""
from collections import OrderedDict
import types
import torch
from transformers.cache_utils import DynamicCache,DynamicLayer
from research_notes.inference_conformance.hf_reference import HFReference


def accelerate(source):
    ref=source.reference
    original_bind=ref.bind_prompt
    original_next=ref.next_logits
    ref.snapshots=OrderedDict()
    ref.snapshot_restorations=0
    ref.snapshot_oracle_checks=0
    oracle=HFReference(source.backend,source.config,validity_mask=source.validity)

    def bind(self,ids):
        self.snapshots.clear()
        original_bind(ids)
        oracle.bind_prompt(ids)

    @torch.inference_mode()
    def next_logits(self,ids):
        key=tuple(int(x) for x in ids)
        restored=False
        if self.key is not None and key!=self.key and not (len(key)>len(self.key) and key[:len(self.key)]==self.key):
            prefixes=[p for p in self.snapshots if key[:len(p)]==p]
            if prefixes:
                p=max(prefixes,key=len)
                layers,logits=self.snapshots[p]
                cache=DynamicCache(config=self.model.config)
                if len(cache.layers)!=len(layers): raise ValueError("Snapshot layer count mismatch")
                for layer,saved in zip(cache.layers,layers):
                    layer.__dict__.update(saved)
                    layer.keys=saved["keys"].to(self.device,copy=True)
                    layer.values=saved["values"].to(self.device,copy=True)
                self.cache,self.key,self.logits=cache,p,logits.to(self.device,copy=True)
                self.snapshot_restorations+=1
                restored=True
        logits=original_next(key)
        if restored and self.snapshot_oracle_checks<16:
            expected=oracle.next_logits(key)
            if not torch.equal(logits,expected):
                raise ValueError("Snapshot KV replay differs from canonical HF logits")
            actual_p=self.processors(torch.tensor([key],device=self.device),logits.clone()[None])[0]
            expected_p=oracle.scores(key,expected)
            if not torch.equal(actual_p,expected_p) or not torch.equal(torch.softmax(actual_p,0),torch.softmax(expected_p,0)):
                raise ValueError("Snapshot KV replay differs from canonical processed probability law")
            self.snapshot_oracle_checks+=1
        if key not in self.snapshots:
            layers=[]
            for layer in self.cache.layers:
                if type(layer) is not DynamicLayer:
                    raise ValueError("Snapshot proof is restricted to full-attention DynamicLayer")
                data=dict(layer.__dict__)
                data["keys"]=layer.keys.to("cpu",copy=True)
                data["values"]=layer.values.to("cpu",copy=True)
                layers.append(data)
            self.snapshots[key]=(layers,logits.to("cpu",copy=True))
            while len(self.snapshots)>32: self.snapshots.popitem(last=False)
        else:
            self.snapshots.move_to_end(key)
        return logits

    ref.bind_prompt=types.MethodType(bind,ref)
    ref.next_logits=types.MethodType(next_logits,ref)
    return source
