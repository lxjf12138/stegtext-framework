"""Adapters around pinned upstream cores; no mathematical AST edits."""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import math
from pathlib import Path
import random
import sys
import types

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def extract(path, namespace, names=None):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and
             (names is None or n.name in names)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


class PRG:
    def __init__(self, rng): self.rng = rng
    def generate_random(self, n): return self.rng.random()


class Liao:
    def __init__(self, variant):
        self.variant = variant
        base = ROOT/"external_baselines/Liao_framework/Artifacts/baselines"
        self.ns = dict(torch=torch, math=math, random=random)
        extract(base/"utils.py", self.ns, {"lsb_bits2int", "lsb_int2bits"})
        extract(base/"uni_stego.py", self.ns)
        extract(base/"recombination.py", self.ns)
        extract(base/"encode.py", self.ns)
        extract(base/"decode.py", self.ns)

    def prepare(self, probs):
        # Keep CUDA for the upstream stability module's explicit .cuda() calls.
        p, ids = torch.sort(probs.double().to("cuda" if self.variant=="stability" else "cpu"), descending=True)
        return p/p.sum(), ids

    def encode(self, probs, stream, *, rng):
        p, ids = self.prepare(probs)
        token, n = self.ns[self.variant+"_based_encoder"](p,ids,stream,0,PRG(rng))
        return stream[:n],int(token.item())

    def decode(self, probs, action, *, rng):
        p, ids = self.prepare(probs)
        return self.ns[self.variant+"_based_decoder"](p,ids,torch.tensor(action,device=p.device),PRG(rng))


def load_discop():
    # Compile the complete original file in an ignored build directory. Its
    # old model imports are unused by encode_step/decode_step, so supply import
    # shims that reject calls instead of loading obsolete model dependencies.
    from types import SimpleNamespace
    modules = {}
    def unsupported(*a, **kw): raise RuntimeError("Legacy model frontend is outside this adapter")
    for name, attrs in {
        "config": dict(text_default_settings=None,image_default_settings=None,audio_default_settings=None),
        "model": dict(get_model=unsupported,get_tokenizer=unsupported,get_feature_extractor=unsupported),
        "utils": dict(get_probs_indices_past=unsupported,set_seed=unsupported,SingleExampleOutput=unsupported),
    }.items():
        modules[name] = sys.modules.get(name)
        shim=types.ModuleType(name)
        shim.__dict__.update(attrs)
        sys.modules[name]=shim
    try:
        builds=list((ROOT/"runs/full_validation_v1/build/discop").glob("stega_cy*.so"))
        if len(builds)!=1:
            raise RuntimeError("Compile upstream DisCop with build_upstream.py first")
        spec=importlib.util.spec_from_file_location("stega_cy",builds[0])
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name,value in modules.items():
            if value is None: sys.modules.pop(name,None)
            else: sys.modules[name]=value


class OfficialDiscop:
    def __init__(self):
        self.module = load_discop()
        self.settings=types.SimpleNamespace(algo="Discop")

    def order(self, probs):
        p, ids=torch.sort(probs.double(),descending=True)
        return ids.tolist(),(p/p.sum()).tolist()

    def encode(self, probs, stream, *, rng):
        ids,p=self.order(probs)
        # Upstream uses module random.random; restore it after this step.
        old=self.module.random.random
        try:
            self.module.random.random=rng.random
            result=self.module.encode_step(self.settings,ids,p,stream)
        finally:
            self.module.random.random=old
        return stream[:result.n_bits],result.sampled_index

    def decode(self, probs, action, *, rng):
        ids,p=self.order(probs)
        old=self.module.random.random
        try:
            self.module.random.random=rng.random
            return self.module.decode_step(self.settings,ids,p,action)
        finally:
            self.module.random.random=old


def upstream_coder(name, mode):
    if name.startswith("liao_"): return Liao(name[5:])
    if name=="discop_upstream": return OfficialDiscop()
    raise ValueError(name)
