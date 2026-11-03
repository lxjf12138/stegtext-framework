"""Small, explicit paper-grounded corrections outside frozen source trees."""
from __future__ import annotations

import ast
import hashlib
import hmac
import types

import torch

from research_notes.full_validation.suite import PoolMethod, ROOT
from research_notes.full_validation.upstream import Liao


class ShapeSafeStepwise(PoolMethod):
    """Algorithm 1 still tests exactly retokenize(detokenize(h+c)) == h+c."""
    def __init__(self, source):
        super().__init__(source, "consistency")
        from tokenization_consistency import pools
        from tokenization_consistency.tokenization import candidate_consistency_mask

        def vector_mask(tokenizer, prefix_ids, candidate_ids, batch_size=1024):
            prefix = prefix_ids.reshape(-1).tolist() if isinstance(prefix_ids, torch.Tensor) else list(prefix_ids)
            candidates = candidate_ids.reshape(-1).tolist() if isinstance(candidate_ids, torch.Tensor) else list(candidate_ids)
            result = candidate_consistency_mask(tokenizer, prefix, candidates, batch_size)
            return result.to(candidate_ids.device) if isinstance(candidate_ids, torch.Tensor) else result

        original = pools._pool_consistency
        namespace = dict(original.__globals__, candidate_consistency_mask=vector_mask)
        corrected = types.FunctionType(original.__code__, namespace, original.__name__, original.__defaults__)

        def build(strategy, tokenizer, prefix, probabilities, top_k, **kwargs):
            if strategy != "consistency":
                raise ValueError("This adapter is restricted to Stepwise")
            return corrected(tokenizer, prefix, probabilities, top_k, **kwargs)

        self.build_pool = build


class SingletonSafeLiao(Liao):
    """p=(1): Section 4.2.3 gives one mass-1 singleton bin, zero bits."""
    def __init__(self):
        super().__init__("stability")

    def encode(self, probs, stream, *, rng):
        if probs.numel() == 1 and float(probs[0]) > 0:
            return "", 0
        return super().encode(probs, stream, rng=rng)

    def decode(self, probs, action, *, rng):
        if probs.numel() == 1 and float(probs[0]) > 0:
            if int(action) != 0:
                raise ValueError("Invalid singleton action")
            return ""
        return super().decode(probs, action, rng=rng)


class HMACMaskBits:
    """Keyed bit stream for Algorithm 5's CSPRNG, under the HMAC PRF assumption."""
    def __init__(self, key, seed):
        if len(key) < 16:
            raise ValueError("ANS masks require at least 16 bytes of shared key")
        self.key = key
        self.domain = b"stegatext:anstega-mask:v1:" + int(seed).to_bytes(8, "big")
        self.counter = 0
        self.bits = ""

    def choice(self, alphabet):
        if alphabet != "01":
            raise ValueError("This adapter supplies binary mask bits only")
        if not self.bits:
            block = hmac.new(self.key, self.domain+self.counter.to_bytes(8, "big"), hashlib.sha256).digest()
            self.counter += 1
            self.bits = "".join(f"{b:08b}" for b in block)
        bit, self.bits = self.bits[0], self.bits[1:]
        return bit


def anstega_paper_namespace(bound, *, mask_key=None):
    """Execute original AST with precisely Algorithm 2's receiver threshold.

    ceil(t*p*M) -> t*ceil(p*M). No other arithmetic or control-flow edits.
    Invalid initial states and EOS limitations are diagnosed separately.
    """
    from research_notes.full_validation.systems import upstream
    namespace = upstream("anstega", bound)
    path = ROOT / "external_baselines/ANStega/Artifact_ANStega/anstega_1031.py"
    tree = ast.parse(path.read_text())
    old = "ceil(ans_t * probs_s_list[i] * M)"
    matches = 0

    class Threshold(ast.NodeTransformer):
        def visit_Call(self, node):
            nonlocal matches
            if ast.unparse(node) == old:
                matches += 1
                return ast.copy_location(ast.parse("ans_t * ceil(probs_s_list[i] * M)", mode="eval").body, node)
            return self.generic_visit(node)

    decoder = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "decode_anstega")
    decoder = Threshold().visit(decoder)
    assert matches == 1, "Pinned ANStega source no longer matches the reviewed correction"
    exec(compile(ast.fix_missing_locations(ast.Module(body=[decoder], type_ignores=[])), str(path), "exec"), namespace)
    if mask_key is not None:
        namespace["random"] = types.SimpleNamespace(Random=lambda seed: HMACMaskBits(mask_key, seed))
    return namespace
