
from __future__ import annotations
from typing import Any, Dict
import json, pathlib, time

class Hook:
    def on_step(self, event: Dict[str, Any]): ...
    def on_finish(self): ...

class JSONLTraceHook(Hook):
    """Write per-step events as JSONL for offline analysis."""
    def __init__(self, path: str):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("w", encoding="utf-8")
        self._count = 0
        self._t0 = time.time()
    def on_step(self, event: Dict[str, Any]):
        self._count += 1
        event = dict(event)
        event.setdefault("ts_rel", time.time() - self._t0)
        self.f.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.f.flush()
    def on_finish(self):
        try: self.f.close()
        except: pass


class TokenParityHook(Hook):
    """Collect per-step source prefix token-ids for encode/decode and report first divergence."""
    def __init__(self):
        # sentence_index -> step -> ids
        self._enc: dict[int, dict[int, list[int]]] = {}
        self._dec: dict[int, dict[int, list[int]]] = {}
        self._first_div: tuple[int, int] | None = None

    def on_gen_prefix(self, **event):
        mode = event.get("mode")
        s = int(event.get("sentence_index", 0))
        step = int(event.get("step", -1))
        ids = list(event.get("prefix_ids") or [])
        if mode == "encode":
            self._enc.setdefault(s, {})[step] = ids
        elif mode == "decode":
            self._dec.setdefault(s, {})[step] = ids

        # Try detect immediate divergence when both sides present
        if s in self._enc and s in self._dec and step in self._enc[s] and step in self._dec[s]:
            if self._first_div is None and self._enc[s][step] != self._dec[s][step]:
                self._first_div = (s, step)

    def on_finish(self):
        if self._first_div is not None:
            s, step = self._first_div
            enc_ids = self._enc.get(s, {}).get(step, [])
            dec_ids = self._dec.get(s, {}).get(step, [])
            print(f"[TOKEN-PARITY] first divergence at sentence={s} step={step}")
            print(f"  encode ids tail: {enc_ids[-10:]}")
            print(f"  decode ids tail: {dec_ids[-10:]}")
        else:
            # Optionally show a short summary
            total_pairs = 0
            matched = 0
            for s, enc_steps in self._enc.items():
                for step, ids in enc_steps.items():
                    total_pairs += 1
                    if self._dec.get(s, {}).get(step) == ids:
                        matched += 1
            print(f"[TOKEN-PARITY] no divergence detected; matched={matched}/{total_pairs} steps")


class PlanDebugHook(Hook):
    """Print last decode on_plan state when decode cannot extend current text.

    Engine will emit an `on_decode_no_extension` event before raising; this hook
    prints a concise summary of the last plan to help diagnose why no group key
    can extend the current visible prefix toward the target.
    """
    def __init__(self, sample: int = 5, show_tail: int = 80):
        self.sample = int(sample)
        self.show_tail = int(show_tail)
        self._last_event = None

    def on_plan(self, **event):
        # Keep a rolling pointer to the latest decode plan for context
        if event.get("mode") == "decode":
            self._last_event = event

    def on_decode_no_extension(self, **event):
        sent = event.get("sentence_index")
        step = event.get("step")
        cur = event.get("cur_text") or ""
        target = event.get("target_text") or ""
        keys = list(event.get("group_keys") or [])
        print(f"[PLAN-DEBUG] decode no-extension at sent={sent} step={step}")
        print(f"  cur tail: {repr(cur[-self.show_tail:])}")
        print(f"  target tail: {repr(target[-self.show_tail:])}")
        print(f"  groups={len(keys)} (showing up to {self.sample})")
        for i, k in enumerate(keys[: self.sample]):
            k = k or ""
            print(f"   - key[{i}] head: {repr(k[:self.show_tail])}")
            print(f"     key[{i}] tail: {repr(k[-self.show_tail:])}")
        # Also show last on_plan snapshot if available
        if self._last_event and self._last_event is not event:
            le = self._last_event
            lk = list(le.get("group_keys") or [])
            print(f"  last on_plan snapshot: sent={le.get('sentence_index')} step={le.get('step')} groups={len(lk)}")
            for i, k in enumerate(lk[: self.sample]):
                k = k or ""
                print(f"   · last key[{i}] tail: {repr(k[-self.show_tail:])}")
