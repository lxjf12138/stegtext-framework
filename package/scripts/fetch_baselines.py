#!/usr/bin/env python3
"""Print verified baseline locations; deliberately performs no network action."""
import json, pathlib
root=pathlib.Path(__file__).resolve().parents[1]
data=json.loads((root/'references/repos.json').read_text())
for r in data['repositories']:
    print(f"{r['name']}: {r['url']}  [{r['status']}]")
print()
print("Clone/download into external_baselines/, record resolved commit/hash, run official tests, then create an adapter. Do not silently port first.")
