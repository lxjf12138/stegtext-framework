#!/usr/bin/env python3
import hashlib, pathlib
root=pathlib.Path(__file__).resolve().parents[1]
ignore={'.git','__pycache__'}; rows=[]
for p in sorted(root.rglob('*')):
 if not p.is_file() or any(x in ignore for x in p.parts): continue
 if p.name in {'PACKAGE_SHA256SUMS.txt'}: continue
 h=hashlib.sha256(p.read_bytes()).hexdigest(); rows.append((h,p.relative_to(root).as_posix()))
for h,n in rows: print(f'{h}  {n}')
