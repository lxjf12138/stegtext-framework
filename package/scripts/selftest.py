#!/usr/bin/env python3
import pathlib, subprocess, sys, os
root=pathlib.Path(__file__).resolve().parents[1]
env=dict(os.environ); env['PYTHONPATH']=str(root/'src') + (os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
for suite in (root/'tests/core', root/'tests/starter'):
    cmd=[sys.executable,'-m','unittest','discover','-s',str(suite),'-p','test_*.py','-q']
    code=subprocess.call(cmd,env=env,cwd=root)
    if code: raise SystemExit(code)
print("all starter selftests passed")
