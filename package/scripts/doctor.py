#!/usr/bin/env python3
import json, platform, sys
out={"python":sys.version,"platform":platform.platform()}
try:
 import torch
 out.update(torch=torch.__version__,cuda_available=torch.cuda.is_available(),cuda_version=torch.version.cuda)
 if torch.cuda.is_available():
  p=torch.cuda.get_device_properties(0); out["gpu"]={"name":p.name,"total_memory_bytes":int(p.total_memory)}
except Exception as e: out["torch_error"]=repr(e)
try:
 import transformers; out["transformers"]=transformers.__version__
except Exception as e: out["transformers_error"]=repr(e)
print(json.dumps(out,indent=2,ensure_ascii=False))
