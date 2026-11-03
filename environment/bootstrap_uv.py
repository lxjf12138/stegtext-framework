"""Reinstall the pinned project-local uv from its verified PyPI wheel."""
from pathlib import Path
import hashlib
import io
import json
import urllib.request
import zipfile

root = Path(__file__).resolve().parents[1]
spec = json.loads((root / "environment/uv-bootstrap.json").read_text())
with urllib.request.urlopen(spec["wheel_url"], timeout=120) as response:
    data = response.read()
if hashlib.sha256(data).hexdigest() != spec["sha256"]:
    raise SystemExit("uv wheel SHA256 mismatch")
dest = root / spec["install_location"]
dest.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(data)) as wheel:
    for name in ("uv", "uvx"):
        matches = [p for p in wheel.namelist() if p.endswith("/scripts/" + name)]
        if len(matches) != 1:
            raise SystemExit("Unexpected uv wheel layout")
        target = dest / name
        target.write_bytes(wheel.read(matches[0]))
        target.chmod(0o755)
print("Installed project-local uv", spec["version"])
