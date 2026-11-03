"""Expose frozen source in the venv without writing metadata into package/."""
from pathlib import Path
import site
import sys

root = Path(__file__).resolve().parents[1]
if Path(sys.prefix).resolve() != (root / ".venv").resolve():
    raise SystemExit("Run with the project's .venv Python")
dest = Path(site.getsitepackages()[0]) / "stegatext_workspace.pth"
dest.write_text(str(root / "package/src") + "\n" + str(root / "package/research_runtime") + "\n")
print("Linked frozen project sources through", dest)
