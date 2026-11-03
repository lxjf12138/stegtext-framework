# Source from bash at the repository root: source environment/activate.sh
# Source this file in a shell dedicated to this project.
_stegatext_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$_stegatext_root/.venv/bin/activate" ]; then
    . "$_stegatext_root/.venv/bin/activate"
fi
export STEGTEXT_PROJECT_ROOT="$_stegatext_root"
export PATH="$_stegatext_root/.tools/bin:$PATH"
export UV_CACHE_DIR="$_stegatext_root/.cache/uv"
export HF_HOME="$_stegatext_root/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$_stegatext_root/.cache"
export TORCH_HOME="$_stegatext_root/.cache/torch"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$_stegatext_root:$_stegatext_root/package/src:$_stegatext_root/package/research_runtime"
export STEGTEXT_V2_REPO="$_stegatext_root/package"
export STEGTEXT_QWEN_MODEL_PATH="$_stegatext_root/models/Qwen3-4B-Base"
export STEGTEXT_LLAMA_MODEL_PATH="$_stegatext_root/models/Llama-3.2-3B"
export TOKENIZERS_PARALLELISM=false
export UV_HTTP_TIMEOUT=120
export UV_CONCURRENT_DOWNLOADS=8
export HF_HUB_DISABLE_TELEMETRY=1
# These domestic endpoints measured much faster without the inherited proxy.
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}modelscope.cn,.modelscope.cn,hf-mirror.com,.hf-mirror.com,pypi.tuna.tsinghua.edu.cn,download.pytorch.org,.pytorch.org"
export no_proxy="$NO_PROXY"
unset _stegatext_root
