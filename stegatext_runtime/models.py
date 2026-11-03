"""Model identities and mandatory prompt protocols, without GPU imports."""

MODEL_IDS = {
    "qwen": "Qwen/Qwen3-4B-Base",
    "llama": "meta-llama/Llama-3.2-3B",
    "qwen_instruct": "Qwen/Qwen3-4B-Instruct-2507",
    "llama_instruct": "meta-llama/Llama-3.2-3B-Instruct",
}


def prompt_mode(model_name):
    if model_name not in MODEL_IDS:
        raise ValueError(f"Unknown model {model_name!r}")
    return "chat" if model_name.endswith("_instruct") else "plain_completion"
