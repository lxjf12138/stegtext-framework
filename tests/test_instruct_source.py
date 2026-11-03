"""Official local chat templates and receiver-only native text reconstruction."""
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer, GenerationConfig

from stegatext_runtime.source import ROOT, Source
from stegatext_runtime.experiments.config import cases, resolve
from stegatext_runtime.experiments.transport import attach, text_tokens
from stegtext.v2.model.hf_backend import HFBackend
from stegtext.v2.model.prompt import ChatPromptPolicy, PlainCompletionPromptPolicy


@pytest.fixture(params=["qwen_instruct", "llama_instruct"])
def source(request):
    import json
    from stegatext_runtime.models import MODEL_IDS
    name = request.param
    spec = next(s for s in json.loads((ROOT/"environment/models.lock.json").read_text())["models"] if s["repo_id"] == MODEL_IDS[name])
    folder = ROOT/spec["local_path"]
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    model = torch.nn.Linear(1, 1)
    model.generation_config = GenerationConfig.from_pretrained(folder, local_files_only=True)
    model.config = SimpleNamespace(vocab_size=max(tokenizer.get_vocab().values())+1)
    eos = model.generation_config.eos_token_id
    backend = HFBackend(tokenizer, model, device="cpu", eos_ids=eos,
                        prompt_policy=ChatPromptPolicy(template_options=spec["template_options"]))
    return Source(backend, spec, name, "stego")


def test_official_chat_and_receiver_prefix(source):
    prompt, answer = "请介绍图书馆。", "图书馆提供阅读空间。"
    ids = list(source.backend.render_prompt(prompt).token_ids)
    expected = source.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, **source.spec["template_options"])
    assert ids == expected
    assert source.retokenize_text(prompt, answer) == source.tokenizer.encode(answer, add_special_tokens=False)
    assert source.tokenizer.encode(prompt+answer, add_special_tokens=True)[:len(ids)] != ids
    assert source.manifest()["prompt_mode"] == "chat"
    assert source.manifest()["version"] == 2
    assert source.manifest()["template_options"] == source.spec["template_options"]
    if source.model_name == "llama_instruct":
        assert ids.count(source.tokenizer.bos_token_id) == 1
        assert "date_string" in source.spec["template_options"]


def test_native_terminal_identity_is_public_and_checked(source):
    prompt, answer = "Write one sentence.", "A library is open."
    tokens = source.retokenize_text(prompt, answer)
    for terminal in sorted(source.eos_ids):
        packet = attach(source, dict(prompt=prompt, text=answer, natural_eos=True), {"tokens": tokens+[terminal]})
        assert text_tokens(source, packet) == tokens+[terminal]
        assert packet["text_transport"]["eos_index_bits"] in (1, 2)
        assert "tokens" not in packet
    packet["text_transport"]["eos_index"] = len(source.eos_ids)
    with pytest.raises(ValueError, match="EOS index"):
        text_tokens(source, packet)
    with pytest.raises(ValueError, match="missing"):
        text_tokens(source, dict(prompt=prompt, text=answer, natural_eos=False))


def test_instruct_cannot_silently_fall_back_to_completion(source):
    source.backend.prompt_policy = PlainCompletionPromptPolicy()
    with pytest.raises(ValueError, match="explicit chat"):
        Source(source.backend, source.spec, source.model_name)


def test_instruct_experiment_uses_locked_chat_models(tmp_path):
    config = resolve(ROOT/"configs/experiments/instruct_quality.toml")
    assert set(config["source"]["models"]) == {"qwen_instruct", "llama_instruct"}
    assert all(s["prompt_mode"] == "chat" for s in config["model_revisions"].values())
    path = tmp_path/"bad.toml"
    path.write_text(f'extends = "{ROOT}/configs/experiments/instruct_quality.toml"\n[source]\nprompt_mode="plain_completion"\n')
    with pytest.raises(ValueError, match="mandatory model protocol"):
        resolve(path)


@pytest.mark.parametrize("name,count", [("instruct_quality", 16), ("instruct_single", 8), ("instruct_methods", 24)])
def test_model_language_policy_applies_to_every_expanded_case(name, count):
    config = resolve(ROOT/f"configs/experiments/{name}.toml")
    expanded = list(cases(config))
    assert len(expanded) == count
    for item in expanded:
        expected = "zh" if item["model"] == "qwen_instruct" else "en"
        assert item["prompt"]["language"] == expected
        if expected == "zh":
            assert "中文" in item["prompt"]["text"]
        else:
            assert item["prompt"]["text"].isascii()


def test_empty_language_assignment_rejected(tmp_path):
    path = tmp_path/"bad.toml"
    path.write_text(f'extends = "{ROOT}/configs/experiments/instruct_quality.toml"\n[inputs]\nprompts=[{{id="en",language="en",text="Write in English."}}]\n')
    with pytest.raises(ValueError, match="No zh prompt"):
        resolve(path)
