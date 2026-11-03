# Look-ahead Sync (LAS) for neural linguistic steganography

Code and Lean 4 proofs for the paper

> Y. Feng, F. Jiang, S. Wu, Z. Lu, and H. Zhong, "A High-Capacity and Secure
> Disambiguation Algorithm for Neural Linguistic Steganography."

LAS(τ) resolves tokenization ambiguity in distribution-preserving linguistic
steganography. After the coder selects a group of candidate histories, LAS
compares the group's uncertainty U with a public threshold τ. If U ≥ τ, it
synchronizes only the histories that end at the group's identifying prefix and
carries longer histories forward for later payload coding; otherwise it
synchronizes the whole group, as SyncPool does. LAS(0) always retains, and
τ = ∞ gives SyncPool on the same planner.

## Repository layout

| Path | Contents |
|---|---|
| `package/research_runtime/perf_research_selective.py` | LAS(τ) planner: candidate grouping, retaining and whole-group updates, uncertainty U |
| `stegatext_runtime/` | model sources and chat prompting, iMEC coder (`imec.py`), sender and receiver engine (`experiments/engine_v2.py`), token budget, perplexity and KL scoring |
| `stegatext_runtime/experiments/baseline_adapters.py` | Basic, MWIS, and Stepwise Verification on the same sources and coder |
| `research_notes/` | method registry and reference model backend used by the runtime |
| `package/src/stegtext/v2/` | protocol types, coders, and methods shared by the runtime |
| `tools/review_main_*_20261023.py` | preparation, execution, scoring, and analysis of the main comparison |
| `tools/` (other files) | modules used by the drivers and tests: record archiving, perplexity scoring, targeted repeats, capacity diagnostics, and finite reference oracles |
| `configs/experiments/` | experiment configurations |
| `tests/`, `package/tests/` | unit, protocol, and sender-receiver tests |
| `formal/` | Lean 4 formalization of the analysis; see [formal/README.md](formal/README.md) |
| `analysis/` | scripts that derived the paper's tables and figures from the run records |
| `environment/` | environment setup and activation, pinned model revisions |

## Methods

The main comparison (Section V) uses these configurations on both models:

| Paper | `method` in the configuration |
|---|---|
| LAS(τ=0) | `las` |
| LAS(τ=4) | `adaptive` with `tau = 4.0` |
| SyncPool | `syncpool` with `implementation = "shared_planner_v1"`: the LAS planner with every update synchronizing the whole group, not the original SyncPool code |
| Stepwise | `stepwise_chat_v1`: calls the unchanged candidate predicate of the upstream [Consistency](https://github.com/ryehr/Consistency) repository |
| MWIS | `mwis_bytes_v1` |
| Basic | `basic_bytes_v1` |

All methods use the coder `imec_prefix_exact` with block size 6, window size
3, and greedy coupling. Sampling is top-k (k = 16, 32, 128, 256) or the full
vocabulary at temperature 1, with a 256-token budget per history.

## Setup

Requirements: Linux, Python 3.12, [uv](https://docs.astral.sh/uv/), git, and a
CUDA GPU. The paper's timings were measured on one NVIDIA RTX 4070 SUPER
(12 GB) with PyTorch 2.8.0 and Transformers 4.57.6.

```bash
bash environment/sync.sh              # create .venv from uv.lock
source environment/activate.sh        # PYTHONPATH, caches, and model paths
bash scripts/fetch_baselines.sh       # upstream Consistency at the pinned commit
```

`uv.lock` resolves packages from the TUNA PyPI mirror and the PyTorch CUDA 12.8
index. Download the two source models at the pinned revisions in
`environment/models.lock.json`; Llama 3.2 requires accepting Meta's license on
Hugging Face:

```bash
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct \
  --revision 0cb88a4f764b7a12671c53f0838cd831a0843b95 --local-dir models/Llama-3.2-3B-Instruct
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 --local-dir models/Qwen3-4B-Instruct-2507
```

## Tests

```bash
python -m pytest tests package/tests
```

Some tests load the tokenizers of the two models. Four tests replay the
prepared main-comparison inputs or recorded runs and are skipped when those
records are absent. The CPU contract suite cited in Section V (181 tests) is

```bash
python -m pytest tests/test_finite_source_law.py tests/test_adaptive_thresholds.py \
  tests/test_token_budget.py tests/test_imec_exact_contract.py tests/test_imec_terminal_prefix.py \
  tests/test_planner_precision.py tests/test_las_interval_cache.py tests/test_discop_reward.py \
  -k "not historical_64_case_terminal_prefix_vector"
```

## Running the methods

A short end-to-end check runs ordinary sampling and the six methods on one
prompt per model with 16 body tokens. Every steganographic case runs an
independent receiver that must recover the delivered payload prefix exactly.

```bash
python -m stegatext_runtime experiment plan --config configs/experiments/main_table_smoke.toml
python -m stegatext_runtime experiment run --config configs/experiments/main_table_smoke.toml --output-root runs/smoke
```

The same `experiment run` command accepts any configuration in
`configs/experiments/`.

## Main comparison

The main comparison was run with `tools/review_main_run_20261023.py`:

```bash
python tools/review_main_run_20261023.py prepare     # freeze the design from the prepared inputs
python tools/review_main_run_20261023.py preflight   # check inputs and code fingerprints without loading models
python tools/review_main_run_20261023.py run         # sender and receiver workers, scoring; resumable
python tools/review_main_run_20261023.py analyze
```

Its inputs were 32 English documents from `stanfordnlp/imdb` (revision
`e6281661ce1c48d982bc483cf8a173c1bbeb5d31`, train split) and 32 Chinese
documents from `GT610/douban` (revision
`d1c97233eb7bb0db8417cd6ef731c411707e12ef`, train split), selected and
converted into run configurations by `tools/review_main_prepare_20261023.py`,
`tools/review_main_gt610_prepare_20261023.py`, and
`tools/review_main_uniform_prepare_20261023.py`. The prepared document set, the
experiment randomness key, and the run records are not distributed with this
repository. The two confirmation batches were run with
`tools/review_main_followup_20261025.py`. The scripts in `analysis/` show how
the records were turned into the reported tables and figures.

## Lean formalization

`formal/` machine-checks, for finite token sources and an ideal coder, the
distribution preservation of LAS(τ) for every threshold (Theorem 1), the bound
0 ≤ d ≤ U of Eq. (14), and the threshold budget comparison with SyncPool
(Theorem 2). It does not cover concrete coders, floating-point execution, or
the Python implementation. Build instructions and the theorem correspondence
are in [formal/README.md](formal/README.md).

## License

The code in this repository is released under the MIT License. The upstream
baselines, models, and datasets are distributed under their own licenses.
