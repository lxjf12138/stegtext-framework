StegText Framework
===================

Pluggable neural text steganography with modular Source / Disambiguation / Coder / Engine components. The framework hides a bitstream in next‑token choices while guaranteeing decode‑side recovery from visible text only.

Highlights
- Modular: swap `Source` (LLM or toy), `Disambiguation` (lookahead/syncpool/MWIS), and `Coder` (IMEC/DisCop/RangeAC)
- Deterministic round‑trip: encode/decode share a scoped CSPRNG; visible bytes drive decode
- Byte‑level EOS handling: unify model EOS as a sentinel to avoid false stops


Install
-------
Clone this repo and install in editable mode:

```
pip install -e .
```

Required: Python 3.9+. The core package depends on `numpy` and `torch`. If you plan to use HF models, also install `transformers` (and a CUDA‑enabled PyTorch if you want GPU):

```
pip install transformers
# optional: install a CUDA build of torch matching your system
```


Quick Start (CLI)
-----------------
We provide a single‑sentence demo script that runs end‑to‑end encode→decode and verifies bit parity.

1) Toy source (zero extra deps, CPU)
```
python scripts/quickstart.py \
  --source toy --coder imec --dis lookahead \
  --prompt "Say hi" --message "secret" --max-steps 64
```

2) HF sources (requires transformers and a model path or Hub id)
- Qwen2.5 Instruct via Hub id (small and easy to try):
```
CUDA_VISIBLE_DEVICES=0 \
python scripts/quickstart.py \
  --source qwen25 --model "Qwen/Qwen2.5-3B-Instruct" \
  --coder rangeac --dis syncpool \
  --prompt "Say hi" --message "secret" --max-steps 32
```
- Qwen3 4B (if you have access to this model locally or via Hub):
```
CUDA_VISIBLE_DEVICES=0 \
python scripts/quickstart.py \
  --source qwen3 --model "Qwen/Qwen3-4B" \
  --coder rangeac --dis syncpool \
  --prompt "Say hi" --message "secret" --max-steps 32
```

The script prints visible text and reports whether emitted/decoded bits match ("Round‑trip: OK"). Whether the full message is recovered depends on the step budget and if EOS is hit early. See `scripts/hf_source_recipes.py` for other HF sources: `qwen25`, `qwen3`, `deepseek`, `glm4`, `llama3`.


Minimal Programmatic Examples
-----------------------------
A) Toy source (no external model)
```python
from stegtext.core.engine import StegoEngine
from stegtext.core.framing import pack_bits, parse_framed_bits
from stegtext.core.rng import CSPRNG

from stegtext.components.disambiguation.lookahead import LookAhead
from stegtext.components.coding.imec import IMEC

from stegtext.components.source.base import EOS_STEGA
from stegtext.components.source.vocab import TokenByteVocab
from stegtext.components.source.toy import ToySource, ToySourceConfig

# Byte-level vocab: 0..255 map to single bytes, 256 is EOS
mapping = {i: bytes([i]) for i in range(256)}
mapping[256] = EOS_STEGA
vocab = TokenByteVocab(mapping=mapping, vocab_size=257)
# Minimal transition spec: from any prefix, allow a few bytes + EOS
transitions = {(): ((32, 0.4), (97, 0.2), (98, 0.2), (46, 0.1), (33, 0.08), (256, 0.02))}
source = ToySource(ToySourceConfig(vocab=vocab, transitions=transitions, end_token_ids=(256,)))

# Build modules
dis = LookAhead(); dis.init()
coder = IMEC(block_size=6); coder.init()
engine = StegoEngine(source, dis, coder)
rng = CSPRNG(key=b'demo-seed')

# Frame payload bits
payload = b'secret'
bits = ''.join(f'{b:08b}' for b in payload)
framed = pack_bits(bits, len_bits=16)

# Encode then decode back
enc = engine.encode('Say hi', framed, rng=rng, max_steps=64)
dis.init(); coder.init()
dec = engine.decode('Say hi', enc.text, rng=rng, max_steps=64)

print('Visible:', enc.text)
print('Parity OK:', enc.emitted_bits == dec.bits)
status, recovered_bits, _ = parse_framed_bits(dec.bits, len_bits=16)
```

B) HF source via recipe (example: Qwen2.5 Instruct)
```python
from stegtext.core.engine import StegoEngine
from stegtext.core.framing import pack_bits
from stegtext.core.rng import CSPRNG
from stegtext.components.disambiguation.lookahead import LookAhead
from stegtext.components.coding.range_ac import RangeAC
from scripts.hf_source_recipes import make_qwen25_source

source = make_qwen25_source('Qwen/Qwen2.5-3B-Instruct', device='cuda:0', dtype='float16', chat=True)

dis = LookAhead(); dis.init()
coder = RangeAC(precision=40); coder.init()
engine = StegoEngine(source, dis, coder)
rng = CSPRNG(key=b'demo-seed')

bits = ''.join(f'{b:08b}' for b in b'secret')
enc = engine.encode('Say hi', pack_bits(bits, 16), rng=rng, max_steps=32)
```


Available Modules
-----------------
- Sources
  - `toy` (in‑repo deterministic toy source)
  - HF sources via `scripts/hf_source_recipes.py`: `qwen25`, `qwen3`, `deepseek`, `glm4`, `llama3`
- Disambiguation
  - `lookahead`: prefix‑group planning with representative expansion
  - `syncpool`: sample within group and expand
  - `mwis`: maximum‑weight independent set over prefix‑conflict graph
- Coders
  - `imec`: iterative minimum‑entropy coupling (block posterior)
  - `discop`: Huffman + double‑U(0,1) gating
  - `rangeac`: fixed‑precision arithmetic coder


Parity Matrix (optional)
------------------------
For encode/decode parity over many combos and prompts, use:
```
PYTHONPATH=src python scripts/test_matrix_encode_decode.py \
  --sources qwen3 --coders rangeac discop imec --disambiguators lookahead syncpool mwis \
  --prompt "Say hi" --max-steps 32
```


Notes & Tips
------------
- Decode uses byte‑level visible text (via a unified EOS sentinel), ensuring robust prefix matching across tokenizers.
- All coders rely on an external RNG provided by the engine with scoped derivation to keep encode/decode in lockstep.
- For HF models, ensure your PyTorch build matches your CUDA driver if you want GPU acceleration.


License
-------
MIT. See `LICENSE`.
