StegText Framework
===================

Pluggable neural text steganography with modular Source / Disambiguation / Coder / Engine components. The framework hides a bitstream in next-token choices while guaranteeing decode-side recovery from visible text only.

Overview
- Components: modular `Source` / `Disambiguation` / `Coder` / `Engine` that you can mix and match
  - `Source` turns a prompt into token candidates with probabilities (e.g., HF models via `hf_recipes`, or a toy source)
  - `Disambiguation` resolves prefix ambiguity among candidates to ensure correct decoding (e.g., `LookAhead`, `SyncPool`, `MWIS`)
  - `Coder` hides bits in the candidate probability distribution (e.g., `IMEC`, `DisCop`, `RangeAC`)
  - `Engine` composes the modules to run the end-to-end steganography process


Install
-------
Use the provided requirements (Torch, Transformers, etc.):

```
pip install -r requirements.txt
```

Then install this package (editable):

```
pip install -e .
```


Quickstart
----------
The minimal quickstart is a single script that runs both single- and multi-sentence flows and checks decode parity.

Run with a Hugging Face model id (downloads from Hub):

```
PYTHONPATH=src python scripts/quickstart.py
```

By default, `scripts/quickstart.py` uses Qwen2.5-3B-Instruct on CUDA:

```python
# [Source]
source = make_qwen25_source(
    "Qwen/Qwen2.5-3B-Instruct",
    device="cuda:0", dtype="float16", chat=True,
    temperature=1.0, top_k=16, top_p=1.0,
)

# [Disambiguator]
dis = LookAhead()   # or SyncPool() or MWIS()

# [Coder]
coder = DisCop()    # or RangeAC(...), IMEC(...)

engine = StegoEngine(source, dis, coder)
```

Use a local model path (offline):

```python
source = make_qwen25_source(
    "Qwen/Qwen2.5-3B-Instruct",  # local directory with model files
    device="cuda:0", dtype="float16", chat=True,
    temperature=1.0, top_k=16, top_p=1.0,
)
```

Notes
- The script prints the generated text, bits/token, and verifies that the recovered secret equals the embedded secret.
- To switch models, import from `stegtext.integrations.hf_recipes` (e.g., `make_llama3_8b_source`) and replace the source line.
- For a zero-dependency demo, change `source = make_toy_source()`; this uses a lightweight in-repo toy source (CPU only) but still demonstrates the full pipeline.

License
-------
MIT. See `LICENSE`.


Minimal Examples
----------------
Single sentence (embed random bits, decode prefix):

```python
from stegtext.integrations.hf_recipes import make_qwen25_source
from stegtext.components.disambiguation.lookahead import LookAhead
from stegtext.components.coding.discop import DisCop
from stegtext.core.engine import StegoEngine
from stegtext.core.rng import CSPRNG
from stegtext.core.payload import random_bitstring

prompt = "Explain quantum computing in one short sentence."
source = make_qwen25_source("Qwen/Qwen2.5-3B-Instruct", device="cuda:0", dtype="float16", chat=True,
                            temperature=1.0, top_k=16, top_p=1.0)
engine = StegoEngine(source, LookAhead(), DisCop())
rng = CSPRNG(key=b"demo-single")
bits = random_bitstring(64)

enc = engine.encode(prompt, bits, rng=rng, max_steps=128)
dec = engine.decode(prompt, enc.text, rng=rng, max_steps=128)

print(enc.text)
print(bits.startswith(dec.bits))  # True if decoded is a prefix of embedded
```

Multi sentence (embed readable secret, recover exactly):

```python
from stegtext.core.framing import pack_bits, parse_framed_bits
from stegtext.core.payload import text_to_bits, bits_to_text
from stegtext.core.rng import CSPRNG

prompt = "Write a short beautiful poem."
secret = "This is a secret message hidden within the text."
rng = CSPRNG(key=b"demo-multi")

framed = pack_bits(text_to_bits(secret), len_bits=16)
enc_seq = engine.encode_sequence(prompt, framed, rng=rng, max_steps=128, max_sentences=32)
sentences = [s.text for s in enc_seq.sentences]

dec_seq = engine.decode_sequence(prompt, sentences, rng=rng, max_steps=128)
status, recovered_bits, _ = parse_framed_bits(dec_seq.bits, len_bits=16)
print(bits_to_text(recovered_bits))  # equals `secret` when status == 'done'
```
