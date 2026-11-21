# Olala configuration fixture

`olala-mini.config.json` is the unmodified configuration from
[`OVHaiLLM/Olala-v1.0-mini-7A1B`](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B/blob/b40f8165ee2254c18601220e19842600c6d91794/config.json)
at revision `b40f8165ee2254c18601220e19842600c6d91794`.
SHA-256: `f2668c0a369eeb2285e21edb46dd6307c32d55aaaefa985a1abf0eeb73dbd8e3`.
The upstream model card identifies the model's license as Apache-2.0.
No model weights are included.

`olala-attention-header.json` contains only tensor names, shapes and dtypes for
layer 4's attention from the pinned checkpoint's Safetensors header. The
source header including its eight-byte length prefix has SHA-256
`81df2fdc92cc4faa6c66b9786d2103045c6b466a575b31f97151d64f653882f5`.
Extract it with `python tests/fixtures/generate_attention_header.py /path/to/header`;
no tensor values are included.

`olala-routing-reference.json` was generated with PyTorch 2.15.0 by invoking
the actual `OlalaMoE.forward` implementation from
[`modeling_olala.py`](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B/blob/b40f8165ee2254c18601220e19842600c6d91794/modeling_olala.py)
at the same pinned revision, with synthetic inputs, biases, and an identity
router projection. The generator extracts the upstream class from source and
checks its SHA-256 `357c01b34490ee8650a42a8acdd38bc186bd35863edbbeb4e2bf0746e9b31bcf`
before execution; no upstream model weights are included. To regenerate:

```sh
python tests/fixtures/generate_routing_reference.py /path/to/modeling_olala.py
```

`olala-geodesic-reference.json` uses the same verified upstream source and
PyTorch version, invoking its `OlalaGeodesicNorm.forward` directly with
synthetic vectors and scalar parameters. It covers ordinary, zero, near-zero,
collinear, upper-clamped, negative-angle, and later-layer cases. Regenerate
it with `python tests/fixtures/generate_geodesic_reference.py /path/to/modeling_olala.py`.

`olala-mamba3-reference.json` executes the pinned source's independent
PyTorch prefill and cached-step functions with synthetic projected inputs,
including one and two Q/K groups. Tests compare outputs and all four states,
chunk continuation and reset. This fixture tests the recurrence core, not the
checkpoint's projection, normalization or full model. Regenerate it with
`python tests/fixtures/generate_mamba3_reference.py /path/to/modeling_olala.py`.
