# All-in-One Colab Fine-Tune

**`qwen36_27b_finetune_colab.ipynb`** is the single-file deliverable: upload
it to Google Colab (96GB-VRAM GPU runtime), optionally adjust the
CONFIGURATION cell, and `Runtime → Run all`. It performs the entire
pipeline — environment checks, dependency install, dataset download and
Hermes-format conversion, the mask-verification gate, bf16 LoRA SFT with
Unsloth (checkpointed and auto-resuming, to Google Drive by default), merge
to bf16, GGUF export, and optional Hugging Face upload.

## Do not edit the notebook directly

It is **generated**. The embedded pipeline code is read verbatim from
`training/trainkit/` at build time so it always matches the unit-tested
sources. To change anything:

```bash
# edit colab/build_notebook.py (notebook cells) or training/trainkit/ (pipeline)
python colab/build_notebook.py     # regenerate
python colab/validate_notebook.py  # verify (JSON, syntax, module identity, config drift)
```

`validate_notebook.py` fails the build if an embedded module ever drifts
from its tested source, if any cell has a syntax error, or if a
configuration knob is used but no longer defined.

## What the notebook assumes

- A Colab GPU runtime with ~96GB VRAM / ~180GB RAM / ~200GB disk. On less
  VRAM, set `LOAD_IN_4BIT = True` (QLoRA fallback).
- `MODEL_ID = "Qwen/Qwen3.6-27B"` — verify the exact repo id on hf.co
  before the run.
- Data is downloaded from public, non-gated Hugging Face datasets and
  converted defensively (per-source yield reported, licenses recorded in
  `data/manifest.json`). A `personal.jsonl` uploaded to `/content/data/`
  is mixed in automatically as the specialization slice.

## Relationship to the rest of the repo

The notebook is the "just works" packaging of the same pipeline:
`training/` holds the script-based version (plus the ORPO stage and the
QLoRA config), `evals/` holds the Phase 6 evaluation suite for judging the
result against the stock model.
