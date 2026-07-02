"""trainkit — Phase 4 training infrastructure for the Qwen3.6-27B agentic assistant.

Modules:
    config       YAML-driven run configuration with hard validation.
    chat_format  ChatML rendering of Hermes-format conversations into
                 loss-labelled segments.
    masking      Segment-wise tokenization -> (input_ids, labels) with
                 response-only loss.
    packing      First-fit-decreasing sequence packing with per-example
                 position_id reset.
    data         End-to-end dataset build: JSONL -> tokenized/packed rows.
    mask_check   Pre-flight verification that loss lands only on assistant
                 tokens. Hard gate before any GPU-hours are spent.
    storage      Disk-space guards and checkpoint retention for the
                 200GB-constrained training box.
    resume       Discovery and integrity-checking of resumable checkpoints.

Everything in this package is importable without torch/unsloth/trl so the
full test suite runs on CPU-only machines. GPU frameworks are imported only
inside the top-level entrypoint scripts (train_sft.py, train_orpo.py,
merge_and_export.py).
"""

__version__ = "0.1.0"
