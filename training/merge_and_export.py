#!/usr/bin/env python3
"""Merge the SFT LoRA adapter into a bf16 checkpoint, then (optionally)
export GGUF quants one at a time under the 200GB storage budget.

Usage:
    python merge_and_export.py --config configs/sft_plan_a.yaml \\
        --adapter runs/sft_plan_a/final_adapter \\
        --output artifacts/merged_sft_bf16

    # After verifying the merge, reclaim ~55GB:
    python merge_and_export.py ... --delete-hf-cache-after

    # GGUF exports, sequential, guarded:
    python merge_and_export.py ... --gguf q4_k_m q6_k \\
        --offload-cmd 'rclone move {path} remote:models/'

Storage-aware behavior (the Phase 1 budget makes this mandatory, not nice
to have):
    * Guard BEFORE merging: merged bf16 needs ~55GB free. A merge that runs
      out of disk mid-write corrupts the output.
    * Guard BEFORE each GGUF: each quant is exported, then optionally
      offloaded (your rclone/scp command) and deleted locally before the
      next quant starts, so at most one quant occupies disk at a time.
    * ``--delete-hf-cache-after`` removes the base-model hub cache after a
      successful merge is verified on disk.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trainkit.config import load_config
from trainkit.storage import estimate_dir_gb, require_free_gb

# Rough GGUF sizes for a 27B model, used only for pre-flight disk guards.
GGUF_EST_GB = {
    "q4_k_m": 18.0,
    "q5_k_m": 21.0,
    "q6_k": 24.0,
    "q8_0": 31.0,
    "f16": 56.0,
}
MERGED_BF16_EST_GB = 58.0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", required=True, help="path to the trained LoRA adapter")
    ap.add_argument("--output", default="artifacts/merged_sft_bf16")
    ap.add_argument("--gguf", nargs="*", default=[], choices=sorted(GGUF_EST_GB),
                    help="quant types to export sequentially")
    ap.add_argument("--gguf-dir", default="artifacts/gguf")
    ap.add_argument("--llama-cpp-dir", default=None,
                    help="path to a llama.cpp checkout (for convert + quantize); "
                         "required when --gguf is used")
    ap.add_argument("--offload-cmd", default=None,
                    help="command run after each GGUF export; '{path}' is replaced "
                         "with the file path. On success the local file is deleted.")
    ap.add_argument("--delete-hf-cache-after", action="store_true",
                    help="delete the HF hub cache of the base model after a verified merge")
    ap.add_argument("--skip-merge", action="store_true",
                    help="reuse an existing --output merge (e.g. GGUF-only invocation)")
    return ap.parse_args()


def merge(cfg, adapter: Path, output: Path) -> None:
    require_free_gb(output.parent if output.parent.exists() else ".",
                    MERGED_BF16_EST_GB, "bf16 merge")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[merge] loading base {cfg.model.name_or_path} in bf16 "
          "(device_map=cpu — merging needs RAM, not VRAM; 180GB is plenty)")
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=cfg.model.trust_remote_code,
    )
    print(f"[merge] applying adapter {adapter}")
    model = PeftModel.from_pretrained(base, str(adapter))
    model = model.merge_and_unload()

    output.mkdir(parents=True, exist_ok=True)
    print(f"[merge] writing merged bf16 to {output}")
    model.save_pretrained(str(output), safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(str(adapter), trust_remote_code=True)
    tok.save_pretrained(str(output))


def verify_merge(output: Path) -> None:
    size_gb = estimate_dir_gb(output)
    has_weights = any(output.glob("*.safetensors"))
    has_config = (output / "config.json").is_file()
    if not (has_weights and has_config and size_gb > 10):
        raise SystemExit(
            f"[merge] verification FAILED at {output} "
            f"(weights={has_weights}, config={has_config}, size={size_gb:.1f}GB). "
            "Not deleting anything; inspect before retrying."
        )
    print(f"[merge] verified: {size_gb:.1f} GB on disk")


def delete_hf_cache(model_name: str) -> None:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    slug = "models--" + model_name.replace("/", "--")
    target = cache_root / slug
    if target.is_dir():
        freed = estimate_dir_gb(target)
        shutil.rmtree(target)
        print(f"[storage] deleted HF cache {target} ({freed:.1f} GB freed)")
    else:
        print(f"[storage] no HF cache at {target}; nothing to delete")


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("[exec] " + " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def export_ggufs(args, merged: Path) -> None:
    llama = Path(args.llama_cpp_dir) if args.llama_cpp_dir else None
    if llama is None:
        raise SystemExit("--gguf requires --llama-cpp-dir pointing at a llama.cpp checkout")
    convert_py = llama / "convert_hf_to_gguf.py"
    quantize_bin = next(
        (p for p in (llama / "build" / "bin" / "llama-quantize",
                     llama / "llama-quantize") if p.exists()),
        None,
    )
    if not convert_py.exists():
        raise SystemExit(f"missing {convert_py} — clone/build llama.cpp first")

    gguf_dir = Path(args.gguf_dir)
    gguf_dir.mkdir(parents=True, exist_ok=True)

    # One f16 intermediate feeds every quant; it is the largest artifact, so
    # guard for it plus the largest requested quant.
    f16_path = gguf_dir / "model-f16.gguf"
    quants = [q for q in args.gguf if q != "f16"]
    largest_quant = max((GGUF_EST_GB[q] for q in quants), default=0.0)
    require_free_gb(gguf_dir, GGUF_EST_GB["f16"] + largest_quant, "GGUF conversion")

    if not f16_path.exists():
        run([sys.executable, str(convert_py), str(merged),
             "--outfile", str(f16_path), "--outtype", "f16"])

    for quant in quants:
        if quantize_bin is None:
            raise SystemExit("llama-quantize binary not found — build llama.cpp "
                             "(cmake -B build && cmake --build build)")
        out_path = gguf_dir / f"model-{quant}.gguf"
        require_free_gb(gguf_dir, GGUF_EST_GB[quant], f"quantize {quant}")
        run([str(quantize_bin), str(f16_path), str(out_path), quant.upper()])
        print(f"[gguf] {out_path} written ({estimate_dir_gb(out_path.parent):.1f} GB dir total)")

        if args.offload_cmd:
            cmd = args.offload_cmd.replace("{path}", str(out_path))
            print(f"[offload] {cmd}")
            subprocess.run(cmd, shell=True, check=True)
            out_path.unlink()
            print(f"[offload] {out_path.name} offloaded and deleted locally")

    if "f16" not in args.gguf:
        f16_path.unlink(missing_ok=True)
        print("[gguf] removed f16 intermediate")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    adapter = Path(args.adapter)
    output = Path(args.output)

    if not args.skip_merge:
        if not (adapter / "adapter_config.json").is_file():
            raise SystemExit(f"{adapter} does not look like a LoRA adapter "
                             "(no adapter_config.json)")
        merge(cfg, adapter, output)

    verify_merge(output)

    if args.delete_hf_cache_after:
        delete_hf_cache(cfg.model.name_or_path)

    if args.gguf:
        export_ggufs(args, output)

    print("[done] merge/export complete")
    print(f"[next] ORPO pass: python train_orpo.py --config configs/orpo_plan_a.yaml "
          f"(orpo.merged_model_path should point at {output})")


if __name__ == "__main__":
    main()
