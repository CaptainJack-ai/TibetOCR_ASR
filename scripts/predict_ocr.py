"""Transcribe one or more Tibetan page images.

  python scripts/predict_ocr.py data/ocr/images/0187.png
  python scripts/predict_ocr.py scan.jpg --engine tesseract
  python scripts/predict_ocr.py page1.png page2.png --temperature 0.4 --num-samples 3 --json

`--temperature > 0` re-samples the page a few times and keeps the cleanest output,
which is the documented fallback for the rare repetition loop that survives DRY.
"""
from __future__ import annotations

import os
import sys

# Must happen before transformers is imported: skip all Hub metadata calls.
if "--offline" in sys.argv:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ocr import TesseractEngine, YigdzinEngine, pick_device  # noqa: E402


def loop_score(text: str, n: int = 12) -> int:
    """How many times the most frequent n-gram repeats (1 = no repetition)."""
    grams = [text[i: i + n] for i in range(max(0, len(text) - n + 1))]
    return max(Counter(grams).values()) if grams else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="page image path(s)")
    ap.add_argument("--engine", default="yigdzin", choices=["yigdzin", "tesseract"])
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy; 0.4 = loop-recovery sampling")
    ap.add_argument("--num-samples", type=int, default=3, help="samples kept when temperature > 0")
    ap.add_argument("--no-dry", action="store_true", help="disable the DRY anti-loop guard")
    ap.add_argument("--offline", action="store_true", help="skip HuggingFace Hub metadata calls (models already cached)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = args.dtype if args.dtype != "auto" else ("bf16" if device == "mps" else "fp32")
    if args.engine == "yigdzin":
        engine = YigdzinEngine(device, dtype, args.max_new_tokens, not args.no_dry)
    else:
        engine = TesseractEngine()

    results = []
    for img in args.images:
        path = Path(img)
        if not path.exists():
            print(f"skip (missing): {path}", file=sys.stderr)
            continue
        t0 = time.time()
        samples = engine(path, temperature=args.temperature, num_samples=args.num_samples)
        elapsed = time.time() - t0
        best = min(samples, key=lambda t: (loop_score(t), -len(t)))
        results.append({
            "file": str(path),
            "text": best,
            "seconds": round(elapsed, 2),
            "chars": len(best),
            "loop_score": loop_score(best),
            "n_samples": len(samples),
        })
        if not args.json:
            print(f"\n=== {path} ({elapsed:.1f}s, {len(best)} chars, loop_score {loop_score(best)}) ===")
            print(best)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
