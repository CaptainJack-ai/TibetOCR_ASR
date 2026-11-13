"""Run Tibetan OCR models over the BDRC open benchmark pages.

Engines:
  yigdzin    - BDRC/tibetan-ocr (Yigdzin 1, 0.8B, PaddleOCR-VL architecture)
  tesseract  - Tesseract 5 + tessdata_best 'bod'

Example:
  python scripts/run_ocr.py --engine yigdzin --split pilot
  python scripts/run_ocr.py --engine tesseract --split sample
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from common import OCR_DIR, OUT_DIR, TESSDATA, dump_json, get_logger, load_json

log = get_logger("run_ocr")

MODEL_ID = "BDRC/tibetan-ocr"
PROMPT = "Extract all Tibetan text. Preserve line breaks."


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_pages(split: str) -> list[dict]:
    meta = pd.read_csv(OCR_DIR / "metadata.csv", dtype={"id": str}).set_index("id")
    if split == "all":
        ids = meta.index.tolist()
    else:
        ids = load_json(OCR_DIR / "sample_ids.json")[split]
    return [{"id": i, **meta.loc[i].to_dict()} for i in ids]


class YigdzinEngine:
    """BDRC/tibetan-ocr inference (sequential image-token M-RoPE regime)."""

    def __init__(self, device: str, dtype: str, max_new_tokens: int, use_dry: bool, use_cache: bool = True):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.use_cache = use_cache
        self.dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[dtype]
        log.info("loading %s (device=%s dtype=%s)", MODEL_ID, device, dtype)
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        kwargs = {"dtype": self.dtype}
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                MODEL_ID, attn_implementation="sdpa", **kwargs
            )
        except TypeError:
            self.model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **kwargs)
        self.model.to(device).eval()

        self.dry = None
        if use_dry:
            sys.path.insert(0, str(Path(__file__).parent / "vendor"))
            from dry_logits_processor import make_hf_dry_processor  # noqa: PLC0415

            self._make_dry = make_hf_dry_processor
            log.info("DRY anti-loop guard enabled")

    def __call__(self, image_path: Path, temperature: float = 0.0, num_samples: int = 1) -> list[str]:
        """Transcribe one page image; returns one string per sample."""
        from transformers import LogitsProcessorList

        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
        inputs = {
            k: (v.to(self.device, dtype=self.dtype) if torch.is_floating_point(v) else v.to(self.device))
            for k, v in inputs.items()
        }
        # required: sequential 1-D image-token positions instead of the grid regime
        inputs["mm_token_type_ids"] = torch.zeros_like(inputs["input_ids"])

        # The checkpoint ships generation_config.use_cache=false, which makes decoding
        # O(n^2) and ~10x slower; enabling KV cache gives byte-identical output.
        gen_kwargs = {"do_sample": temperature > 0, "max_new_tokens": self.max_new_tokens,
                      "use_cache": self.use_cache}
        if temperature > 0:
            gen_kwargs.update(temperature=temperature, num_return_sequences=max(1, num_samples))
        if self.dry is not None:
            gen_kwargs["logits_processor"] = LogitsProcessorList(
                [
                    self._make_dry(
                        prompt_len=inputs["input_ids"].shape[1],
                        multiplier=0.8,
                        base=1.75,
                        allowed_length=12,
                    )
                ]
            )
        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[:, inputs["input_ids"].shape[1] :]
        return [t.strip() for t in self.processor.batch_decode(new_tokens, skip_special_tokens=True)]


class TesseractEngine:
    def __init__(self, psm: int = 6):
        self.psm = psm
        if not (TESSDATA / "bod.traineddata").exists():
            raise SystemExit("missing tessdata/bod.traineddata")

    def __call__(self, image_path: Path, temperature: float = 0.0, num_samples: int = 1) -> list[str]:
        cmd = [
            "tesseract",
            str(image_path),
            "stdout",
            "-l",
            "bod",
            "--tessdata-dir",
            str(TESSDATA),
            "--psm",
            str(self.psm),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.strip()[:500])
        return [res.stdout.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["yigdzin", "tesseract"], required=True)
    ap.add_argument("--split", choices=["pilot", "sample", "all"], default="sample")
    ap.add_argument("--limit", type=int, default=0, help="cap number of pages (0 = no cap)")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--psm", type=int, default=6)
    ap.add_argument("--use-dry", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="disable KV cache (slow, for verification)")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    pages = load_pages(args.split)
    if args.limit:
        pages = pages[: args.limit]
    run_id = args.run_id or f"{args.engine}_{args.split}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = OUT_DIR / "ocr" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    dtype = args.dtype
    if dtype == "auto":
        dtype = "bf16" if device == "mps" else "fp32"
    if args.engine == "yigdzin":
        engine = YigdzinEngine(device, dtype, args.max_new_tokens, args.use_dry, not args.no_cache)
    else:
        engine = TesseractEngine(args.psm)
        device, dtype = "cpu", "n/a"

    rows, t0 = [], time.time()
    for i, page in enumerate(pages, 1):
        img = OCR_DIR / "images" / f"{page['id']}.png"
        start = time.time()
        status, text = "ok", ""
        try:
            text = engine(img)[0]
        except Exception as exc:  # noqa: BLE001
            status = f"error: {type(exc).__name__}: {exc}"[:300]
        elapsed = time.time() - start
        rows.append(
            {
                "id": page["id"],
                "transcription": text,
                "seconds": round(elapsed, 3),
                "status": status,
                "technology": page["technology"],
                "script": page["script"],
                "n_chars": page["n_chars"],
            }
        )
        log.info("[%d/%d] %s %s %.1fs (%d chars)", i, len(pages), page["id"], status, elapsed, len(text))

    df = pd.DataFrame(rows)
    df[["id", "transcription"]].to_csv(out_dir / "predictions.csv", index=False, quoting=csv.QUOTE_ALL)
    df.to_csv(out_dir / "details.csv", index=False)
    dump_json(
        out_dir / "run.json",
        {
            "engine": args.engine,
            "model": MODEL_ID if args.engine == "yigdzin" else "tesseract-bod",
            "split": args.split,
            "pages": len(pages),
            "device": device,
            "dtype": dtype,
            "use_dry": args.use_dry,
            "use_cache": not args.no_cache,
            "max_new_tokens": args.max_new_tokens if args.engine == "yigdzin" else None,
            "total_seconds": round(time.time() - t0, 1),
            "mean_seconds_per_page": round(df.seconds.mean(), 2),
            "errors": int((df.status != "ok").sum()),
        },
    )
    log.info("wrote %s (%.1fs total, %.2fs/page)", out_dir, time.time() - t0, df.seconds.mean())


if __name__ == "__main__":
    main()
