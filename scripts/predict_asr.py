"""Transcribe one or more audio files with a Tibetan ASR checkpoint.

Thin CLI on top of the same `Transcriber` used by the evaluation pipeline, plus
silence-aware chunking so long recordings do not hit Whisper's 448-token cap
(≈145 Tibetan characters per call).

Examples:
  python scripts/predict_asr.py data/asr/nict_test/wav/nict_00045.wav --model whisper-small
  python scripts/predict_asr.py mytalk.m4a --model wav2vec2-base --json
  python scripts/predict_asr.py long.wav --model whisper-small --chunk-seconds 28
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
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_asr import MODELS, Transcriber, resolve_model  # noqa: E402

SR = 16000


def split_on_silence(audio: np.ndarray, max_seconds: float, top_db: int = 32) -> list[tuple[float, float]]:
    """Split audio into <=max_seconds spans, cutting at silence where possible."""
    intervals = librosa.effects.split(audio, top_db=top_db, frame_length=1024, hop_length=256)
    if len(intervals) == 0:
        return [(0.0, min(len(audio) / SR, max_seconds))]

    spans: list[list[float]] = []
    cur_start, cur_end = intervals[0][0] / SR, intervals[0][1] / SR
    for s, e in intervals[1:]:
        s, e = s / SR, e / SR
        if e - cur_start <= max_seconds:
            cur_end = e
            continue
        spans.append([cur_start, cur_end])
        cur_start, cur_end = s, e
        # a single long non-silent stretch: hard-cut it
        while cur_end - cur_start > max_seconds:
            spans.append([cur_start, cur_start + max_seconds])
            cur_start += max_seconds
    spans.append([cur_start, cur_end])
    return [(a, min(b, len(audio) / SR)) for a, b in spans if b - a > 0.05]


def transcribe_file(pipe: Transcriber, path: Path, chunk_seconds: float) -> tuple[str, dict]:
    audio, _ = librosa.load(path, sr=SR, mono=True)
    duration = len(audio) / SR
    t0 = time.time()

    if chunk_seconds and duration > chunk_seconds:
        spans = split_on_silence(audio, chunk_seconds)
        parts = []
        for a, b in spans:
            seg = audio[int(a * SR): int(b * SR)]
            parts.append(pipe.decode_array(seg))
        text = " ".join(p.strip() for p in parts if p.strip())
    else:
        spans = [(0.0, duration)]
        text = pipe.decode_array(audio)

    return text, {
        "duration": round(duration, 2),
        "seconds": round(time.time() - t0, 2),
        "rtf": round((time.time() - t0) / max(duration, 1e-9), 3),
        "chunks": len(spans),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="+", help="audio file(s), any format ffmpeg/librosa can read")
    ap.add_argument("--model", default="whisper-small", help="alias or full HF repo id")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "fp32", "bf16"])
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=440)
    ap.add_argument("--chunk-seconds", type=float, default=None,
                    help="silence-aware chunk length; 0 disables. "
                         "Default: 9 s for Whisper (448-token cap ≈ 145 chars ≈ 9 s), "
                         "0 (off) for CTC models")
    ap.add_argument("--offline", action="store_true", help="skip HuggingFace Hub metadata calls (models already cached)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    device = args.device if args.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = args.dtype if args.dtype != "auto" else ("fp16" if device == "mps" else "fp32")

    model_id = resolve_model(args.model)
    pipe = Transcriber(model_id, device, dtype, args.beams, None, args.max_new_tokens)
    chunk = args.chunk_seconds
    if chunk is None:
        chunk = 9.0 if pipe.kind == "whisper" else 0.0

    results = []
    for a in args.audio:
        path = Path(a)
        if not path.exists():
            print(f"skip (missing): {path}", file=sys.stderr)
            continue
        text, meta = transcribe_file(pipe, path, chunk)
        results.append({"file": str(path), "text": text, **meta})
        if not args.json:
            print(f"\n=== {path} ({meta['duration']}s, {meta['chunks']} chunk(s) of {chunk or '∞'}s, RTF {meta['rtf']}) ===")
            print(text)

    if args.json:
        print(json.dumps({"model": model_id, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
