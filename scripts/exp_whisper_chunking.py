"""Measure how silence-aware chunking affects Whisper Tibetan ASR accuracy.

Whisper's decoder tops out at 448 tokens ≈ 145 Tibetan characters, while NICT-Tib1
read speech runs at ~16 characters/second — so a 20 s clip needs ~320 characters
and WILL be truncated. This experiment sweeps chunk lengths on the longest test
utterances to find the practical setting.

  python scripts/exp_whisper_chunking.py --n 20 --chunks 0 9 12 16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ASR_DIR, REPORTS, cer, get_logger, prepare_text, swer  # noqa: E402
from predict_asr import SR, split_on_silence  # noqa: E402
from run_asr import Transcriber, resolve_model  # noqa: E402

log = get_logger("exp_chunking")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="whisper-small")
    ap.add_argument("--n", type=int, default=20, help="longest utterances to evaluate")
    ap.add_argument("--chunks", type=float, nargs="+", default=[0, 9, 12, 16])
    ap.add_argument("--min-ref-chars", type=int, default=150)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(REPORTS / "whisper_chunking.csv"))
    args = ap.parse_args()

    meta = pd.read_csv(ASR_DIR / "nict_test" / "metadata.csv", dtype={"id": str})
    meta["n_chars"] = meta.transcription.str.len()
    meta = meta[meta.n_chars >= args.min_ref_chars].nlargest(args.n, "n_chars")
    log.info("%d utterances, ref %.0f-%.0f chars, %.1f s audio",
             len(meta), meta.n_chars.min(), meta.n_chars.max(), meta.duration.sum())

    device = args.device if args.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = "fp16" if device == "mps" else "fp32"
    pipe = Transcriber(resolve_model(args.model), device, dtype, 1, None, 440)

    rows = []
    for chunk in args.chunks:
        cers, swers = [], []
        for r in meta.itertuples():
            audio, _ = librosa.load(ASR_DIR.parent.parent / r.audio_path, sr=SR, mono=True)
            if chunk and len(audio) / SR > chunk:
                parts = [pipe.decode_array(audio[int(a * SR): int(b * SR)])
                         for a, b in split_on_silence(audio, chunk)]
                hyp = " ".join(p.strip() for p in parts if p.strip())
            else:
                hyp = pipe.decode_array(audio)
            cers.append(cer(hyp, prepare_text(r.transcription)))
            swers.append(swer(hyp, r.transcription))
        row = {
            "model": args.model,
            "chunk_seconds": chunk or None,
            "n": len(meta),
            "cer": round(sum(cers) / len(cers), 4),
            "swer": round(sum(swers) / len(swers), 4),
            "median_cer": round(pd.Series(cers).median(), 4),
        }
        rows.append(row)
        log.info("chunk=%-4s CER=%.2f%% SWER=%.2f%%", chunk or "-", row["cer"] * 100, row["swer"] * 100)

    df = pd.DataFrame(rows)
    out = Path(args.out)
    if out.exists():
        old = pd.read_csv(out)
        key = lambda d: (d.model, d.n, d.chunk_seconds.fillna(0))
        keep = ~pd.MultiIndex.from_arrays(key(old)).isin(pd.MultiIndex.from_arrays(key(df)))
        df = pd.concat([old[keep], df], ignore_index=True)
    df = df.sort_values(["model", "n", "chunk_seconds"], na_position="first")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    log.info("wrote %s", out)

    jpath = REPORTS / "whisper_chunking.json"
    merged = json.loads(jpath.read_text(encoding="utf-8")) if jpath.exists() else {}
    for r in rows:
        merged[f"{r['model']}|n{r['n']}|chunk{r['chunk_seconds'] or '-'}"] = r
    jpath.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
