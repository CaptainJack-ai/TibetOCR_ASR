"""Export a handful of qualitative samples (image/audio + reference + prediction)."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from common import OCR_DIR, OUT_DIR, REPORTS, get_logger, load_json


def det_id_to_dataset(run_id: str) -> str:
    return load_json(OUT_DIR / "asr" / run_id / "run.json")["dataset"] == "nict" and "nict_test" or "openpecha_bench"

log = get_logger("make_samples")
OUT = REPORTS / "samples"


def ocr_samples(run_id: str) -> None:
    run_dir = OUT_DIR / "ocr" / run_id
    det = pd.read_csv(run_dir / "details_scored.csv", dtype={"id": str})
    picks = []
    for tech in det.technology.unique():
        sub = det[det.technology == tech].sort_values("cer")
        picks.append(sub.iloc[0])
    picks = picks[:6]

    lines = [f"# OCR 样例（{run_id}）\n", "| 页面 | 技术 | CER | 参考（截断） | 预测（截断） |", "|---|---|---:|---|---|"]
    for r in picks:
        img = OCR_DIR / "images" / f"{r['id']}.png"
        dst = OUT / "ocr" / f"{r['id']}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(img, dst)
        ref = " ".join(str(r["gt"]).split()).replace("|", "/")[:140]
        hyp = " ".join(str(r["transcription"]).split()).replace("|", "/")[:140]
        lines.append(f"| [{r['id']}](ocr/{r['id']}.png) | {r['technology']} | {r['cer']*100:.1f}% | {ref} | {hyp} |")
    (OUT / "ocr_samples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s", OUT / "ocr_samples.md")


def asr_samples(run_id: str) -> None:
    run_dir = OUT_DIR / "asr" / run_id
    det = pd.read_csv(run_dir / "predictions_scored.csv", dtype={"id": str})
    meta = pd.read_csv(OUT_DIR.parent / "data" / "asr" / det_id_to_dataset(run_id) / "metadata.csv", dtype={"id": str})
    det = det.merge(meta[["id", "audio_path"]], on="id", how="left")
    det = det.sort_values("duration")
    picks = pd.concat([det.head(2), det.tail(2)])

    lines = [f"# ASR 样例（{run_id}）\n", "| 音频 | 时长 | CER | 参考 | 预测 |", "|---|---:|---:|---|---|"]
    for r in picks.itertuples():
        src = Path(r.audio_path) if Path(r.audio_path).is_absolute() else OCR_DIR.parent.parent / r.audio_path
        dst = OUT / "asr" / f"{r.id}.wav"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy(src, dst)
        ref = " ".join(str(r.reference).split()).replace("|", "/")[:140]
        hyp = " ".join(str(r.transcription).split()).replace("|", "/")[:140]
        lines.append(f"| [{r.id}](asr/{r.id}.wav) | {r.duration:.1f}s | {r.cer*100:.1f}% | {ref} | {hyp} |")
    (OUT / "asr_samples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s", OUT / "asr_samples.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-run")
    ap.add_argument("--asr-run")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.ocr_run:
        ocr_samples(args.ocr_run)
    if args.asr_run:
        asr_samples(args.asr_run)


if __name__ == "__main__":
    main()
