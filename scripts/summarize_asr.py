"""Build the final ASR comparison table on a single common evaluation subset.

NICT runs are all scored on the same 300-utterance speaker-stratified sample
(full-set runs are filtered down to it), OpenPecha runs on all 160 clips.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import ASR_DIR, OUT_DIR, REPORTS, get_logger, load_json, prepare_text

log = get_logger("summarize_asr")

FINAL_MODELS = [
    ("wav2vec2-base", "billingsmoore/tibetan-asr-nict-tib1-wav2vec2-base"),
    ("openpecha-general", "openpecha/general_stt_base_model"),
    ("whisper-tiny", "billingsmoore/tibetan-asr-nict-tib1-whisper-tiny"),
    ("whisper-base", "billingsmoore/tibetan-asr-nict-tib1-whisper-base"),
    ("whisper-small", "billingsmoore/tibetan-asr-nict-tib1-whisper-small"),
    ("tenzin-small", "tenzin/whisper-small-tibetan"),
]
SHORT = 150


def latest_run(model: str, dataset: str) -> Path | None:
    runs = []
    for d in (OUT_DIR / "asr").iterdir():
        rj = d / "run.json"
        if not rj.exists():
            continue
        meta = load_json(rj)
        if meta.get("model") != model or meta.get("dataset") != dataset:
            continue
        if not (d / "predictions_scored.csv").exists():
            continue
        stamp = "_".join(d.name.split("_")[-2:])
        runs.append((stamp, d))
    return max(runs)[1] if runs else None


def metrics(run_dir: Path, nict_ids: set[str] | None, alias: str):
    det = pd.read_csv(run_dir / "predictions_scored.csv", dtype={"id": str})
    meta = load_json(run_dir / "run.json")
    if nict_ids is not None:
        det = det[det.id.isin(nict_ids)]
    ref_len = det.reference.fillna("").map(lambda x: len(prepare_text(x)))
    short = det[ref_len <= SHORT]
    return {
        "model": alias,
        "hf_id": meta["model"],
        "dataset": meta["dataset"],
        "n": len(det),
        "cer": round(det.cer.mean(), 4),
        "swer": round(det.swer.mean(), 4),
        "n_short": len(short),
        "cer_short": round(short.cer.mean(), 4),
        "swer_short": round(short.swer.mean(), 4),
        "rtf": meta.get("rtf"),
        "peak_memory_mb": meta.get("peak_memory_mb"),
        "run_id": run_dir.name,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nict", choices=["nict", "openpecha", "both"])
    args = ap.parse_args()

    nict_ids = set(load_json(ASR_DIR / "nict_test" / "sample_ids.json")["sample"])
    rows = []
    datasets = ["nict", "openpecha"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        for alias, repo in FINAL_MODELS:
            run = latest_run(repo, ds)
            if run is None:
                log.warning("missing run: %s / %s", alias, ds)
                continue
            rows.append(metrics(run, nict_ids if ds == "nict" else None, alias))
    df = pd.DataFrame(rows)
    out = REPORTS / f"asr_final_{args.dataset}.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
