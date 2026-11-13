"""Score ASR runs: CER (character) + BoTok-SWER (segmented word) + RTF."""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import pandas as pd

from common import ASR_DIR, OUT_DIR, REPORTS, get_logger, load_json, prepare_text, swer, cer

log = get_logger("score_asr")


def score(run_dir: Path) -> dict:
    det = pd.read_csv(run_dir / "predictions.csv", dtype={"id": str})
    meta = load_json(run_dir / "run.json")

    det["cer"] = [cer(h if isinstance(h, str) else "", prepare_text(r or "")) for h, r in zip(det.transcription, det.reference)]
    det["swer"] = [swer(h if isinstance(h, str) else "", r or "") for h, r in zip(det.transcription, det.reference)]
    det["rtf"] = det.seconds / det.duration.clip(lower=1e-3)
    det.to_csv(run_dir / "predictions_scored.csv", index=False)

    # common 300-utterance NICT subset, so full-set and sampled runs stay comparable
    sub = det
    if meta.get("dataset") == "nict":
        sample_ids = set(load_json(ASR_DIR / "nict_test" / "sample_ids.json")["sample"])
        sub = det[det.id.isin(sample_ids)]

    ref_len = det.reference.fillna("").map(lambda x: len(prepare_text(x)))
    short = det[ref_len <= 150]
    result = {
        "run_id": run_dir.name,
        "model": meta.get("model"),
        "kind": meta.get("kind"),
        "dataset": meta.get("dataset"),
        "n": len(det),
        "errors": int((det.status != "ok").sum()),
        "cer": round(det.cer.mean(), 4),
        "cer_median": round(det.cer.median(), 4),
        "swer": round(det.swer.mean(), 4),
        "n_sub": len(sub),
        "cer_sub": round(sub.cer.mean(), 4) if len(sub) else float("nan"),
        "swer_sub": round(sub.swer.mean(), 4) if len(sub) else float("nan"),
        "n_short": len(short),
        "cer_short": round(short.cer.mean(), 4) if len(short) else float("nan"),
        "swer_short": round(short.swer.mean(), 4) if len(short) else float("nan"),
        "rtf": round(meta.get("rtf", det.seconds.sum() / det.duration.sum()), 3),
        "peak_memory_mb": meta.get("peak_memory_mb"),
        "beams": meta.get("beams"),
        "device": meta.get("device"),
    }
    if "dept" in det and det.dept.notna().any() and det.dept.astype(str).str.len().gt(0).any():
        for dept, grp in det.groupby("dept"):
            result[f"cer:{dept}"] = round(grp.cer.mean(), 4)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", nargs="*")
    args = ap.parse_args()

    run_dirs = (
        [OUT_DIR / "asr" / r for r in args.run_id]
        if args.run_id
        else sorted(p for p in (OUT_DIR / "asr").iterdir() if (p / "predictions.csv").exists())
    )
    results = [score(p) for p in run_dirs]

    out = REPORTS / "asr_results.csv"
    df = pd.DataFrame(results)
    if out.exists():
        old = pd.read_csv(out)
        df = pd.concat([old[~old.run_id.isin(df.run_id)], df], ignore_index=True)
    df.to_csv(out, index=False)
    log.info("wrote %s", out)

    cols = ["run_id", "dataset", "n", "cer", "swer", "rtf", "peak_memory_mb"]
    print(df[cols].sort_values(["dataset", "cer"]).to_string(index=False))
    for r in results:
        extras = {k: v for k, v in r.items() if k.startswith("cer:")}
        if extras:
            print(f"\n{r['run_id']}: " + ", ".join(f"{k[4:]}={v*100:.1f}%" for k, v in sorted(extras.items())))


if __name__ == "__main__":
    main()
