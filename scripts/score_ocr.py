"""Score OCR runs with the BDRC protocol (botok normalization + capped CER)."""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import pandas as pd

from common import OCR_DIR, OUT_DIR, REPORTS, cer, get_logger, prepare_text

log = get_logger("score_ocr")
CAT_THRESHOLD = 0.5


def score(run_dir: Path) -> dict:
    gt = pd.read_csv(OCR_DIR / "ground_truth.csv", dtype={"id": str}).set_index("id")
    gt_prepared = {i: prepare_text(t or "") for i, t in gt.transcription.items()}

    det = pd.read_csv(run_dir / "details.csv", dtype={"id": str})
    det["gt"] = det.id.map(gt.transcription)
    det["cer"] = [cer(h if isinstance(h, str) else "", gt_prepared.get(i, "")) for i, h in zip(det.id, det.transcription)]
    det["catastrophic"] = det.cer > CAT_THRESHOLD
    det.to_csv(run_dir / "details_scored.csv", index=False)

    per_page = det.cer.tolist()
    result = {
        "run_id": run_dir.name,
        "pages": len(det),
        "coverage": int((det.status == "ok").sum()),
        "mean_cer": round(statistics.mean(per_page), 4),
        "median_cer": round(statistics.median(per_page), 4),
        "catastrophic_pct": round(100 * det.catastrophic.mean(), 2),
        "mean_seconds_per_page": round(det.seconds.mean(), 2),
    }
    for key in ("technology", "script"):
        for val, grp in det.groupby(key):
            result[f"{key}:{val}"] = round(grp.cer.mean(), 4)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", nargs="*", default=None)
    args = ap.parse_args()

    run_dirs = (
        [OUT_DIR / "ocr" / r for r in args.run_id]
        if args.run_id
        else sorted(p for p in (OUT_DIR / "ocr").iterdir() if (p / "details.csv").exists())
    )
    results = [score(p) for p in run_dirs]

    out = REPORTS / "ocr_results.csv"
    df = pd.DataFrame(results)
    if out.exists() and not args.run_id:
        old = pd.read_csv(out)
        df = pd.concat([old[~old.run_id.isin(df.run_id)], df], ignore_index=True)
    df.to_csv(out, index=False)
    log.info("wrote %s", out)

    cols = ["run_id", "pages", "mean_cer", "median_cer", "catastrophic_pct", "mean_seconds_per_page"]
    print(df[cols].to_string(index=False))
    for r in results:
        extras = {k: v for k, v in r.items() if ":" in k}
        print(f"\n{r['run_id']}: " + ", ".join(f"{k}={v*100:.1f}%" for k, v in sorted(extras.items())))


if __name__ == "__main__":
    main()
