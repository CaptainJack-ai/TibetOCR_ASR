"""Download the BDRC Tibetan OCR benchmark and export images + ground truth.

Dataset: https://huggingface.co/datasets/BDRC/tibetan-ocr-benchmark (CC0)
472 open pages with hand transcriptions, plus BDRC's official CER scorer.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from common import OCR_DIR, dump_json, get_logger

log = get_logger("prepare_ocr")

REPO = "BDRC/tibetan-ocr-benchmark"
PARQUET = "data/test-00000-of-00001.parquet"
OFFICIAL_FILES = ["runs/compute_cer.py", "runs/ground_truth.csv", "runs/requirements.txt"]

# 60-page stratified sample (quotas follow the benchmark's technology mix).
SAMPLE_QUOTAS = {
    "manuscripts": 18,
    "digital_fonts": 14,
    "blockprints": 14,
    "modern": 5,
    "metal_types": 5,
    "typewriters": 4,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    images_dir = OCR_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = Path(
        hf_hub_download(REPO, PARQUET, repo_type="dataset", local_dir=OCR_DIR / "hf")
    )
    log.info("parquet: %s (%.1f MB)", parquet_path, parquet_path.stat().st_size / 1e6)
    df = pd.read_parquet(parquet_path)
    log.info("rows=%d columns=%s", len(df), list(df.columns))

    gt_rows, meta_rows = [], []
    for row in df.itertuples(index=False):
        page_id = str(row.id).zfill(4)
        img_path = images_dir / f"{page_id}.png"
        if args.force or not img_path.exists():
            img_path.write_bytes(row.image["bytes"])
        gt_rows.append({"id": page_id, "transcription": row.transcription})
        meta_rows.append(
            {
                "id": page_id,
                "technology": row.technology,
                "script": row.script,
                "script_4": row.script_4,
                "period": row.period,
                "rarity": row.rarity,
                "legibility": row.legibility,
                "n_chars": len(row.transcription or ""),
            }
        )

    with (OCR_DIR / "ground_truth.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "transcription"])
        w.writeheader()
        w.writerows(gt_rows)
    pd.DataFrame(meta_rows).to_csv(OCR_DIR / "metadata.csv", index=False)

    official_dir = OCR_DIR / "official"
    official_dir.mkdir(parents=True, exist_ok=True)
    for f in OFFICIAL_FILES:
        p = hf_hub_download(REPO, f, repo_type="dataset", local_dir=OCR_DIR / "hf")
        (official_dir / Path(f).name).write_bytes(Path(p).read_bytes())
    log.info("official scorer -> %s", official_dir)

    tech = Counter(m["technology"] for m in meta_rows)
    script = Counter(m["script"] for m in meta_rows)
    dump_json(OCR_DIR / "stats.json", {"n_pages": len(meta_rows), "technology": tech, "script": script})

    by_tech: dict[str, list[dict]] = defaultdict(list)
    for m in meta_rows:
        by_tech[m["technology"]].append(m)

    rng = random.Random(args.seed)

    # pilot: one page per technology, favour short/legible pages for a fast smoke test
    pilot = []
    for t in SAMPLE_QUOTAS:
        pool = [m for m in by_tech[t] if not m["legibility"].startswith("1")]
        pool.sort(key=lambda m: (m["n_chars"], m["id"]))
        pool = pool[: max(1, len(pool) // 3)]
        if pool:
            pilot.append(rng.choice(pool)["id"])

    # main sample: quotas per technology, balanced across Uchen/Ume inside each bucket
    sample = []
    for t, quota in SAMPLE_QUOTAS.items():
        pool = sorted(by_tech[t], key=lambda m: m["id"])
        by_script: dict[str, list[dict]] = defaultdict(list)
        for m in pool:
            by_script[m["script"]].append(m)
        picked: list[str] = []
        scripts = sorted(by_script)
        i = 0
        while len(picked) < quota and any(by_script[s] for s in scripts):
            s = scripts[i % len(scripts)]
            if by_script[s]:
                picked.append(by_script[s].pop(rng.randrange(len(by_script[s])))["id"])
            i += 1
        sample.extend(picked)

    dump_json(OCR_DIR / "sample_ids.json", {"pilot": pilot, "sample": sorted(sample)})
    log.info("pilot pages: %s", pilot)
    log.info("sample: %d pages (%s)", len(sample), dict(Counter(m["technology"] for m in meta_rows if m["id"] in sample)))
    log.info("technology distribution: %s", dict(tech))
    log.info("script distribution: %s", dict(script))


if __name__ == "__main__":
    main()
