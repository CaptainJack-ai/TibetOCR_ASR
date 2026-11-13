"""Prepare ASR evaluation data.

1) NICT-Tib1 test split (Lhasa Tibetan, 1,547 utts / 3 speakers) - the exact
   test set used by the billingsmoore/* Tibetan ASR checkpoints.
2) OpenPecha / Monlam Tibetan Voice benchmark - 8 domains, sampled per domain
   for cross-domain generalisation checks.
"""
from __future__ import annotations

import argparse
import csv
import io
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
import soundfile as sf
from huggingface_hub import hf_hub_download

from common import ASR_DIR, dump_json, get_logger

log = get_logger("prepare_asr")

NICT_REPO = "billingsmoore/nict-tib1"
NICT_PARQUET = "data/test-00000-of-00001.parquet"
OPENPECHA_CSV = "test.csv"
OPENPECHA_REPO = "openpecha/tibetan-voice-benchmark"
OPENPECHA_PER_DEPT = 20
SR = 16000


def decode_to_wav(raw: bytes, out_path: Path) -> float:
    """Decode arbitrary audio bytes to 16 kHz mono wav; returns duration (s)."""
    import librosa

    y, _ = librosa.load(io.BytesIO(raw), sr=SR, mono=True)
    sf.write(out_path, y, SR, subtype="PCM_16")
    return len(y) / SR


def prepare_nict(sample_size: int, seed: int) -> None:
    out_dir = ASR_DIR / "nict_test"
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = Path(
        hf_hub_download(NICT_REPO, NICT_PARQUET, repo_type="dataset", local_dir=ASR_DIR / "hf")
    )
    log.info("parquet: %s (%.1f MB)", parquet_path, parquet_path.stat().st_size / 1e6)
    df = pd.read_parquet(parquet_path)
    log.info("rows=%d columns=%s", len(df), list(df.columns))

    rows = []
    for i, row in enumerate(df.itertuples(index=False)):
        utt_id = f"nict_{i:05d}"
        wav_path = wav_dir / f"{utt_id}.wav"
        dur = decode_to_wav(row.audio["bytes"], wav_path) if not wav_path.exists() else sf.info(wav_path).duration
        rows.append(
            {
                "id": utt_id,
                "audio_path": str(wav_path.relative_to(ASR_DIR.parent.parent)),
                "transcription": row.transcription,
                "speaker": str(row.speaker_id),
                "duration": round(dur, 3),
                "dataset": "nict",
                "dept": "news",
            }
        )

    meta = pd.DataFrame(rows)
    meta.to_csv(out_dir / "metadata.csv", index=False)

    rng = random.Random(seed)
    by_spk: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_spk[r["speaker"]].append(r["id"])
    sample: list[str] = []
    per_spk = max(1, sample_size // max(1, len(by_spk)))
    for spk in sorted(by_spk):
        pool = sorted(by_spk[spk])
        rng.shuffle(pool)
        sample.extend(pool[:per_spk])

    dump_json(
        out_dir / "sample_ids.json",
        {"sample": sorted(sample), "all": [r["id"] for r in rows]},
    )
    log.info(
        "NICT: %d utts, %.1f min, speakers=%s; sample=%d",
        len(rows),
        meta.duration.sum() / 60,
        dict(Counter(r["speaker"] for r in rows)),
        len(sample),
    )


def _download_one(args) -> dict | None:
    row, wav_dir = args
    wav_path = wav_dir / f"{row['file_name']}.wav"
    if not wav_path.exists():
        try:
            resp = requests.get(row["url"], timeout=60)
            resp.raise_for_status()
            wav_path.write_bytes(resp.content)
        except Exception as exc:  # noqa: BLE001
            log.warning("download failed %s: %s", row["file_name"], exc)
            return None
    return {
        "id": row["file_name"],
        "audio_path": str(wav_path.relative_to(ASR_DIR.parent.parent)),
        "transcription": (row["uni"] or "").strip(),
        "speaker": "",
        "duration": round(sf.info(wav_path).duration, 3),
        "dataset": "openpecha",
        "dept": row["dept"],
        "grade": row["grade"],
    }


def prepare_openpecha(seed: int, per_dept: int = OPENPECHA_PER_DEPT) -> None:
    out_dir = ASR_DIR / "openpecha_bench"
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(
        hf_hub_download(OPENPECHA_REPO, OPENPECHA_CSV, repo_type="dataset", local_dir=ASR_DIR / "hf")
    )
    df = pd.read_csv(csv_path)
    log.info("openpecha rows=%d depts=%s", len(df), dict(Counter(df.dept)))

    rng = random.Random(seed)
    chosen = []
    for dept, grp in df.groupby("dept"):
        grp = grp[grp.grade >= 2] if (grp.grade >= 2).sum() >= per_dept else grp
        idx = list(grp.index)
        rng.shuffle(idx)
        chosen.extend(df.loc[idx[:per_dept]].to_dict("records"))

    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = [r for r in ex.map(_download_one, [(r, wav_dir) for r in chosen]) if r]
    meta = pd.DataFrame(rows)
    meta.to_csv(out_dir / "metadata.csv", index=False)
    dump_json(out_dir / "sample_ids.json", {"all": meta.id.tolist()})
    log.info(
        "OpenPecha: %d clips, %.1f min, per-dept=%s",
        len(meta),
        meta.duration.sum() / 60,
        dict(Counter(meta.dept)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nict-sample", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--openpecha-per-dept", type=int, default=OPENPECHA_PER_DEPT,
                    help="clips per domain (use 2000 to fetch the whole benchmark)")
    ap.add_argument("--skip-nict", action="store_true")
    ap.add_argument("--skip-openpecha", action="store_true")
    args = ap.parse_args()

    if not args.skip_nict:
        prepare_nict(args.nict_sample, args.seed)
    if not args.skip_openpecha:
        prepare_openpecha(args.seed, args.openpecha_per_dept)


if __name__ == "__main__":
    main()
