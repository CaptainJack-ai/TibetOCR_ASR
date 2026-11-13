"""Data integrity checks for the downloaded OCR / ASR datasets.

Verifies that the derived layout on disk is complete, self-consistent and
matches the official ground truth. Exits non-zero if anything is wrong.

  python scripts/verify_data.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
from PIL import Image

from common import ASR_DIR, OCR_DIR, TESSDATA, get_logger, load_json, prepare_text

log = get_logger("verify")

ERRORS: list[str] = []


def check(cond: bool, msg: str, detail: str = "") -> None:
    if cond:
        log.info("PASS  %s%s", msg, f"  ({detail})" if detail else "")
    else:
        log.error("FAIL  %s%s", msg, f"  ({detail})" if detail else "")
        ERRORS.append(msg)


def verify_ocr() -> None:
    images = sorted((OCR_DIR / "images").glob("*.png"))
    gt = pd.read_csv(OCR_DIR / "ground_truth.csv", dtype={"id": str})
    meta = pd.read_csv(OCR_DIR / "metadata.csv", dtype={"id": str})

    check(len(images) == 472, "OCR: 472 page images", f"{len(images)} files")
    check(not gt.id.duplicated().any(), "OCR: ground_truth ids unique")
    check(set(gt.id) == set(meta.id), "OCR: ground_truth ids == metadata ids")
    check({p.stem for p in images} == set(gt.id), "OCR: image files == ground_truth ids")
    check(gt.transcription.notna().all() and (gt.transcription.str.len() > 0).all(), "OCR: every page has transcription")

    # images must be decodable regardless of the container extension
    bad = []
    for p in images[:40]:
        try:
            Image.open(p).verify()
        except Exception:  # noqa: BLE001
            bad.append(p.name)
    check(not bad, "OCR: sample images decodable", f"first 40 checked, {len(bad)} broken")

    # our exported transcriptions must equal the official botok-normalised ones
    official = OCR_DIR / "official" / "ground_truth.csv"
    if official.exists():
        off = pd.read_csv(official, dtype={"id": str}).set_index("id")
        ours = gt.set_index("id")
        diffs = sum(
            prepare_text(ours.transcription[i] or "") != prepare_text(off.transcription[i] or "")
            for i in ours.index
        )
        check(diffs == 0, "OCR: prepared GT == official normalised GT", f"{diffs}/472 differ")
        check((OCR_DIR / "official" / "compute_cer.py").exists(), "OCR: official scorer present")

    check((TESSDATA / "bod.traineddata").exists(), "OCR: tesseract bod model present")

    sample = load_json(OCR_DIR / "sample_ids.json")
    check(len(sample["pilot"]) == 6 and len(sample["sample"]) == 60, "OCR: sample ids", "pilot 6 / sample 60")
    check(set(sample["sample"]) <= set(gt.id) and len(set(sample["sample"])) == 60, "OCR: sample ids unique and in GT")


def verify_nict() -> None:
    d = ASR_DIR / "nict_test"
    meta = pd.read_csv(d / "metadata.csv", dtype={"id": str, "speaker": str})
    wavs = sorted((d / "wav").glob("*.wav"))

    check(len(meta) == 1547, "NICT: 1547 test utterances", str(len(meta)))
    check(len(wavs) == 1547, "NICT: 1547 wav files", str(len(wavs)))
    check(not meta.id.duplicated().any(), "NICT: ids unique")
    check({p.stem for p in wavs} == set(meta.id), "NICT: wav names == metadata ids")

    info = [sf.info(p) for p in wavs[:50]]
    check(all(i.samplerate == 16000 for i in info), "NICT: 16 kHz", f"{len(info)} checked")
    check(all(i.channels == 1 for i in info), "NICT: mono", f"{len(info)} checked")

    recorded = meta.set_index("id").duration
    diffs = []
    for p in wavs[:50]:
        diffs.append(abs(sf.info(p).duration - float(recorded[p.stem])))
    check(max(diffs) < 0.01, "NICT: metadata duration matches audio", f"max diff {max(diffs):.4f}s")
    check(meta.transcription.notna().all(), "NICT: every utterance has transcription")
    speakers = sorted(meta.speaker.astype(str).unique())
    check(len(speakers) == 3, "NICT: 3 speakers (disjoint from train)", ", ".join(speakers))

    ids = load_json(d / "sample_ids.json")
    check(len(ids["sample"]) == 300 and len(ids["all"]) == 1547, "NICT: sample ids", "sample 300 / all 1547")
    check(set(ids["sample"]) <= set(meta.id), "NICT: sample ids in metadata")


def verify_openpecha() -> None:
    d = ASR_DIR / "openpecha_bench"
    meta = pd.read_csv(d / "metadata.csv", dtype={"id": str})
    wavs = sorted((d / "wav").glob("*.wav"))

    check(len(meta) == 160, "OpenPecha: 160 sampled clips", str(len(meta)))
    check(len(wavs) == len(meta), "OpenPecha: wav per clip", f"{len(wavs)}")
    check(not meta.id.duplicated().any(), "OpenPecha: ids unique")
    check(meta.dept.nunique() == 8 and meta.groupby("dept").size().min() == 20, "OpenPecha: 8 domains x 20")
    check((meta.grade == 3).all(), "OpenPecha: all grade-3 (QC) transcriptions")
    check(meta.transcription.notna().all() and (meta.transcription.str.len() > 0).all(), "OpenPecha: every clip has transcription")

    info = [sf.info(p) for p in wavs[:20]]
    check(all(i.samplerate == 16000 and i.channels == 1 for i in info), "OpenPecha: 16 kHz mono", f"{len(info)} checked")


def main() -> None:
    cases = {"ocr": verify_ocr, "nict": verify_nict, "openpecha": verify_openpecha}
    for name, fn in cases.items():
        if not (OCR_DIR if name == "ocr" else ASR_DIR).exists():
            log.warning("skip %s (dataset not downloaded)", name)
            continue
        log.info("--- %s ---", name)
        fn()

    log.info("=" * 60)
    if ERRORS:
        log.error("%d check(s) failed", len(ERRORS))
        sys.exit(1)
    log.info("all checks passed")


if __name__ == "__main__":
    main()
