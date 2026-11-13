"""Shared paths, normalization and metrics for the Tibetan OCR/ASR evaluation."""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OCR_DIR = DATA / "ocr"
ASR_DIR = DATA / "asr"
OUT_DIR = ROOT / "outputs"
REPORTS = ROOT / "reports"
TESSDATA = ROOT / "tessdata"

for _d in (OCR_DIR, ASR_DIR, OUT_DIR, REPORTS, TESSDATA):
    _d.mkdir(parents=True, exist_ok=True)

_MULTI_TSHEG_RE = re.compile("\u0F0B{2,}")
PLACEHOLDERS = frozenset("KOBIS")


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Normalization: identical to BDRC's official OCR scorer
# (botok -> strip whitespace -> fold tsheg -> drop placeholders).
# --------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    from botok.utils.corpus_normalization import normalize_spaces
    from botok.utils.lenient_normalization import normalize_graphical
    from botok.utils.unicode_normalization import normalize_unicode

    text = text.replace("\u00A0", " ")
    text = normalize_spaces(text, tibetan_specific=False)
    text = normalize_unicode(text)
    text = normalize_graphical(text)
    return text.strip(" \n\r")


def prepare_text(text: str) -> str:
    text = "".join(text.split())
    text = _MULTI_TSHEG_RE.sub("\u0F0B", text)
    return "".join(c for c in text if c not in PLACEHOLDERS)


def cer(hyp_raw: str, ref_prepared: str) -> float:
    """Character error rate, capped at 1.0 (BDRC protocol)."""
    from rapidfuzz.distance import Levenshtein

    hyp = prepare_text(normalize_text(hyp_raw))
    if not ref_prepared:
        return 0.0 if not hyp else 1.0
    return min(Levenshtein.distance(hyp, ref_prepared) / len(ref_prepared), 1.0)


_wt = None


def word_tokenize(text: str) -> list[str]:
    """BoTok word segmentation, punctuation dropped (BoTok-SWER protocol)."""
    global _wt
    if _wt is None:
        from botok import WordTokenizer

        _wt = WordTokenizer()
    tokens = []
    for tok in _wt.tokenize(prepare_text(normalize_text(text))):
        if tok.chunk_type == "TEXT":
            tokens.append(tok.text.strip())
    return [t for t in tokens if t]


def swer(hyp_raw: str, ref_raw: str) -> float:
    """Segmented word error rate: WER over BoTok word tokens."""
    from rapidfuzz.distance import Levenshtein

    ref = word_tokenize(ref_raw)
    hyp = word_tokenize(hyp_raw)
    if not ref:
        return 0.0 if not hyp else 1.0
    return min(Levenshtein.distance(hyp, ref) / len(ref), 1.0)
