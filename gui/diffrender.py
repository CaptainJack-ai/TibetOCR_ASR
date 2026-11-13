"""Character alignment + coloured HTML for OCR/ASR error inspection.

Single source of truth for the four diff colours used by the GUI:
  correct  - plain ink
  replace  - red background (ref char differs)
  delete   - grey + strike-through (model missed it)
  insert   - amber + dashed underline (model added it)
"""
from __future__ import annotations

import html
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import cer, normalize_text, prepare_text, swer, word_tokenize  # noqa: E402

SPACE_GLYPH = "\u2423"   # ␣
NEWLINE_GLYPH = "\u00b6"  # ¶


def _vis(ch: str) -> str:
    if ch == "\n":
        return f'<span class="ws">{NEWLINE_GLYPH}</span>'
    if ch in " \t":
        return f'<span class="ws">{SPACE_GLYPH}</span>'
    return html.escape(ch)


@dataclass
class DiffStats:
    match: int = 0
    replace: int = 0
    insert: int = 0
    delete: int = 0

    @property
    def total_ref(self) -> int:
        return self.match + self.replace + self.delete

    @property
    def char_accuracy(self) -> float:
        return self.match / self.total_ref if self.total_ref else 0.0

    def as_dict(self) -> dict:
        return {
            "match": self.match,
            "replace": self.replace,
            "insert": self.insert,
            "delete": self.delete,
            "ref_chars": self.total_ref,
            "char_accuracy": round(self.char_accuracy, 4),
        }


def align(ref: str, hyp: str) -> tuple[list[tuple[str, str, str]], DiffStats]:
    """Return [(kind, ref_text, hyp_text)] plus counts. kind in match/replace/delete/insert."""
    sm = SequenceMatcher(a=ref, b=hyp, autojunk=False)
    ops: list[tuple[str, str, str]] = []
    st = DiffStats()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        r, h = ref[i1:i2], hyp[j1:j2]
        if tag == "equal":
            ops.append(("match", r, h))
            st.match += len(r)
        elif tag == "replace":
            ops.append(("replace", r, h))
            st.replace += max(len(r), len(h))
        elif tag == "delete":
            ops.append(("delete", r, h))
            st.delete += len(r)
        else:  # insert
            ops.append(("insert", r, h))
            st.insert += len(h)
    return ops, st


def render(ops: list[tuple[str, str, str]], mode: str = "inline") -> str:
    """mode='inline' colours the hypothesis, with the reference shown on hover/strike."""
    out: list[str] = []
    for kind, r, h in ops:
        if kind == "match":
            out.append("".join(_vis(c) for c in h))
        elif kind == "replace":
            ref_txt = html.escape(r).replace("\n", " ") or "-"
            out.append(
                f'<span class="d-rep" title="参考：{ref_txt}">'
                + "".join(_vis(c) for c in h)
                + "</span>"
            )
        elif kind == "delete":
            out.append(f'<span class="d-del" title="漏识别">{ "".join(_vis(c) for c in r) }</span>')
        else:
            out.append(f'<span class="d-ins" title="多识别">{ "".join(_vis(c) for c in h) }</span>')
    return "".join(out)


def side_by_side(ops: list[tuple[str, str, str]]) -> tuple[str, str]:
    """Two aligned columns: reference (left) and hypothesis (right)."""
    left, right = [], []
    for kind, r, h in ops:
        if kind == "match":
            left.append("".join(_vis(c) for c in r))
            right.append("".join(_vis(c) for c in h))
        elif kind == "replace":
            left.append(f'<span class="d-del">{ "".join(_vis(c) for c in r) }</span>')
            right.append(f'<span class="d-rep">{ "".join(_vis(c) for c in h) }</span>')
        elif kind == "delete":
            left.append(f'<span class="d-del">{ "".join(_vis(c) for c in r) }</span>')
            right.append(f'<span class="ws">{"∅" * 0}</span>')
        else:
            left.append("")
            right.append(f'<span class="d-ins">{ "".join(_vis(c) for c in h) }</span>')
    return "".join(left), "".join(right)


def analyse(ref_raw: str, hyp_raw: str, kind: str = "ocr", normalize_reference: bool = True) -> dict:
    """Full result bundle for one sample.

    kind='ocr'  -> character diff + CER
    kind='asr'  -> character diff + CER + BoTok word diff + SWER
    """
    ref_prepared = prepare_text(normalize_text(ref_raw or "") if normalize_reference else ref_raw or "")
    ops, stats = align(ref_prepared, prepare_text(normalize_text(hyp_raw or "")))
    left, right = side_by_side(ops)
    result = {
        "kind": kind,
        "cer": round(cer(hyp_raw or "", ref_prepared), 4),
        "metrics": stats.as_dict(),
        "diff_inline": render(ops, "inline"),
        "diff_ref": left,
        "diff_hyp": right,
        "ref_len": len(ref_prepared),
        "hyp_len": len(prepare_text(normalize_text(hyp_raw or ""))),
        "ref_text": ref_prepared,
        "hyp_text": hyp_raw or "",
    }
    if kind == "asr":
        ref_words = word_tokenize(ref_raw or "")
        hyp_words = word_tokenize(hyp_raw or "")
        wops, wstats = align(ref_words, hyp_words)
        result["swer"] = round(swer(hyp_raw or "", ref_raw or ""), 4)
        result["word_metrics"] = wstats.as_dict()
        cls = {"match": None, "replace": "d-rep", "delete": "d-del", "insert": "d-ins"}
        chunks = []
        for kind_w, r_words, h_words in wops:
            words = h_words if kind_w in ("match", "replace", "insert") else r_words
            for w in words:
                if cls[kind_w]:
                    chunks.append(f'<span class="{cls[kind_w]}">{html.escape(w)}</span>')
                else:
                    chunks.append(html.escape(w))
        result["word_diff"] = " ".join(chunks)
    return result


if __name__ == "__main__":  # tiny self-check
    ref = "བཀྲ་ཤིས་བདེ་ལེགས། ང་ཚོས་བོད་ཡིག་སློབ་སྦྱོང་བྱེད་ཀྱི་ཡོད།"
    hyp = "བཀྲ་ཤིས་བདེ་ལེག ང་ཚོས་བོད་ཡིག་སློབ་སྦྱོང་བྱེད་ཀྱི་ཡོད།"
    r = analyse(ref, hyp, "asr")
    print("cer", r["cer"], "swer", r["swer"], "stats", r["metrics"])
    assert abs(r["cer"] - 0.0364) < 0.002, r["cer"]
    assert abs(r["swer"] - 0.2222) < 0.002, r["swer"]
    assert "d-del" in r["diff_inline"]
    print("self-check ok")
