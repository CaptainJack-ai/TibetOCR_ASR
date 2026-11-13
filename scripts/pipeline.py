"""End-to-end batch pipeline: real folders/PDFs/audio/video in, text out.

Two subcommands, both resumable (re-running skips finished items) and crash safe
(one JSONL line per item, flushed immediately).

  python scripts/pipeline.py ocr  book.pdf notes/ --out outputs/myocr
  python scripts/pipeline.py asr  talk.mp4 rec/ --out outputs/myasr --model whisper-small

Outputs per run directory:
  pages.jsonl / items.jsonl   one record per page/clip (append-only)
  pages.csv   / items.csv     same as a table
  transcripts/                one .txt per input (+ .srt when chunked)
  report.md                   counts, throughput, flagged items
"""
from __future__ import annotations

import os
import sys

# Must happen before transformers is imported: skip all Hub metadata calls.
if "--offline" in sys.argv:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_asr import SR, split_on_silence  # noqa: E402
from run_asr import Transcriber, resolve_model  # noqa: E402
from run_ocr import TesseractEngine, YigdzinEngine, pick_device  # noqa: E402

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
MEDIA_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma",
             ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".flv"}
LOOP_NGRAM = 12
# Degenerate loops repeat a 12-gram dozens of times; scripture repeats 3-5 times.
LOOP_RETRY_AT = 8
LOOP_FLAG = 6


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def loop_score(text: str, n: int = LOOP_NGRAM) -> int:
    from collections import Counter

    if len(text) < n:
        return 1
    grams = (text[i: i + n] for i in range(len(text) - n + 1))
    return max(Counter(grams).values())


def derive_flags(chars: int, loops: int) -> list[str]:
    flags: list[str] = []
    if chars == 0:
        flags.append("empty")
    elif chars < 10:
        flags.append("too_short")
    if loops >= LOOP_FLAG:
        flags.append("loop")
    return flags


def existing_ids(jsonl: Path) -> set[str]:
    ids: set[str] = set()
    if jsonl.exists():
        with jsonl.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("status") == "ok":
                    ids.add(rec["id"])
    return ids


def append_jsonl(path: Path, rec: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()


# --------------------------------------------------------------------------- OCR
def collect_pages(inputs: list[str], out: Path, dpi: int) -> list[dict]:
    pages: list[dict] = []
    page_dir = out / "_pages"
    for raw in inputs:
        p = Path(raw)
        if not p.exists():
            log(f"skip (missing): {p}")
            continue
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.suffix.lower() in IMAGE_EXT:
                    pages.append({"id": f"{f.parent.name}_{f.stem}", "source": str(f), "image": f})
        elif p.suffix.lower() == ".pdf":
            if shutil.which("pdftoppm") is None:
                raise SystemExit("need poppler's pdftoppm to read PDFs (brew install poppler)")
            dest = page_dir / p.stem
            dest.mkdir(parents=True, exist_ok=True)
            if not any(dest.glob("*.png")):
                subprocess.run(["pdftoppm", "-r", str(dpi), "-png", str(p), str(dest / "page")], check=True)
            for img in sorted(dest.glob("*.png")):
                num = re.sub(r"\D", "", img.stem.split("-")[-1]) or img.stem
                pages.append({"id": f"{p.stem}_{num.zfill(4)}", "source": f"{p}#{num}", "image": img})
        elif p.suffix.lower() in IMAGE_EXT:
            pages.append({"id": p.stem, "source": str(p), "image": p})
        else:
            log(f"skip (unsupported): {p}")
    return pages


def cmd_ocr(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "pages.jsonl"

    pages = collect_pages(args.inputs, out, args.dpi)
    if args.limit:
        pages = pages[: args.limit]
    done = existing_ids(jsonl)
    todo = [p for p in pages if p["id"] not in done]
    log(f"ocr: {len(pages)} page(s), {len(done)} already done, {len(todo)} to run")

    device = pick_device(args.device)
    dtype = args.dtype if args.dtype != "auto" else ("bf16" if device == "mps" else "fp32")
    if args.engine == "yigdzin":
        engine = YigdzinEngine(device, dtype, args.max_new_tokens, use_dry=not args.no_dry)
    else:
        engine = TesseractEngine()

    t_start = time.time()
    for i, page in enumerate(todo, 1):
        t0 = time.time()
        rec = {"id": page["id"], "source": page["source"]}
        try:
            text = engine(page["image"], temperature=0.0)[0]
            loops = loop_score(text)
            retried = False
            if loops >= LOOP_RETRY_AT and args.retry_loops and args.engine == "yigdzin":
                samples = engine(page["image"], temperature=0.4, num_samples=3)
                text = min(samples, key=lambda t: (loop_score(t), -len(t)))
                loops = loop_score(text)
                retried = True
            flags = []
            if not text.strip():
                flags.append("empty")
            elif len(text) < 10:
                flags.append("too_short")
            if loops >= LOOP_FLAG:
                flags.append("loop")
            rec.update(text=text, chars=len(text), loop_score=loops, retried=retried,
                       seconds=round(time.time() - t0, 2), status="ok", flags=flags)
        except Exception as exc:  # noqa: BLE001
            rec.update(text="", chars=0, loop_score=0, retried=False,
                       seconds=round(time.time() - t0, 2), status=f"error: {type(exc).__name__}: {exc}"[:300], flags=["error"])
        append_jsonl(jsonl, rec)
        log(f"[{i}/{len(todo)}] {page['id']} {rec['status']} {rec['chars']} chars "
            f"{rec['seconds']}s {'/'.join(rec['flags']) or 'ok'}")

    write_ocr_report(out, engine_name=args.engine, elapsed=time.time() - t_start)


def write_ocr_report(out: Path, engine_name: str, elapsed: float) -> None:
    df = pd.read_json(out / "pages.jsonl", lines=True, dtype={"id": str, "source": str})
    if df.empty:
        log("nothing to report")
        return
    df["flags"] = [derive_flags(int(c), int(l)) for c, l in zip(df.chars.fillna(0), df.loop_score.fillna(0))]
    ok = df[df.status == "ok"]
    (out / "pages.csv").unlink(missing_ok=True)
    df.to_csv(out / "pages.csv", index=False)
    tdir = out / "transcripts"
    tdir.mkdir(exist_ok=True)
    for src, grp in ok.groupby("source"):
        stem = re.sub(r"[^\w\-]+", "_", Path(str(src).split("#")[0]).stem) or "doc"
        suffix = str(src).split("#")[-1] if "#" in str(src) else ""
        (tdir / f"{stem}{'_' + suffix if suffix else ''}.txt").write_text(
            "\n\n".join(grp.sort_values("id").text), encoding="utf-8")

    flagged = df[df["flags"].map(lambda f: bool(f) and f != [])]
    lines = [
        "# OCR 批处理报告", "",
        f"- 引擎：`{engine_name}`",
        f"- 页面：{len(df)}（成功 {len(ok)}，失败 {len(df) - len(ok)}）",
        f"- 总字符：{int(ok.chars.sum()):,}",
        (f"- 本次运行：{elapsed/60:.1f} 分钟" + (f"（推理合计 {ok.seconds.sum()/60:.1f} 分钟）" if len(ok) else ""))
        if elapsed > 1 else "- 本次运行：无新页面（全部命中缓存）",
        f"- 平均每页：{ok.seconds.mean():.1f}s，吞吐 {len(ok)/(ok.seconds.sum()/3600):.0f} 页/小时" if len(ok) and ok.seconds.sum() > 0 else "",
        "", f"## 需要人工复核（{len(flagged)}）", "",
        "| id | 来源 | 字符数 | loop_score | 标记 |", "|---|---|---:|---:|---|",
    ]
    for r in flagged.itertuples():
        lines.append(f"| {r.id} | {str(r.source)[:60]} | {r.chars} | {r.loop_score} | {','.join(r.flags)} |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"report -> {out/'report.md'}")


# --------------------------------------------------------------------------- ASR
def collect_media(inputs: list[str], out: Path) -> list[dict]:
    items: list[dict] = []
    for raw in inputs:
        p = Path(raw)
        if not p.exists():
            log(f"skip (missing): {p}")
            continue
        cands = sorted(p.rglob("*")) if p.is_dir() else [p]
        for f in cands:
            if f.is_file() and f.suffix.lower() in MEDIA_EXT:
                items.append({"id": re.sub(r"[^\w\-]+", "_", f.stem), "source": str(f)})
    return items


def to_wav16k(src: Path, out: Path) -> Path:
    if src.suffix.lower() == ".wav":
        try:
            info = sf.info(src)
            if info.samplerate == SR and info.channels == 1:
                return src
        except Exception:  # noqa: BLE001
            pass
    safe_stem = re.sub(r"[^\w\-]+", "_", src.stem)
    dst = out / "_audio" / f"{safe_stem}.wav"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        if shutil.which("ffmpeg") is None:
            raise SystemExit("need ffmpeg to read non-wav media (brew install ffmpeg)")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                        "-vn", "-ac", "1", "-ar", str(SR), str(dst)], check=True)
    return dst


def fmt_ts(sec: float) -> str:
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}".replace(".", ",")


def cmd_asr(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "items.jsonl"

    items = collect_media(args.inputs, out)
    if args.limit:
        items = items[: args.limit]
    done = existing_ids(jsonl)
    todo = [it for it in items if it["id"] not in done]
    log(f"asr: {len(items)} input(s), {len(done)} already done, {len(todo)} to run")

    device = args.device if args.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = args.dtype if args.dtype != "auto" else ("fp16" if device == "mps" else "fp32")
    pipe = Transcriber(resolve_model(args.model), device, dtype, args.beams, None, args.max_new_tokens)
    chunk = args.chunk_seconds
    if chunk is None:
        chunk = 9.0 if pipe.kind == "whisper" else 30.0
    log(f"model={pipe.model_id} kind={pipe.kind} chunk={chunk or 'off'}s device={device}")

    t_start = time.time()
    for i, it in enumerate(todo, 1):
        t0 = time.time()
        rec = {"id": it["id"], "source": it["source"], "model": pipe.model_id}
        try:
            wav = to_wav16k(Path(it["source"]), out)
            audio, _ = librosa.load(wav, sr=SR, mono=True)
            duration = len(audio) / SR
            if chunk and duration > chunk:
                spans = split_on_silence(audio, chunk)
            else:
                spans = [(0.0, duration)]
            segs = []
            for a, b in spans:
                txt = pipe.decode_array(audio[int(a * SR): int(b * SR)]).strip()
                if txt:
                    segs.append({"start": round(a, 2), "end": round(b, 2), "text": txt})
            text = " ".join(s["text"] for s in segs)
            flags = []
            if not text.strip():
                flags.append("empty")
            cps = len(text) / duration if duration else 0
            if text and cps < 3:
                flags.append("low_char_rate")
            rec.update(text=text, chars=len(text), duration=round(duration, 2),
                       chunks=len(spans), segments=segs, seconds=round(time.time() - t0, 2),
                       rtf=round((time.time() - t0) / max(duration, 1e-9), 3),
                       status="ok", flags=flags)
            tdir = out / "transcripts"
            tdir.mkdir(exist_ok=True)
            (tdir / f"{it['id']}.txt").write_text(text, encoding="utf-8")
            if len(segs) > 1:
                srt = [f"{n}\n{fmt_ts(s['start'])} --> {fmt_ts(s['end'])}\n{s['text']}\n"
                       for n, s in enumerate(segs, 1)]
                (tdir / f"{it['id']}.srt").write_text("\n".join(srt), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            rec.update(text="", chars=0, duration=0, chunks=0, segments=[],
                       seconds=round(time.time() - t0, 2), rtf=None,
                       status=f"error: {type(exc).__name__}: {exc}"[:300], flags=["error"])
        append_jsonl(jsonl, rec)
        log(f"[{i}/{len(todo)}] {it['id']} {rec['status']} {rec['duration']}s audio "
            f"-> {rec['chars']} chars, RTF {rec['rtf']}")

    write_asr_report(out, elapsed=time.time() - t_start)


def write_asr_report(out: Path, elapsed: float) -> None:
    df = pd.read_json(out / "items.jsonl", lines=True, dtype={"id": str, "source": str})
    if df.empty:
        log("nothing to report")
        return
    df["flags"] = [[("empty" if int(c) == 0 else ("low_char_rate" if int(c) / max(float(d), 1e-9) < 3 else ""))
                    for c, d in [(c, d)]] [0] for c, d in zip(df.chars.fillna(0), df.duration.fillna(0))]
    df["flags"] = [[f] if f else [] for f in df["flags"]]
    ok = df[df.status == "ok"]
    df.drop(columns=["segments"], errors="ignore").to_csv(out / "items.csv", index=False)
    audio_min = ok.duration.sum() / 60 if len(ok) else 0
    flagged = df[df["flags"].map(lambda f: bool(f) and f != [] and f != ["error"])]
    lines = [
        "# ASR 批处理报告", "",
        f"- 模型：`{df.model.iloc[0] if 'model' in df else '?'}`",
        f"- 文件：{len(df)}（成功 {len(ok)}，失败 {len(df) - len(ok)}）",
        f"- 音频时长：{audio_min:.1f} 分钟",
        f"- 输出字符：{int(ok.chars.sum()):,}",
        f"- 总耗时：{elapsed/60:.1f} 分钟，RTF {elapsed/(audio_min*60):.3f}" if audio_min else "",
        "", f"## 需要复核（{len(flagged)}）", "",
        "| id | 时长 | 字符 | 字符/秒 | 标记 |", "|---|---:|---:|---:|---|",
    ]
    for r in flagged.itertuples():
        cps = r.chars / r.duration if r.duration else 0
        lines.append(f"| {r.id} | {r.duration}s | {r.chars} | {cps:.1f} | {','.join(r.flags)} |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"report -> {out/'report.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    o = sub.add_parser("ocr", help="images / PDFs / folders -> text")
    o.add_argument("inputs", nargs="+")
    o.add_argument("--out", required=True)
    o.add_argument("--engine", default="yigdzin", choices=["yigdzin", "tesseract"])
    o.add_argument("--dpi", type=int, default=250, help="PDF rasterisation DPI")
    o.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    o.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    o.add_argument("--max-new-tokens", type=int, default=3072)
    o.add_argument("--retry-loops", action="store_true", default=True)
    o.add_argument("--no-retry-loops", dest="retry_loops", action="store_false")
    o.add_argument("--no-dry", action="store_true")
    o.add_argument("--limit", type=int, default=0)
    o.add_argument("--offline", action="store_true", help="skip HuggingFace Hub metadata calls (models already cached)")
    o.set_defaults(func=cmd_ocr)

    a = sub.add_parser("asr", help="audio / video / folders -> transcripts")
    a.add_argument("inputs", nargs="+")
    a.add_argument("--out", required=True)
    a.add_argument("--model", default="whisper-small")
    a.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    a.add_argument("--dtype", default="auto", choices=["auto", "fp16", "fp32", "bf16"])
    a.add_argument("--beams", type=int, default=1)
    a.add_argument("--max-new-tokens", type=int, default=440)
    a.add_argument("--chunk-seconds", type=float, default=None,
                   help="default: 9s for Whisper, 30s for CTC; 0 disables")
    ap.add_argument("--offline", action="store_true", help="skip HuggingFace Hub metadata calls (models already cached)")
    a.add_argument("--limit", type=int, default=0)
    a.set_defaults(func=cmd_asr)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
