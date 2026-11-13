"""Local evaluation GUI for Tibetan OCR / ASR (stdlib only, no extra deps).

    python gui/server.py --port 8765            # then open http://127.0.0.1:8765
    python gui/server.py --offline              # skip HuggingFace Hub metadata calls

Design notes
  * one global inference lock: MPS is shared, concurrent runs are slower not faster
  * long jobs run in a background thread; the page polls /api/job/<id> every 500 ms
  * uploads are raw request bodies (filename/blade in headers) -> no multipart parsing
  * every metric delegates to scripts/common.py so the GUI matches reports/*.csv
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "gui"))

from common import ASR_DIR, OCR_DIR  # noqa: E402
from diffrender import analyse  # noqa: E402

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 40 * 1024 * 1024

OCR_ENGINES = ["yigdzin", "tesseract"]
ASR_MODELS = {
    "wav2vec2-base": "billingsmoore/tibetan-asr-nict-tib1-wav2vec2-base",
    "whisper-small": "billingsmoore/tibetan-asr-nict-tib1-whisper-small",
    "whisper-base": "billingsmoore/tibetan-asr-nict-tib1-whisper-base",
    "whisper-tiny": "billingsmoore/tibetan-asr-nict-tib1-whisper-tiny",
    "tenzin-small": "tenzin/whisper-small-tibetan",
    "openpecha-general": "openpecha/general_stt_base_model",
}

INFER_LOCK = threading.Lock()
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
_engines: dict[str, object] = {}
_device: str | None = None


def device() -> str:
    global _device
    if _device is None:
        import torch

        _device = "mps" if torch.backends.mps.is_available() else "cpu"
    return _device


# --------------------------------------------------------------------- samples
def _clean(obj):
    """Replace NaN/inf with None so the JSON stays valid for browsers."""
    import math

    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def load_ocr_samples() -> list[dict]:
    import pandas as pd

    meta = pd.read_csv(OCR_DIR / "metadata.csv", dtype={"id": str})
    return _clean(meta.to_dict("records"))


def load_asr_samples() -> dict[str, list[dict]]:
    import pandas as pd

    out = {}
    for key, sub in (("nict", "nict_test"), ("openpecha", "openpecha_bench")):
        p = ASR_DIR / sub / "metadata.csv"
        df = pd.read_csv(p, dtype={"id": str, "speaker": str})
        out[key] = _clean(df.to_dict("records"))
    return out


OCR_SAMPLES = load_ocr_samples()
ASR_SAMPLES = load_asr_samples()
OCR_BY_ID = {r["id"]: r for r in OCR_SAMPLES}


def ocr_ref(page_id: str) -> str:
    import pandas as pd

    df = pd.read_csv(OCR_DIR / "ground_truth.csv", dtype={"id": str}).set_index("id")
    return str(df.transcription.get(page_id, "") or "")


def asr_ref(dataset: str, item_id: str) -> str:
    for r in ASR_SAMPLES[dataset]:
        if r["id"] == item_id:
            return str(r["transcription"] or "")
    return ""


# --------------------------------------------------------------------- engines
def ocr_engine(name: str):
    if name in _engines:
        return _engines[name]
    from run_ocr import TesseractEngine, YigdzinEngine

    if name == "tesseract":
        eng = TesseractEngine()
    else:
        dtype = "bf16" if device() == "mps" else "fp32"
        eng = YigdzinEngine(device(), dtype, 3072, use_dry=True)
    _engines[name] = eng
    return eng


def asr_pipe(model_alias: str, chunk: float | None):
    key = f"asr:{model_alias}:{chunk}"
    if key in _engines:
        return _engines[key]
    from run_asr import Transcriber

    dtype = "fp16" if device() == "mps" else "fp32"
    pipe = Transcriber(ASR_MODELS.get(model_alias, model_alias), device(), dtype, 1, None, 440)
    _engines[key] = pipe
    return pipe


# ----------------------------------------------------------------------- jobs
def new_job(kind: str, label: str) -> str:
    jid = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[jid] = {"id": jid, "kind": kind, "label": label, "status": "queued",
                     "started": time.time(), "elapsed": 0.0, "result": None, "error": None}
    return jid


def finish_job(jid: str, result: dict | None = None, error: str | None = None) -> None:
    with JOBS_LOCK:
        j = JOBS[jid]
        j["status"] = "error" if error else "done"
        j["result"] = result
        j["error"] = error
        j["elapsed"] = round(time.time() - j["started"], 2)


def run_ocr_job(jid: str, image_path: Path, engine_name: str, ref: str | None) -> None:
    try:
        with INFER_LOCK:
            with JOBS_LOCK:
                JOBS[jid]["status"] = "running"
            t0 = time.time()
            eng = ocr_engine(engine_name)
            text = eng(image_path, temperature=0.0)[0]
            took = time.time() - t0
        res = {"text": text, "engine": engine_name, "took": round(took, 2),
               "device": device(), "chars": len(text)}
        if ref:
            res.update(analyse(ref, text, "ocr"))
        finish_job(jid, res)
    except Exception:  # noqa: BLE001
        finish_job(jid, error=traceback.format_exc(limit=3))


def run_asr_job(jid: str, wav_path: Path, model_alias: str, chunk: float, ref: str | None) -> None:
    try:
        import librosa

        from predict_asr import split_on_silence

        with INFER_LOCK:
            with JOBS_LOCK:
                JOBS[jid]["status"] = "running"
            t0 = time.time()
            pipe = asr_pipe(model_alias, chunk)
            audio, _ = librosa.load(wav_path, sr=16000, mono=True)
            duration = len(audio) / 16000
            spans = split_on_silence(audio, chunk) if chunk and duration > chunk else [(0.0, duration)]
            segments, parts = [], []
            for a, b in spans:
                txt = pipe.decode_array(audio[int(a * 16000): int(b * 16000)]).strip()
                if txt:
                    segments.append({"start": round(a, 2), "end": round(b, 2), "text": txt})
                    parts.append(txt)
            text = " ".join(parts)
            took = time.time() - t0
        res = {"text": text, "model": model_alias, "duration": round(duration, 2),
               "took": round(took, 2), "rtf": round(took / max(duration, 1e-9), 3),
               "chars": len(text), "segments": segments, "device": device(),
               "chunk": chunk or 0}
        if ref:
            res.update(analyse(ref, text, "asr"))
        finish_job(jid, res)
    except Exception:  # noqa: BLE001
        finish_job(jid, error=traceback.format_exc(limit=3))


def _clean_ref(value: str | None) -> str | None:
    if not value:
        return None
    return unquote(value)


def to_wav16k(src: Path) -> Path:
    from pipeline import to_wav16k as conv

    tmp = Path(tempfile.mkdtemp(prefix="gui_asr_"))
    return conv(src, tmp)


# ------------------------------------------------------------------ HTTP layer
class Handler(BaseHTTPRequestHandler):
    server_version = "TibetEval/1.0"

    def log_message(self, fmt, *args):  # quieter console
        if self.path.startswith("/api/health"):
            return
        sys.stderr.write("%s %s\n" % (self.command, self.path))

    # -- helpers
    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def _err(self, code: int, msg: str) -> None:
        self._json({"error": msg}, code)

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD:
            raise ValueError(f"文件过大（{n/1e6:.1f}MB > 40MB）")
        return self.rfile.read(n)

    def _save_upload(self, suffix_hint: str) -> Path:
        name = (self.headers.get("X-Filename") or f"upload{suffix_hint}")
        name = Path(name).name
        suffix = Path(name).suffix or suffix_hint
        tmpdir = Path(tempfile.mkdtemp(prefix="gui_up_"))
        path = tmpdir / (Path(name).stem[:60] + suffix)
        path.write_bytes(self._body())
        return path

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        q = parse_qs(u.query)
        p = u.path
        try:
            if p in ("/", "/index.html"):
                return self._static("index.html")
            if p.startswith("/static/"):
                return self._static(p[len("/static/"):])
            if p == "/api/health":
                import torch

                return self._json({"status": "ok", "device": device(),
                                   "torch": torch.__version__,
                                   "loaded": sorted(_engines.keys())})
            if p == "/api/samples/ocr":
                rows = OCR_SAMPLES
                for field in ("technology", "script", "legibility"):
                    val = (q.get(field) or [""])[0]
                    if val:
                        rows = [r for r in rows if str(r[field]).startswith(val)]
                limit = int((q.get("limit") or ["200"])[0])
                cols = ("id", "technology", "script", "legibility", "n_chars")
                return self._json({"total": len(rows),
                                   "items": [{k: r[k] for k in cols} for r in rows[:limit]]})
            if p.startswith("/api/samples/ocr/"):
                pid = p.rsplit("/", 1)[-1]
                r = OCR_BY_ID.get(pid)
                if not r:
                    return self._err(404, "sample not found")
                return self._json({**r, "ref": ocr_ref(pid),
                                   "image_url": f"/api/image/{pid}"})
            if p.startswith("/api/image/"):
                pid = p.rsplit("/", 1)[-1]
                img = OCR_DIR / "images" / f"{pid}.png"
                if not img.exists():
                    return self._err(404, "image not found")
                return self._send(200, img.read_bytes(), "image/png")
            if p == "/api/samples/asr":
                ds = (q.get("dataset") or ["nict"])[0]
                rows = ASR_SAMPLES.get(ds, [])
                dept = (q.get("dept") or [""])[0]
                spk = (q.get("speaker") or [""])[0]
                if dept:
                    rows = [r for r in rows if r["dept"] == dept]
                if spk:
                    rows = [r for r in rows if str(r.get("speaker", "")) == spk]
                limit = int((q.get("limit") or ["200"])[0])
                depts = sorted({r["dept"] for r in ASR_SAMPLES.get(ds, [])})
                return self._json({"total": len(rows), "depts": depts,
                                   "items": [{k: r.get(k) for k in ("id", "dept", "speaker", "duration")}
                                             for r in rows[:limit]]})
            if p.startswith("/api/samples/asr/"):
                _, _, _, _, ds, iid = p.split("/", 5)
                hit = next((r for r in ASR_SAMPLES.get(ds, []) if r["id"] == iid), None)
                if not hit:
                    return self._err(404, "sample not found")
                return self._json({**hit, "ref": hit["transcription"],
                                   "audio_url": f"/api/audio/{ds}/{iid}"})
            if p.startswith("/api/audio/"):
                _, _, _, ds, iid = p.split("/", 4)
                wav = ASR_DIR / ("nict_test" if ds == "nict" else "openpecha_bench") / "wav" / f"{iid}.wav"
                if not wav.exists():
                    return self._err(404, "audio not found")
                return self._serve_audio(wav)
            if p.startswith("/api/job/"):
                jid = p.rsplit("/", 1)[-1]
                with JOBS_LOCK:
                    j = JOBS.get(jid)
                    if not j:
                        return self._err(404, "job not found")
                    j = dict(j)
                if j["status"] in ("queued", "running"):
                    j["elapsed"] = round(time.time() - j["started"], 1)
                return self._json(j)
            return self._err(404, "not found")
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            return self._err(500, traceback.format_exc(limit=2))

    def do_POST(self):  # noqa: N802
        u = urlparse(self.path)
        p = u.path
        try:
            if p == "/api/sample/ocr":
                body = json.loads(self._body() or b"{}")
                pid = body["id"]
                ref = ocr_ref(pid)
                jid = new_job("ocr", f"sample {pid}")
                threading.Thread(target=run_ocr_job, daemon=True, args=(
                    jid, OCR_DIR / "images" / f"{pid}.png", body.get("engine", "yigdzin"), ref)).start()
                return self._json({"job": jid})
            if p == "/api/sample/asr":
                body = json.loads(self._body() or b"{}")
                ds, iid = body["dataset"], body["id"]
                folder = "nict_test" if ds == "nict" else "openpecha_bench"
                chunk = float(body.get("chunk") or 0)
                jid = new_job("asr", f"{ds}/{iid}")
                threading.Thread(target=run_asr_job, daemon=True, args=(
                    jid, ASR_DIR / folder / "wav" / f"{iid}.wav",
                    body.get("model", "whisper-small"), chunk, asr_ref(ds, iid))).start()
                return self._json({"job": jid})
            if p == "/api/infer/ocr":
                path = self._save_upload(".png")
                ref = _clean_ref(self.headers.get("X-Ref"))
                jid = new_job("ocr", path.name)
                threading.Thread(target=run_ocr_job, daemon=True, args=(
                    jid, path, self.headers.get("X-Engine", "yigdzin"), ref)).start()
                return self._json({"job": jid})
            if p == "/api/infer/asr":
                path = self._save_upload(".wav")
                wav = to_wav16k(path)
                ref = _clean_ref(self.headers.get("X-Ref"))
                chunk = float(self.headers.get("X-Chunk") or 0)
                jid = new_job("asr", path.name)
                threading.Thread(target=run_asr_job, daemon=True, args=(
                    jid, wav, self.headers.get("X-Model", "whisper-small"), chunk, ref)).start()
                return self._json({"job": jid})
            if p in ("/api/batch/ocr", "/api/batch/asr"):
                body = json.loads(self._body() or b"{}")
                jid = new_job("batch", f"{p.rsplit('/',1)[-1]} n={body.get('n', 5)}")
                threading.Thread(target=run_batch, daemon=True, args=(jid, p.endswith("ocr"), body)).start()
                return self._json({"job": jid})
            return self._err(404, "not found")
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            return self._err(500, traceback.format_exc(limit=2))

    # -- static / audio
    def _static(self, rel: str):
        path = (STATIC / rel).resolve()
        if not str(path).startswith(str(STATIC.resolve())) or not path.is_file():
            return self._err(404, "static not found")
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix in (".js", ".css", ".html"):
            ctype += "; charset=utf-8"
        return self._send(200, path.read_bytes(), ctype)

    def _serve_audio(self, path: Path):
        data = path.read_bytes()
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            try:
                s, e = rng[6:].split("-")
                start = int(s or 0)
                end = int(e) if e else len(data) - 1
                chunk = data[start: end + 1]
                self.send_response(206)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            except ValueError:
                return self._send(200, data, "audio/wav")
        else:
            chunk = data
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        try:
            self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


def run_batch(jid: str, is_ocr: bool, body: dict) -> None:
    """Serial batch over N sampled items; writes progress into the job record."""
    import random

    n = max(1, min(int(body.get("n", 5)), 50))
    seed = int(body.get("seed", 42))
    rng = random.Random(seed)
    try:
        with JOBS_LOCK:
            JOBS[jid]["status"] = "running"
        rows: list[dict] = []
        if is_ocr:
            pool = [r for r in OCR_SAMPLES
                    if (not body.get("technology") or r["technology"] == body["technology"])
                    and (not body.get("script") or str(r["script"]).startswith(body["script"]))]
            picks = rng.sample(pool, min(n, len(pool)))
            for i, r in enumerate(picks, 1):
                with INFER_LOCK:
                    t0 = time.time()
                    text = ocr_engine(body.get("engine", "yigdzin"))(
                        OCR_DIR / "images" / f"{r['id']}.png", temperature=0.0)[0]
                    took = time.time() - t0
                res = analyse(ocr_ref(r["id"]), text, "ocr")
                rows.append({"id": r["id"], "technology": r["technology"], "script": r["script"],
                             "cer": res["cer"], "seconds": round(took, 1),
                             "ref_len": res["ref_len"], "pred": text[:200]})
                with JOBS_LOCK:
                    JOBS[jid]["progress"] = f"{i}/{len(picks)}"
        else:
            ds = body.get("dataset", "nict")
            pool = ASR_SAMPLES[ds]
            if body.get("dept"):
                pool = [r for r in pool if r["dept"] == body["dept"]]
            picks = rng.sample(pool, min(n, len(pool)))
            chunk = float(body.get("chunk") or 0)
            for i, r in enumerate(picks, 1):
                wav = ASR_DIR / ("nict_test" if ds == "nict" else "openpecha_bench") / "wav" / f"{r['id']}.wav"
                with INFER_LOCK:
                    t0 = time.time()
                    pipe = asr_pipe(body.get("model", "whisper-small"), chunk)
                    import librosa

                    from predict_asr import split_on_silence

                    audio, _ = librosa.load(wav, sr=16000, mono=True)
                    dur = len(audio) / 16000
                    spans = split_on_silence(audio, chunk) if chunk and dur > chunk else [(0.0, dur)]
                    text = " ".join(pipe.decode_array(audio[int(a * 16000): int(b * 16000)]).strip()
                                    for a, b in spans).strip()
                    took = time.time() - t0
                res = analyse(r["transcription"], text, "asr")
                rows.append({"id": r["id"], "dept": r["dept"], "cer": res["cer"], "swer": res["swer"],
                             "duration": dur, "seconds": round(took, 1), "pred": text[:200]})
                with JOBS_LOCK:
                    JOBS[jid]["progress"] = f"{i}/{len(picks)}"

        import statistics

        key2 = "swer" if not is_ocr else None
        cers = sorted(x["cer"] for x in rows)
        summary = {
            "n": len(rows),
            "mean_cer": round(statistics.mean(cers), 4),
            "median_cer": round(statistics.median(cers), 4),
            "p90_cer": round(cers[min(len(cers) - 1, int(0.9 * len(cers)))], 4),
            "catastrophic_pct": round(100 * sum(c > 0.5 for c in cers) / len(cers), 1),
        }
        if key2:
            swers = [x["swer"] for x in rows]
            summary["mean_swer"] = round(statistics.mean(swers), 4)
        finish_job(jid, {"table": sorted(rows, key=lambda x: -x["cer"]), "summary": summary})
    except Exception:  # noqa: BLE001
        finish_job(jid, error=traceback.format_exc(limit=3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Tibetan OCR/ASR evaluation GUI -> http://{args.host}:{args.port}")
    print(f"  device: {device()}   samples: {len(OCR_SAMPLES)} OCR / "
          f"{len(ASR_SAMPLES['nict'])} NICT / {len(ASR_SAMPLES['openpecha'])} OpenPecha")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
