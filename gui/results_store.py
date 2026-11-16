"""Read benchmark snapshots (cloud runs) and local runs for the reporting UI.

Two sources are merged into one catalog so the GUI can show, for every model and
task, the per-item predictions:

  reports/large_models/   cloud runs published by the collaboration partner
  outputs/{ocr,asr}/      local runs produced by scripts/run_ocr.py / run_asr.py

Local runs are exposed under the id ``local:<run_dir_name>`` and are scored on the
same sample subset as the cloud runs so the table stays comparable.
"""

import csv
import json
import statistics
from pathlib import Path

from diffrender import analyse

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "large_models"
OUTPUTS = ROOT / "outputs"
LOCAL_PREFIX = "local:"
MIN_LOCAL_ITEMS = 20          # skip 3-page pilot runs
SHORT_CHARS = 150             # ASR "short utterance" subset, as in score_asr.py

_cache: dict[str, tuple[float, dict]] = {}


# --------------------------------------------------------------------- helpers
def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _num(value):
    if value in (None, "", "NaN", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _sample_ids(dataset: str) -> set[str]:
    """The fixed evaluation subset shared with the cloud runs."""
    path = (ROOT / "data" / "asr" / ("nict_test" if dataset == "nict" else "openpecha_bench")
            / "sample_ids.json")
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(data.get("sample") or data.get("all") or [])


def _ocr_ids() -> set[str]:
    path = ROOT / "data" / "ocr" / "sample_ids.json"
    return set(json.loads(path.read_text(encoding="utf-8"))["sample"]) if path.exists() else set()


# ------------------------------------------------------------------ local runs
def _local_metric(rows: list[dict], task: str, seconds_key: str) -> dict:
    from common import prepare_text

    cers = [c for c in (_num(r.get("cer")) for r in rows) if c is not None]
    seconds = [s for s in (_num(r.get(seconds_key)) for r in rows) if s is not None]
    out = {
        "cer": round(statistics.mean(cers), 6) if cers else None,
        "median_cer": round(statistics.median(cers), 6) if cers else None,
        "catastrophic_pct": round(100 * sum(c > 0.5 for c in cers) / len(cers), 2) if cers else None,
        "mean_seconds": round(statistics.mean(seconds), 3) if seconds else None,
        "client_peak_rss_gb": None,
        "remote_peak_memory_gb": None,
    }
    if task == "asr":
        swers = [s for s in (_num(r.get("swer")) for r in rows) if s is not None]
        out["swer"] = round(statistics.mean(swers), 6) if swers else None
        short = [r for r in rows if len(prepare_text(r.get("reference") or "")) <= SHORT_CHARS]
        scers = [c for c in (_num(r.get("cer")) for r in short) if c is not None]
        sswers = [s for s in (_num(r.get("swer")) for r in short) if s is not None]
        out["short_cer"] = round(statistics.mean(scers), 6) if scers else None
        out["short_swer"] = round(statistics.mean(sswers), 6) if sswers else None
        durations = [_num(r.get("duration")) or 0 for r in rows]
        total = sum(durations)
        out["rtf"] = round(sum(seconds) / total, 4) if total and seconds else None
    else:
        out.update(swer=None, short_cer=None, short_swer=None, rtf=None)
    return out


def _local_groups(rows: list[dict], task: str) -> list[dict]:
    fields = ("technology", "script") if task == "ocr" else ("dept", "speaker")
    groups: list[dict] = []
    for field in fields:
        buckets: dict[str, list[dict]] = {}
        for row in rows:
            key = row.get(field)
            if key in (None, ""):
                continue
            buckets.setdefault(str(key), []).append(row)
        for value, bucket in sorted(buckets.items()):
            cers = [_num(r.get("cer")) for r in bucket if _num(r.get("cer")) is not None]
            if not cers:
                continue
            item = {"field": field, "value": value, "n": len(bucket),
                    "cer": round(statistics.mean(cers), 6), "swer": None}
            if task == "asr":
                swers = [_num(r.get("swer")) for r in bucket if _num(r.get("swer")) is not None]
                item["swer"] = round(statistics.mean(swers), 6) if swers else None
            groups.append(item)
    return groups


def _load_local(run_dir: Path, task: str) -> dict | None:
    """Build a catalog entry + keep rows in the cache."""
    scored = run_dir / ("details_scored.csv" if task == "ocr" else "predictions_scored.csv")
    meta_path = run_dir / "run.json"
    if not (scored.exists() and meta_path.exists()):
        return None

    key = f"{LOCAL_PREFIX}{run_dir.name}"
    stamp = max(p.stat().st_mtime for p in (scored, meta_path))
    if key in _cache and _cache[key][0] == stamp:
        return _cache[key][1]

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rows = _read_csv(scored)
    dataset = "ocr" if task == "ocr" else (meta.get("dataset") or "nict")

    # score on the same subset as the cloud runs, so the table is comparable
    keep = _ocr_ids() if task == "ocr" else _sample_ids(dataset)
    if keep and len(rows) > len(keep):
        rows = [r for r in rows if r.get("id") in keep]

    if task == "ocr":
        for row in rows:
            row["reference"] = row.get("gt") or ""
            row["status"] = "ok" if str(row.get("status", "")).startswith("ok") else "error"
    else:
        for row in rows:
            status = str(row.get("status", "ok"))
            row["status"] = status if status in {"ok", "empty", "truncated", "refused"} else "error"
        for row in rows:
            row.setdefault("duration", None)

    for row in rows:
        for field in ("cer", "swer", "seconds", "duration", "rtf"):
            row[field] = _num(row.get(field))
        row["n_chars"] = int(_num(row.get("n_chars")) or 0)
        if "catastrophic" in row:
            row["catastrophic"] = str(row["catastrophic"]).lower() in {"true", "1", "yes"}

    alias = run_dir.name.split("_")[0]
    seconds_key = "seconds"
    ok = [r for r in rows if r["status"] == "ok"]
    entry = {
        "id": key,
        "alias": alias,
        "model": alias,
        "hf_id": meta.get("model"),
        "task": task,
        "dataset": dataset,
        "n": len(rows),
        "attempted": len(rows),
        "ok": len(ok),
        "errors": sum(1 for r in rows if r["status"] == "error"),
        "empty": sum(1 for r in rows if r["status"] == "empty"),
        "refused": sum(1 for r in rows if r["status"] == "refused"),
        "truncated": sum(1 for r in rows if r["status"] == "truncated"),
        "pending": 0,
        "state": "complete",
        "round": 1,
        "metrics": _local_metric(rows, task, seconds_key),
        "comparable": True,
        "channel": "local",
        "concurrency": 1,
        "error": "",
        "timing_scope": "本机单进程，含模型预热之外的全部推理时间",
        "updated_utc": meta.get("finished") or "",
        "baseline": False,
        "local": True,
        "run_id": run_dir.name,
        "note": (f"本地实测 · {meta.get('device', '?')}/{meta.get('dtype', '?')} · "
                 f"{run_dir.name}"),
        "has_details": True,
        "has_groups": True,
    }
    payload = {"entry": entry, "rows": rows,
               "groups": _local_groups(rows, task), "csv": scored}
    _cache[key] = (stamp, payload)
    return payload


def _local_all() -> list[dict]:
    found: list[dict] = []
    for task, folder in (("ocr", OUTPUTS / "ocr"), ("asr", OUTPUTS / "asr")):
        if not folder.exists():
            continue
        for run_dir in sorted(folder.iterdir()):
            if not run_dir.is_dir():
                continue
            payload = _load_local(run_dir, task)
            if payload and payload["entry"]["n"] >= MIN_LOCAL_ITEMS:
                found.append(payload["entry"])
    return found


# ---------------------------------------------------------------------- catalog
def _cloud_catalog() -> dict:
    path = REPORTS / "catalog.json"
    if not path.exists():
        return {"runs": [], "baselines": [], "updated_utc": ""}
    return json.loads(path.read_text(encoding="utf-8"))


def catalog() -> dict:
    """Cloud runs + local runs, with duplicated baselines removed."""
    data = _cloud_catalog()
    local = _local_all()
    if not local:
        return data
    covered = {(e["task"], e["dataset"], e["alias"]) for e in local}
    baselines = [b for b in data.get("baselines", [])
                 if (b["task"], b["dataset"], b["model"]) not in covered]
    runs = list(data.get("runs", [])) + local
    return {**data, "runs": runs, "baselines": baselines,
            "note_local": "标有「本地」的行由本机推理产生，可查看逐条预测与原数据文件。"}


# ------------------------------------------------------------------- accessors
def entry(run_id: str) -> dict:
    if run_id.startswith(LOCAL_PREFIX):
        name = run_id[len(LOCAL_PREFIX):]
        for task, folder in (("ocr", OUTPUTS / "ocr"), ("asr", OUTPUTS / "asr")):
            run_dir = folder / name
            if run_dir.is_dir():
                payload = _load_local(run_dir, task)
                if payload:
                    return payload["entry"]
        raise FileNotFoundError(run_id)
    return next(item for item in _cloud_catalog()["runs"] if item["id"] == run_id)


def _local_payload(run_id: str) -> dict:
    name = run_id[len(LOCAL_PREFIX):]
    for task, folder in (("ocr", OUTPUTS / "ocr"), ("asr", OUTPUTS / "asr")):
        run_dir = folder / name
        if run_dir.is_dir():
            payload = _load_local(run_dir, task)
            if payload:
                return payload
    raise FileNotFoundError(run_id)


def csv_path(run_id: str) -> Path:
    if run_id.startswith(LOCAL_PREFIX):
        return _local_payload(run_id)["csv"]
    entry(run_id)
    return REPORTS / "runs" / run_id / "predictions_scored.csv"


def rows(run_id: str) -> list[dict]:
    if run_id.startswith(LOCAL_PREFIX):
        return _local_payload(run_id)["rows"]
    path = csv_path(run_id)
    if not path.exists():
        return []
    data = _read_csv(path)
    for row in data:
        for field in ("cer", "swer", "seconds", "duration"):
            row[field] = _num(row.get(field))
        if row.get("status") not in {"ok", "empty", "truncated", "refused"}:
            row["cer"] = row["swer"] = None
    return data


def run(run_id: str) -> dict:
    item = entry(run_id)
    data = rows(run_id)
    if run_id.startswith(LOCAL_PREFIX):
        groups = _local_payload(run_id)["groups"]
    else:
        groups = []
        if item.get("has_groups"):
            groups = _read_csv(REPORTS / "runs" / run_id / "groups.csv")
    return {"run": item, "groups": groups,
            "items": [{k: v for k, v in row.items()
                       if k not in {"reference", "gt", "transcription"}}
                      for row in data]}


def sample(run_id: str, sample_id: str) -> dict:
    item = entry(run_id)
    row = next(row for row in rows(run_id) if row.get("id") == sample_id)
    result = {"run": item, "sample": row}
    if item["task"] == "ocr":
        result["media_url"] = f"/api/image/{sample_id}"
    else:
        result["media_url"] = f'/api/audio/{item["dataset"]}/{sample_id}'
    if row["status"] in {"ok", "empty", "truncated", "refused"}:
        from botok import Config, WordTokenizer
        import common
        pack = ROOT.parent / "LargeModel_Test_Related" / "resources" / "botok" / "general"
        if common._wt is None and pack.exists():
            common._wt = WordTokenizer(config=Config.from_path(pack))
        # local runs are scored against prepare_text(reference) only (BDRC protocol),
        # cloud runs against already-normalised references.
        result["diff"] = analyse(row.get("reference") or "", row.get("transcription") or "",
                                 item["task"], normalize_reference=False)
        result["diff"]["cer"] = row["cer"]
    return result
