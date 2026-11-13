"""Read atomically published benchmark snapshots for the reporting UI."""

import csv
import json
from pathlib import Path

from diffrender import analyse

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "large_models"


def catalog():
    path = REPORTS / "catalog.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"runs": [], "baselines": []}


def entry(run_id):
    return next(item for item in catalog()["runs"] if item["id"] == run_id)


def csv_path(run_id):
    entry(run_id)
    return REPORTS / "runs" / run_id / "predictions_scored.csv"


def rows(run_id):
    path = csv_path(run_id)
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as file:
        data = list(csv.DictReader(file))
    for row in data:
        for field in ("cer", "swer", "seconds", "duration"):
            row[field] = float(row[field]) if row.get(field) else None
        if row["status"] not in {"ok", "empty", "truncated", "refused"}:
            row["cer"] = row["swer"] = None
    return data


def run(run_id):
    item = entry(run_id)
    data = rows(run_id)
    groups = []
    if item.get("has_groups"):
        with (REPORTS / "runs" / run_id / "groups.csv").open(encoding="utf-8-sig", newline="") as file:
            groups = list(csv.DictReader(file))
    return {"run": item, "groups": groups, "items": [{key: value for key, value in row.items()
                                                       if key not in {"reference", "transcription"}} for row in data]}


def sample(run_id, sample_id):
    item = entry(run_id)
    row = next(row for row in rows(run_id) if row["id"] == sample_id)
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
        result["diff"] = analyse(row["reference"], row["transcription"], item["task"], normalize_reference=False)
        result["diff"]["cer"] = row["cer"]
    return result
