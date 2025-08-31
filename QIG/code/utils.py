import os, csv, json
from datetime import datetime, UTC
import uuid

def ensure_dir(path_or_file):
    d = path_or_file if os.path.isdir(path_or_file) else os.path.dirname(path_or_file)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def write_csv_append(path, header=None, rows=None):
    ensure_dir(path)
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file and header:
            w.writerow(header)
        for r in (rows or []):
            w.writerow(r)

def save_json(path, obj):
    ensure_dir(path)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def new_run_id():
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
