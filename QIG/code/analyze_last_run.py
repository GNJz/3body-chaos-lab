#!/usr/bin/env python3
import csv, json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"

SPIKES_CSV = DATA / "spikes.csv"
RUNS_DIR = DATA / "runs"

def last_run_id_from_manifests():
    # data/runs/*/manifest.json 중 가장 최근 디렉터리 선택
    runs = sorted(RUNS_DIR.glob("*"), reverse=True)
    for d in runs:
        mf = d / "manifest.json"
        if mf.exists():
            try:
                j = json.loads(mf.read_text(encoding="utf-8"))
                return j.get("run_id"), d
            except Exception:
                pass
    return None, None

def rows_for_run(csv_path: Path, run_id: str):
    out = []
    with csv_path.open("r", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            if row and row[0] == run_id:
                out.append(row)
    return out

def main():
    run_id, run_dir = last_run_id_from_manifests()
    if not run_id:
        raise SystemExit("최근 run_id를 찾지 못했습니다. 먼저 `make summarize`를 실행하세요.")

    rows = rows_for_run(SPIKES_CSV, run_id)
    if not rows:
        raise SystemExit(f"spikes.csv에 run_id={run_id} 데이터가 없습니다.")

    # rows: [run_id, timestamp, alpha, spikes]
    alphas = [float(r[2]) for r in rows]
    spikes = [int(r[3]) for r in rows]

    # 정렬
    pairs = sorted(zip(alphas, spikes))
    alphas, spikes = zip(*pairs)

    # 표 저장
    out_csv = run_dir / "summary_table.csv"
    out_csv.write_text(
        "alpha,spikes\n" + "\n".join(f"{a},{s}" for a, s in pairs) + "\n",
        encoding="utf-8"
    )

    # 그래프 저장
    fig_out = run_dir / "spikes_line.png"
    plt.figure(figsize=(5,3))
    plt.plot(alphas, spikes, marker="o")
    plt.xlabel("alpha"); plt.ylabel("spikes")
    plt.title(f"Run {run_id} — spikes vs. alpha (line)")
    plt.tight_layout()
    plt.savefig(fig_out, dpi=140)
    plt.close()

    print(f"[OK] table  → {out_csv}")
    print(f"[OK] figure → {fig_out}")

if __name__ == "__main__":
    main()