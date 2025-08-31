#!/usr/bin/env python3
import csv, json
from pathlib import Path
from datetime import datetime, UTC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGS = ROOT / "figures"
RUNS = DATA / "runs"

SPIKES = DATA / "spikes.csv"
ENERGY = DATA / "energy.csv"
CONFIG = DATA / "config.json"

def latest_run_id():
    if CONFIG.exists():
        try:
            j = json.loads(CONFIG.read_text())
            return j.get("run_id")
        except: pass
    # fallback: spikes.csv의 마지막 run_id 사용
    last = None
    if SPIKES.exists():
        with SPIKES.open() as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                last = row[0] if row else last
    return last

def rows_for(csv_path, run_id):
    out = []
    if not csv_path.exists() or not run_id: return out
    with csv_path.open() as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            if row and row[0] == run_id:
                out.append(row)
    return out

def main():
    run_id = latest_run_id()
    if not run_id:
        print("no run_id found.")
        return
    # 아카이브 디렉토리(있으면 사용, 없으면 만들기)
    run_fig = (ROOT/"figures"/"runs"/run_id); run_fig.mkdir(parents=True, exist_ok=True)
    run_dat = (ROOT/"data"/"runs"/run_id); run_dat.mkdir(parents=True, exist_ok=True)

    sp = rows_for(SPIKES, run_id)
    en = rows_for(ENERGY, run_id)

    # 요약 CSV
    sp_out = run_dat/"spikes_subset.csv"
    en_out = run_dat/"energy_subset.csv"
    sp_out.write_text("run_id,timestamp,alpha,spikes\n" + "\n".join(",".join(r) for r in sp) + "\n")
    en_out.write_text("run_id,timestamp,alpha,energy_proxy\n" + "\n".join(",".join(r) for r in en) + "\n")

    # 요약 그래프
    try:
        import matplotlib.pyplot as plt
        alphas = [float(r[2]) for r in sp]
        spikes = [int(r[3]) for r in sp]
        if alphas:
            plt.figure(figsize=(6,3))
            plt.bar([str(a) for a in alphas], spikes)
            plt.xlabel("alpha"); plt.ylabel("spikes")
            plt.title(f"Run {run_id} — spikes by alpha")
            plt.tight_layout()
            out = run_fig / "spikes_bar.png"
            plt.savefig(out, dpi=140)
            plt.close()
            print(f"[OK] {out}")
    except Exception as e:
        print(f"[WARN] plot skipped: {e}")

    # 블로그/로그에 바로 붙여넣을 마크다운 스니펫
    md = []
    md.append(f"### Run `{run_id}` 요약")
    if sp:
        md.append("")
        md.append("| alpha | spikes |")
        md.append("|---:|---:|")
        for r in sp:
            md.append(f"| {r[2]} | {r[3]} |")
        md.append("")
        md.append(f"![spikes_bar](figures/runs/{run_id}/spikes_bar.png)")
    (run_dat/"README_snippet.md").write_text("\n".join(md))
    print(f"[OK] snippet → {run_dat/'README_snippet.md'}")

    # 매니페스트
    manifest = {
        "run_id": run_id,
        "archived_at_utc": datetime.now(UTC).isoformat(),
        "spikes_rows": len(sp),
        "energy_rows": len(en),
        "artifacts": {
            "spikes_subset": str(sp_out.relative_to(ROOT)),
            "energy_subset": str(en_out.relative_to(ROOT)),
            "spikes_bar_png": f"figures/runs/{run_id}/spikes_bar.png",
            "snippet_md": str((run_dat/'README_snippet.md').relative_to(ROOT)),
        }
    }
    (run_dat/"manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[OK] manifest → {run_dat/'manifest.json'}")

if __name__ == "__main__":
    main()