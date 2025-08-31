#!/usr/bin/env python3
"""
DTG(LIF) 자동화 실행 스크립트
- alpha 스윕 실행
- 실행별 전용 출력 폴더(run_타임스탬프) 생성
- 각 alpha 결과를 CSV/PNG로 수집
- 최신 대표 그래프를 figures/ 로도 동시 복사(가독성)
"""
import argparse, subprocess, sys, shutil, json, time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
DATA = ROOT / "data"
FIGS = ROOT / "figures"
LOGS = ROOT / "logs"

def sh(cmd: list[str], cwd=None):
    p = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return p.stdout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2",
                    help="쉼표구분 리스트")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 실행 세션 폴더
    stamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    run_dir = DATA / f"run_{stamp}"
    figs_dir = FIGS / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    # 메타 로그
    meta = {
        "timestamp": stamp,
        "alphas": args.alphas,
        "seed": args.seed,
        "script": "run_experiment.py",
        "dtg_script": "dtg_simulation.py --alpha --outdir --seed",
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # 스윕
    alphas = [a.strip() for a in args.alphas.split(",") if a.strip()]
    summary_rows = []
    for a in alphas:
        outdir = run_dir / f"alpha_{a}"
        outdir.mkdir(exist_ok=True)
        print(f"[RUN] alpha={a} → {outdir}")

        # dtg_simulation.py가 인자(--alpha/--outdir/--seed)를 지원한다는 가정
        # (아래 2단계의 '아주 작은 패치'를 적용하면 됩니다)
        sh([sys.executable, str(CODE / "dtg_simulation.py"),
            "--alpha", a,
            "--outdir", str(outdir),
            "--seed", str(args.seed)
        ])

        # dtg_simulation이 outdir 안에 결과 요약 JSON을 남긴다고 가정
        # (아래 패치 예시 포함) 없으면 건너뜀
        summary_json = outdir / "summary.json"
        if summary_json.exists():
            s = json.loads(summary_json.read_text(encoding="utf-8"))
            summary_rows.append(s)

        # 대표 PNG를 figures/run_타임스탬프/ & figures/ 루트에도 복사
        # 파일명은 dtg_simulation에서 저장한 이름 규칙을 사용
        for cand in ["membrane.png", f"membrane_alpha_{a}.png"]:
            src = outdir / cand
            if src.exists():
                shutil.copy2(src, figs_dir / src.name)
                # 최신 대표본은 figures 루트에도 덮어쓰기 복사(가독성)
                shutil.copy2(src, FIGS / src.name)
                break

        time.sleep(0.1)

    # 스윕 요약 CSV 저장
    if summary_rows:
        import csv
        csv_path = run_dir / "sweep_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=sorted(summary_rows[0].keys()))
            wr.writeheader()
            wr.writerows(summary_rows)
        print(f"[OK] sweep_summary.csv saved → {csv_path}")

    print(f"\n[DONE] run dir: {run_dir}")
    print(f"[DONE] figs dir: {figs_dir}")

if __name__ == "__main__":
    main()