# dtg_simulation.py
import os
import numpy as np
# --- 헤드리스 환경 안전 ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, UTC

from lif_model import LIFNeuron, dynamic_threshold
from utils import ensure_dir, write_csv_append, save_json, new_run_id

# ====== 추가 (자동화 지원) ======
import argparse
from pathlib import Path
import json

# ---- 명령줄 인자 파싱 ----
ap = argparse.ArgumentParser(add_help=False)
ap.add_argument("--alpha", type=float, default=None, help="단일 alpha만 실행 (예: --alpha 0.7)")
ap.add_argument("--outdir", type=str, default=None, help="그림 저장 폴더 오버라이드 (예: --outdir figures/run_123)")
ap.add_argument("--seed", type=int, default=None, help="난수 시드 고정(옵션)")
try:
    _cli_args, _ = ap.parse_known_args()
except SystemExit:
    class _Dummy: ...
    _cli_args = _Dummy()
    _cli_args.alpha = None
    _cli_args.outdir = None
    _cli_args.seed = None

# ---- 외부 제어 변수 ----
ALPHA_OVERRIDE = _cli_args.alpha
OUTDIR_OVERRIDE = Path(_cli_args.outdir) if _cli_args.outdir else None
SEED_OVERRIDE = _cli_args.seed

# ===== 고정 파라미터(재현성) =====
DT          = 1e-3       # 1 ms
T_END       = 1.0        # 1 s
TAU         = 20e-3
V_TH_BASE   = 1.0
I_CONST     = 1.10       # 안정 스파이킹 기본값(필요 시 조정)
ALPHAS      = [1.0, 0.7, 0.5]
REFRACT_MS  = 2.0        # 2 ms 불응기 (0이면 비활성)

# ===== 기본 경로 =====
FIG_DIR     = "figures"
SPIKE_CSV   = "data/spikes.csv"
ENERGY_CSV  = "data/energy.csv"
CONFIG_JSON = "data/config.json"


def run_one(alpha: float, run_id: str, save_dir: str | Path):
    """alpha 하나에 대해 시뮬레이션 1회 실행 및 저장."""
    # (옵션) 시드 고정 — 지금은 난수 사용 안하지만 향후 대비
    if SEED_OVERRIDE is not None:
        np.random.seed(SEED_OVERRIDE)

    t = np.arange(0.0, T_END, DT)
    th = dynamic_threshold(t, v_th_base=V_TH_BASE, alpha=alpha)

    neuron = LIFNeuron(
        dt=DT, tau=TAU, v_rest=0.0, v_reset=0.0,
        v_th_base=V_TH_BASE, refractory_ms=REFRACT_MS
    )

    v_trace = np.empty_like(t)
    spikes_mask = np.zeros_like(t, dtype=bool)

    for i, v_th in enumerate(th):
        v, spiked = neuron.step(I=I_CONST, v_th=v_th)
        v_trace[i] = v
        if spiked:
            spikes_mask[i] = True

    total_spikes = int(spikes_mask.sum())
    energy_proxy = float(total_spikes)  # 단순 근사: 스파이크 수

    # ----- 그림 저장 -----
    # outdir 오버라이드가 있으면 거기로, 없으면 기본 figures
    save_dir = Path(save_dir)
    fig_path = save_dir / f"membrane_alpha_{alpha}.png"
    ensure_dir(str(fig_path))  # 파일 경로를 넣어도 상위 폴더를 만들어주는 유틸(주석에 명시됨)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, v_trace, label="V(t)")
    ax.plot(t, th, "--", label="V_th(t)")
    if total_spikes > 0:
        ax.scatter(t[spikes_mask], th[spikes_mask], s=10, label="spike")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("V")
    ax.set_title(f"Membrane Potential (alpha={alpha})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(fig_path), dpi=160)
    plt.close(fig)

    # ----- CSV 누적 저장 -----
    ts = datetime.now(UTC).isoformat()
    write_csv_append(
        SPIKE_CSV,
        header=["run_id", "timestamp", "alpha", "spikes"],
        rows=[[run_id, ts, alpha, total_spikes]],
    )
    write_csv_append(
        ENERGY_CSV,
        header=["run_id", "timestamp", "alpha", "energy_proxy"],
        rows=[[run_id, ts, alpha, energy_proxy]],
    )

    print(f"[alpha={alpha}] spikes={total_spikes}, energy_proxy={energy_proxy}, fig={fig_path}")
    return total_spikes, energy_proxy


def main():
    # outdir 기본값 결정: 미지정 시 figures, 지정 시 해당 폴더
    # (팁) 실험마다 폴더 분리하고 싶으면 아래 한 줄을 OUTDIR_OVERRIDE or f"{FIG_DIR}/run_{run_id}"로 바꿔도 됨
    run_id = new_run_id()
    outdir = OUTDIR_OVERRIDE if OUTDIR_OVERRIDE else Path(FIG_DIR)

    # 실행 설정 기록
    save_json(
        CONFIG_JSON,
        {
            "run_id": run_id,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "params": {
                "DT": DT, "T_END": T_END, "TAU": TAU,
                "V_TH_BASE": V_TH_BASE, "I_CONST": I_CONST,
                "ALPHAS": ALPHAS if ALPHA_OVERRIDE is None else [ALPHA_OVERRIDE],
                "REFRACT_MS": REFRACT_MS,
                "OUTDIR": str(outdir),
                "SEED": SEED_OVERRIDE,
            },
        },
    )

    # 단일 alpha 오버라이드 지원
    alphas = [ALPHA_OVERRIDE] if ALPHA_OVERRIDE is not None else ALPHAS
    for alpha in alphas:
        run_one(alpha, run_id, outdir)


if __name__ == "__main__":
    main()