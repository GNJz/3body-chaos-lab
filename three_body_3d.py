#!/usr/bin/env python3
# 3D Three-Body simulation
# - CSV/PNG 저장
# - 배치 실행(exp1~3)
# - 에너지 드리프트 체크
# - Lyapunov(Benettin)
# - Lyapunov 그리드 스윕(heatmap + contour)
# - dt 스윕(정밀도 변화 vs 에너지 드리프트/리야푸노프)

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ----------------------- 공통 설정 -----------------------
G = 1.0
m1 = m2 = m3 = 1.0
MASSES = np.array([m1, m2, m3], dtype=np.float64)
EPS = 1e-9   # zero-div softening

# 안전한 dt 포맷(파일명 충돌 방지: 0.0025 같은 값도 깔끔히)
def _fmt_dt(x: float) -> str:
    return f"{x:.6f}".rstrip("0").rstrip(".")

# ----------------------- 물리 코어 -----------------------
def rhs(t, s, G=G, m1=m1, m2=m2, m3=m3):
    x1,y1,z1,vx1,vy1,vz1, x2,y2,z2,vx2,vy2,vz2, x3,y3,z3,vx3,vy3,vz3 = s

    r12 = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2 + EPS)
    r13 = np.sqrt((x3-x1)**2 + (y3-y1)**2 + (z3-z1)**2 + EPS)
    r23 = np.sqrt((x3-x2)**2 + (y3-y2)**2 + (z3-z2)**2 + EPS)

    ax1 = G*m2*(x2-x1)/r12**3 + G*m3*(x3-x1)/r13**3
    ay1 = G*m2*(y2-y1)/r12**3 + G*m3*(y3-y1)/r13**3
    az1 = G*m2*(z2-z1)/r12**3 + G*m3*(z3-z1)/r13**3

    ax2 = G*m1*(x1-x2)/r12**3 + G*m3*(x3-x2)/r23**3
    ay2 = G*m1*(y1-y2)/r12**3 + G*m3*(y3-y2)/r23**3
    az2 = G*m1*(z1-z2)/r12**3 + G*m3*(z3-z2)/r23**3

    ax3 = G*m1*(x1-x3)/r13**3 + G*m2*(x2-x3)/r23**3
    ay3 = G*m1*(y1-y3)/r13**3 + G*m2*(y2-y3)/r23**3
    az3 = G*m1*(z1-z3)/r13**3 + G*m2*(z2-z3)/r23**3

    return np.array([
        vx1,vy1,vz1, ax1,ay1,az1,
        vx2,vy2,vz2, ax2,ay2,az2,
        vx3,vy3,vz3, ax3,ay3,az3
    ], dtype=np.float64)

def energy(s):
    x1,y1,z1,vx1,vy1,vz1, x2,y2,z2,vx2,vy2,vz2, x3,y3,z3,vx3,vy3,vz3 = s
    v2_1 = vx1*vx1 + vy1*vy1 + vz1*vz1
    v2_2 = vx2*vx2 + vy2*vy2 + vz2*vz2
    v2_3 = vx3*vx3 + vy3*vy3 + vz3*vz3
    T = 0.5*(m1*v2_1 + m2*v2_2 + m3*v2_3)
    r12 = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2 + EPS)
    r13 = np.sqrt((x3-x1)**2 + (y3-y1)**2 + (z3-z1)**2 + EPS)
    r23 = np.sqrt((x3-x2)**2 + (y3-y2)**2 + (z3-z2)**2 + EPS)
    V = -G*(m1*m2/r12 + m1*m3/r13 + m2*m3/r23)
    return T + V

# ----------------------- 초기조건 -----------------------
def initial_conditions(mode="exp1", alpha=1.0):
    if mode == "exp1":
        s = np.array([
            -1.0,  0.0,  0.0,   0.00,  0.35*alpha, 0.01,
             1.0,  0.0,  0.0,   0.00, -0.35*alpha, 0.01,
             0.0,  0.0,  0.0,   0.00,  0.00,     -0.02
        ], dtype=np.float64)
    elif mode == "exp2":
        s = np.array([
             0.5,  0.0,  0.0,   0.0,  0.50*alpha, 0.00,
             3.0,  0.0,  0.0,   0.0, -0.25*alpha, 0.00,
             7.5,  0.0,  0.0,   0.0, -0.20*alpha, 0.08
        ], dtype=np.float64)
    else:  # exp3
        s = np.array([
            -0.2,  0.3,  0.0,    0.70*alpha,  0.00, 0.00,
             0.2, -0.3,  0.0,   -0.60*alpha,  0.00, 0.00,
             0.0,  0.0,  0.0,    0.00,        0.00, 0.30*alpha
        ], dtype=np.float64)
    return s

# ----------------------- 보조: 시간 그리드 -----------------------
def _teval_linspace(t0, t1, dt):
    n = max(2, int(np.ceil((t1 - t0)/dt)) + 1)
    return np.linspace(t0, t1, n, dtype=np.float64)

# ----------------------- 시뮬레이션 & 저장 -----------------------
def simulate(s0, t_max, dt):
    t_eval = _teval_linspace(0.0, t_max, dt)
    sol = solve_ivp(rhs, (0.0, t_max), s0, method="DOP853",
                    t_eval=t_eval, rtol=1e-9, atol=1e-12)
    return sol.t, sol.y.T  # (N, 18)

def summarize_energy_drift(t, Y):
    E0 = energy(Y[0])
    Es = np.array([energy(row) for row in Y])
    drift = (Es - E0)/abs(E0)
    return {
        "drift_final": float(drift[-1]),
        "drift_maxabs": float(np.max(np.abs(drift))),
        "drift_rms": float(np.sqrt(np.mean(drift*drift)))
    }, drift

def save_csv(out_root: Path, mode, alpha, t, Y):
    out_root.mkdir(parents=True, exist_ok=True)
    cols = (["t"] + [f"{c}{i}" for i in (1,2,3) for c in ("x","y","z","vx","vy","vz")])
    df = pd.DataFrame(np.column_stack([t, Y]), columns=cols)
    fp = out_root / f"three_body3d_{mode}_a{alpha:.1f}_t{t[-1]:.1f}_dt{_fmt_dt(t[1]-t[0])}.csv"
    df.to_csv(fp, index=False)
    return fp

def plot_trajectory(fig_dir: Path, mode, alpha, t, Y):
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")
    x1,y1,z1 = Y[:,0], Y[:,1], Y[:,2]
    x2,y2,z2 = Y[:,6], Y[:,7], Y[:,8]
    x3,y3,z3 = Y[:,12],Y[:,13],Y[:,14]
    ax.plot(x1,y1,z1,label="Body 1")
    ax.plot(x2,y2,z2,label="Body 2")
    ax.plot(x3,y3,z3,label="Body 3")
    ax.set_title("3D Three-Body Simulation")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    fp = fig_dir / f"three_body3d_{mode}_a{alpha:.1f}_t{t[-1]:.1f}_dt{_fmt_dt(t[1]-t[0])}.png"
    plt.savefig(fp, dpi=150); plt.close(fig)
    return fp

def plot_energy_drift(fig_dir: Path, mode, alpha, t, drift):
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8,6))
    plt.plot(t, drift, marker=".", markersize=2, linewidth=1)
    plt.title("Energy Drift"); plt.xlabel("Time"); plt.ylabel("Relative Energy Drift")
    plt.tight_layout()
    fp = fig_dir / f"energy_drift_{mode}_a{alpha:.1f}_t{t[-1]:.1f}_dt{_fmt_dt(t[1]-t[0])}.png"
    plt.savefig(fp, dpi=150); plt.close(fig)
    return fp

# ----------------------- Lyapunov (Benettin) -----------------------
def lyapunov_benettin(s0, t_max, dt, delta0=1e-8, tau=2.0, seed=123):
    """가장 큰 Lyapunov 지수(두 궤적 재규격화)."""
    s_ref  = s0.copy()
    rng = np.random.default_rng(seed)
    d = rng.normal(size=s0.shape)
    d /= np.linalg.norm(d)
    s_pert = s_ref + delta0 * d

    t = 0.0
    sum_log = 0.0
    while t < t_max - 1e-15:
        t_end = min(t + tau, t_max)
        t_eval = _teval_linspace(t, t_end, dt)
        sol1 = solve_ivp(rhs, (t, t_end), s_ref,  t_eval=t_eval, rtol=1e-9, atol=1e-12, method="DOP853")
        sol2 = solve_ivp(rhs, (t, t_end), s_pert, t_eval=t_eval, rtol=1e-9, atol=1e-12, method="DOP853")

        s_ref  = sol1.y[:, -1]
        s_pert = sol2.y[:, -1]

        diff = s_pert - s_ref
        dist = np.linalg.norm(diff)
        if not np.isfinite(dist) or dist < 1e-300:
            dist = 1e-300
        sum_log += np.log(dist / delta0)

        s_pert = s_ref + (delta0 / dist) * diff  # 재규격화
        t = t_end

    return sum_log / t_max

# ----------------------- 실행 래퍼 -----------------------
def run_once(mode, alpha, tmax, dt, out_root: Path,
             do_lyap=False, delta0=1e-8, tau=2.0, seed=123, lyap_reps=1):
    s0 = initial_conditions(mode, alpha)
    t, Y = simulate(s0, tmax, dt)
    drift_summary, drift_series = summarize_energy_drift(t, Y)

    data_dir = out_root / "data"
    fig_dir  = out_root / "figures"
    csv_fp   = save_csv(data_dir, mode, alpha, t, Y)
    fig_fp   = plot_trajectory(fig_dir, mode, alpha, t, Y)
    drift_fp = plot_energy_drift(fig_dir, mode, alpha, t, drift_series)

    result = {
        "mode": mode, "alpha": alpha, "tmax": tmax, "dt": dt,
        "csv": str(csv_fp.relative_to(out_root)),
        "figure": str(fig_fp.relative_to(out_root)),
        "drift": str(drift_fp.relative_to(out_root)),
        "n_steps": len(t),
        **drift_summary
    }
    if do_lyap:
        vals = []
        for k in range(lyap_reps):
            lam_k = float(lyapunov_benettin(s0, tmax, dt, delta0=delta0, tau=tau, seed=seed + k))
            vals.append(lam_k)
        result["lyapunov"] = float(np.mean(vals))
        result["lyap_std"] = float(np.std(vals))

    print(json.dumps(result, ensure_ascii=False))
    return result

# ----------------------- Lyapunov 그리드(heatmap+contour) -----------------------
def _parse_float_list(csv_like: str):
    return [float(x.strip()) for x in csv_like.split(",") if x.strip()]

def sweep_lyap_grid(mode, alpha, tmax, dt, taus, delta0s, out_root: Path):
    lam = np.zeros((len(delta0s), len(taus)), dtype=float)
    for i, d0 in enumerate(delta0s):
        for j, tau in enumerate(taus):
            lam[i, j] = float(lyapunov_benettin(initial_conditions(mode, alpha), tmax, dt, d0, tau))

    data_dir = out_root / "data"
    fig_dir  = out_root / "figures"
    data_dir.mkdir(exist_ok=True, parents=True)
    fig_dir.mkdir(exist_ok=True, parents=True)

    df = pd.DataFrame(lam, index=delta0s, columns=taus)
    csv_fp = data_dir / f"lyap_grid_{mode}_a{alpha:.1f}_t{tmax:.1f}_dt{_fmt_dt(dt)}.csv"
    df.to_csv(csv_fp, index_label="delta0")

    plt.figure(figsize=(7,5))
    plt.imshow(lam, origin="lower", aspect="auto",
               extent=[min(taus), max(taus), min(delta0s), max(delta0s)])
    plt.colorbar(label="Lyapunov")
    plt.xlabel("tau"); plt.ylabel("delta0")
    plt.title(f"Lyapunov grid: {mode}, a={alpha:.1f}, t={tmax:.1f}, dt={_fmt_dt(dt)}")
    hm_fp = fig_dir / f"lyap_heatmap_{mode}_a{alpha:.1f}_t{tmax:.1f}_dt{_fmt_dt(dt)}.png"
    plt.tight_layout(); plt.savefig(hm_fp, dpi=150); plt.close()

    T, D = np.meshgrid(taus, delta0s)
    plt.figure(figsize=(7,5))
    cs = plt.contour(T, D, lam, levels=8)
    plt.clabel(cs, inline=True, fontsize=8)
    plt.xlabel("tau"); plt.ylabel("delta0")
    plt.title(f"Lyapunov contour: {mode}, a={alpha:.1f}, t={tmax:.1f}, dt={_fmt_dt(dt)}")
    ct_fp = fig_dir / f"lyap_contour_{mode}_a{alpha:.1f}_t{tmax:.1f}_dt{_fmt_dt(dt)}.png"
    plt.tight_layout(); plt.savefig(ct_fp, dpi=150); plt.close()

    print(json.dumps({
        "mode": mode, "alpha": alpha,
        "grid_csv": str(csv_fp.relative_to(out_root)),
        "heatmap": str(hm_fp.relative_to(out_root)),
        "contour": str(ct_fp.relative_to(out_root)),
        "taus": taus, "delta0s": delta0s
    }, ensure_ascii=False))
    return csv_fp, hm_fp, ct_fp

# ----------------------- dt 스윕 -----------------------
def run_dt_scan(mode, alpha, tmax, dt_list, out_root,
                do_lyap=False, delta0=1e-8, tau=2.0,
                seed=123, lyap_reps=1):
    rows = []
    for dt in dt_list:
        res = run_once(mode, alpha, tmax, dt, out_root,
                       do_lyap=do_lyap, delta0=delta0, tau=tau,
                       seed=seed, lyap_reps=lyap_reps)
        rows.append({
            "dt": dt,
            "drift_maxabs": res["drift_maxabs"],
            "drift_rms": res["drift_rms"],
            "drift_final": res["drift_final"],
            "lyapunov": res.get("lyapunov", np.nan)
        })

    df = pd.DataFrame(rows).sort_values("dt")
    data_dir = out_root / "data"; fig_dir = out_root / "figures"
    data_dir.mkdir(exist_ok=True); fig_dir.mkdir(exist_ok=True)

    csv_fp = data_dir / f"dt_scan_{mode}_a{alpha:.1f}_t{tmax:.1f}.csv"
    df.to_csv(csv_fp, index=False)

    plt.figure(figsize=(7,5))
    plt.loglog(df["dt"], df["drift_maxabs"], marker="o")
    plt.xlabel("dt"); plt.ylabel("max |energy drift|")
    plt.title(f"Energy drift vs dt: {mode}, a={alpha:.1f}")
    drift_fp = fig_dir / f"dt_scan_drift_{mode}_a{alpha:.1f}_t{tmax:.1f}.png"
    plt.tight_layout(); plt.savefig(drift_fp, dpi=150); plt.close()

    if do_lyap:
        plt.figure(figsize=(7,5))
        plt.plot(df["dt"], df["lyapunov"], marker="o")
        plt.xlabel("dt"); plt.ylabel("Lyapunov (approx)")
        plt.title(f"Lyapunov vs dt: {mode}, a={alpha:.1f}")
        lyap_fp = fig_dir / f"dt_scan_lyap_{mode}_a{alpha:.1f}_t{tmax:.1f}.png"
        plt.tight_layout(); plt.savefig(lyap_fp, dpi=150); plt.close()
    else:
        lyap_fp = None

    print(json.dumps({
        "mode": mode, "alpha": alpha, "tmax": tmax,
        "csv": str(csv_fp.relative_to(out_root)),
        "drift_plot": str(drift_fp.relative_to(out_root)),
        "lyap_plot": (str(lyap_fp.relative_to(out_root)) if lyap_fp else None)
    }, ensure_ascii=False))
    return csv_fp

# ----------------------- main -----------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ic", choices=["exp1","exp2","exp3"], default="exp1")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--tmax", type=float, default=20.0)
    p.add_argument("--dt",   type=float, default=0.005)
    p.add_argument("--out",  default=".")
    p.add_argument("--batch", action="store_true", help="exp1~3 한 번에 실행")
    p.add_argument("--lyap",  action="store_true", help="Lyapunov 지수 계산")
    p.add_argument("--delta0", type=float, default=1e-8, help="Lyapunov 초기 교란 크기")
    p.add_argument("--tau", type=float, default=2.0, help="Lyapunov 재규격화 간격")
    p.add_argument("--seed", type=int, default=123, help="Lyapunov 난수 시드")
    p.add_argument("--lyap-reps", type=int, default=1, help="Lyapunov 반복 평균 횟수")

    # Lyapunov 그리드 스윕
    p.add_argument("--sweep", action="store_true", help="Lyapunov 그리드 스윕")
    p.add_argument("--tau-list", type=str, default="0.5,1.0,2.0")
    p.add_argument("--delta0-list", type=str, default="1e-9,1e-8,1e-7")

    # dt 스윕
    p.add_argument("--dt-scan", type=str, help="쉼표로 구분된 dt 리스트 (예: '0.01,0.005,0.0025,0.001')")

    args = p.parse_args()
    out_root = Path(args.out)
    (out_root/"data").mkdir(parents=True, exist_ok=True)
    (out_root/"figures").mkdir(parents=True, exist_ok=True)

    # ---- 스윕/스캔 우선 분기 ----
    if args.sweep:
        taus = _parse_float_list(args.tau_list)
        d0s  = _parse_float_list(args.delta0_list)
        targets = [("exp1", 0.8), ("exp2", 1.0), ("exp3", 1.2)] if args.batch else [(args.ic, args.alpha)]
        for mode, a in targets:
            sweep_lyap_grid(mode, a, args.tmax, args.dt, taus, d0s, out_root)
        return

    if args.dt_scan:
        dt_list = _parse_float_list(args.dt_scan)
        targets = [("exp1", 0.8), ("exp2", 1.0), ("exp3", 1.2)] if args.batch else [(args.ic, args.alpha)]
        for mode, a in targets:
            run_dt_scan(mode, a, args.tmax, dt_list, out_root,
                        do_lyap=args.lyap, delta0=args.delta0, tau=args.tau,
                        seed=args.seed, lyap_reps=args.lyap_reps)
        return

    # ---- 기본 실행 (단일/배치) ----
    if args.batch:
        sets = [("exp1", 0.8), ("exp2", 1.0), ("exp3", 1.2)]
        for mode, a in sets:
            run_once(mode, a, args.tmax, args.dt, out_root,
                     do_lyap=args.lyap, delta0=args.delta0, tau=args.tau,
                     seed=args.seed, lyap_reps=args.lyap_reps)
    else:
        run_once(args.ic, args.alpha, args.tmax, args.dt, out_root,
                 do_lyap=args.lyap, delta0=args.delta0, tau=args.tau,
                 seed=args.seed, lyap_reps=args.lyap_reps)

if __name__ == "__main__":
    main()