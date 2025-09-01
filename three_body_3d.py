#!/usr/bin/env python3
# three_body_3d_opt.py — optimized, vectorized, DTG-integrated
import argparse, os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-12

# ---------------- 상태 관리 ----------------
def unpack_state(s, N=3):
    """1D 상태 벡터 → (pos, vel) 분리"""
    s = np.asarray(s, float).reshape(-1)
    if s.size != 6 * N:
        raise ValueError(f"state length must be 6N (= {6*N}), got {s.size}")
    pos = s[:3*N].reshape(3, N, order="F")
    vel = s[3*N:6*N].reshape(3, N, order="F")
    return pos, vel

def pack_state(pos, vel):
    """(pos, vel) → 1D 상태 벡터"""
    pos = np.asarray(pos, float)
    vel = np.asarray(vel, float)
    if pos.shape != vel.shape or pos.shape[0] != 3:
        raise ValueError("pos/vel must have shape (3,N)")
    N = pos.shape[1]
    return np.concatenate([pos.reshape(3*N, order="F"),
                           vel.reshape(3*N, order="F")])

# ---------------- 물리 코어 ----------------
def accelerations(pos, G, masses, eps=EPS):
    """중력 가속도 계산, 완전 벡터화"""
    dr = pos[:, None, :] - pos[:, :, None]  # (3,N,N)
    r2 = np.sum(dr * dr, axis=0) + eps
    np.fill_diagonal(r2, np.inf)
    inv_r3 = r2 ** (-1.5)
    w = masses[None, :] * inv_r3
    return G * np.einsum("kij,ij->ki", dr, w)

def rhs(t, s, G=1.0, masses=(1.0,1.0,1.0)):
    """상미분방정식 RHS"""
    pos, vel = unpack_state(s, N=3)
    acc = accelerations(pos, G, np.asarray(masses, float))
    return pack_state(vel, acc)

def total_energy(s, G=1.0, masses=(1.0,1.0,1.0)):
    """계의 전체 에너지 계산"""
    m = np.asarray(masses, float)
    pos, vel = unpack_state(s, N=3)
    v2 = np.sum(vel * vel, axis=0)
    K = 0.5 * np.sum(m * v2)
    dr = pos[:, None, :] - pos[:, :, None]
    r = np.sqrt(np.sum(dr*dr, axis=0) + EPS)
    iu = np.triu_indices(m.size, k=1)
    U = -G * np.sum(m[iu[0]] * m[iu[1]] / r[iu])
    return K + U

# ---------------- 초기조건 ----------------
def make_ic(mode="exp1", alpha=1.0):
    """IC 생성"""
    if mode == "exp1":
        raw = [0,0,0, 0,0,0, 1,0,0, 0,0.6,0.1, -1,0,0, 0,-0.6,-0.1]
    elif mode == "exp2":
        raw = [0,0,0, 0,0,0, 1,0,0, 0,0.8,0.2, -1,0,0, 0,-0.5,-0.05]
    elif mode == "exp3":
        raw = [0,0,0, 0,0,0, 1,0,0, 0,1.0,0.4, -1,0,0, 0,-0.2,0.0]
    elif mode == "figure8":
        raw = [0.970, 0, 0, -0.932, 0.864, 0, -0.970, 0, 0,
               0.932, -0.864, 0, 0, 0, 0, 0, 0, 0]
    else:
        raise ValueError("mode must be exp1|exp2|exp3|figure8")
    s = np.array(raw, float)
    pos = np.vstack([s[[0,6,12]], s[[1,7,13]], s[[2,8,14]]])
    vel = np.vstack([s[[3,9,15]], s[[4,10,16]], s[[5,11,17]]])
    vel *= alpha
    return pack_state(pos, vel)

# ---------------- 좌표 추출 ----------------
def positions_from_sol(sol, N=3):
    """sol.y에서 위치만 안전하게 추출"""
    Y = np.asarray(sol.y, float)
    rows = np.arange(N)
    x_rows = 3*rows + 0
    y_rows = 3*rows + 1
    z_rows = 3*rows + 2
    return Y[x_rows], Y[y_rows], Y[z_rows]

# ---------------- 실행 ----------------
def run(ic_mode, alpha, t_max, dt, out_root):
    masses = np.array([1.0,1.0,1.0])
    s0 = make_ic(ic_mode, alpha)
    t_eval = np.arange(0.0, t_max + 1e-12, dt)

    sol = solve_ivp(lambda t,s: rhs(t,s,1.0,masses),
                    (0.0, t_max), s0, t_eval=t_eval,
                    method="DOP853", rtol=1e-9, atol=1e-12)

    # 위치 추출
    x, y, z = positions_from_sol(sol, N=3)
    x1,y1,z1 = x[0], y[0], z[0]
    x2,y2,z2 = x[1], y[1], z[1]
    x3,y3,z3 = x[2], y[2], z[2]

    # CSV 저장
    df = pd.DataFrame({"t": sol.t,
                       "x1": x1,"y1": y1,"z1": z1,
                       "x2": x2,"y2": y2,"z2": z2,
                       "x3": x3,"y3": y3,"z3": z3})
    csv_path = os.path.join(out_root, "data", f"threebody3d_{ic_mode}_a{alpha}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    # 3D 경로 플롯
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x1,y1,z1,label="Body 1")
    ax.plot(x2,y2,z2,label="Body 2")
    ax.plot(x3,y3,z3,label="Body 3")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"3D Three-Body (ic={ic_mode}, α={alpha})")
    ax.legend()
    fig_path = os.path.join(out_root, "figures", f"threebody3d_{ic_mode}_a{alpha}.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    # 에너지 드리프트
    E0 = total_energy(sol.y[:,0], 1.0, masses)
    E  = np.array([total_energy(sol.y[:,i], 1.0, masses)
                   for i in range(sol.y.shape[1])])
    drift = (E - E0) / (abs(E0) + 1e-15)
    plt.figure(figsize=(7,4))
    plt.plot(sol.t, drift)
    plt.xlabel("Time"); plt.ylabel("Relative Energy Drift")
    plt.title("Total Energy Drift (lower is better)")
    drift_path = os.path.join(out_root, "figures", f"energy_drift_{ic_mode}_a{alpha}.png")
    os.makedirs(os.path.dirname(drift_path), exist_ok=True)
    plt.savefig(drift_path, dpi=160)
    plt.close()

    print(f"[OK] CSV  : {csv_path}")
    print(f"[OK] FIG  : {fig_path}")
    print(f"[OK] DRIFT: {drift_path}")
    return sol.t, drift

# ---------------- Lyapunov ----------------
def lyapunov_estimate(s0, rhs, tmax=20.0, dt=0.01, delta0=1e-8,
                      G=1.0, masses=(1.0,1.0,1.0)):
    rng = np.random.default_rng(0)
    v = rng.normal(size=s0.size); v /= np.linalg.norm(v)
    s1, s2 = s0.copy(), s0 + delta0 * v
    t_eval = np.arange(0.0, tmax + 1e-12, dt)
    sol1 = solve_ivp(lambda t,s: rhs(t,s,G,masses),
                     (0,tmax), s1, t_eval=t_eval, method="DOP853")
    sol2 = solve_ivp(lambda t,s: rhs(t,s,G,masses),
                     (0,tmax), s2, t_eval=t_eval, method="DOP853")
    deltas = np.linalg.norm(sol2.y - sol1.y, axis=0)
    return np.polyfit(t_eval[1:], np.log(deltas[1:] + 1e-30), 1)[0]

# ---------------- DTG ----------------
def dtg_update(V_0, alpha, beta, lambda_, b, E_t, I_t, theta_t):
    return (1 - lambda_) * theta_t + lambda_ * (b + alpha * E_t - beta * I_t)

# ---------------- 메인 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ic", choices=["exp1","exp2","exp3","figure8"], default="exp1")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--tmax", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--out", default=".")
    ap.add_argument("--lyap", action="store_true")
    args = ap.parse_args()

    t, drift = run(args.ic, args.alpha, args.tmax, args.dt, args.out)

    lam = 0.0
    if args.lyap:
        masses = np.array([1.0,1.0,1.0])
        s0 = make_ic(args.ic, args.alpha)
        lam = lyapunov_estimate(s0, rhs,
                                min(40.0, args.tmax), args.dt, G=1.0, masses=masses)
        print(f"[Lyapunov ≈] {lam:.6f}  (양수면 혼돈 경향)")

    # DTG 업데이트 (예시)
    V_0, alpha_dtg, beta, lambda_, b = 1.0, 0.7, 0.5, 0.1, 0.0
    theta_t = V_0
    for i in range(len(drift)):
        theta_t = dtg_update(V_0, alpha_dtg, beta, lambda_, b, drift[i], lam, theta_t)
    print(f"[DTG Final Theta]: {theta_t:.6f}")

if __name__ == "__main__":
    main()