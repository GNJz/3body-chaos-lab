#!/usr/bin/env python3
import argparse, os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

# ---------- 물리 모델 ----------
def three_body_rhs(t, s, G, m1, m2, m3):
    # s = [x1,y1,z1,vx1,vy1,vz1,  x2,y2,z2,vx2,vy2,vz2,  x3,y3,z3,vx3,vy3,vz3]
    x1,y1,z1,vx1,vy1,vz1, x2,y2,z2,vx2,vy2,vz2, x3,y3,z3,vx3,vy3,vz3 = s
    eps = 1e-9
    r12 = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2 + eps)
    r13 = np.sqrt((x3-x1)**2 + (y3-y1)**2 + (z3-z1)**2 + eps)
    r23 = np.sqrt((x3-x2)**2 + (y3-y2)**2 + (z3-z2)**2 + eps)

    ax1 = G*m2*(x2-x1)/r12**3 + G*m3*(x3-x1)/r13**3
    ay1 = G*m2*(y2-y1)/r12**3 + G*m3*(y3-y1)/r13**3
    az1 = G*m2*(z2-z1)/r12**3 + G*m3*(z3-z1)/r13**3

    ax2 = G*m1*(x1-x2)/r12**3 + G*m3*(x3-x2)/r23**3
    ay2 = G*m1*(y1-y2)/r12**3 + G*m3*(y3-y2)/r23**3
    az2 = G*m1*(z1-z2)/r12**3 + G*m3*(z3-z2)/r23**3

    ax3 = G*m1*(x1-x3)/r13**3 + G*m2*(x2-x3)/r23**3
    ay3 = G*m1*(y1-y3)/r13**3 + G*m2*(y2-y3)/r23**3
    az3 = G*m1*(z1-z3)/r13**3 + G*m2*(z2-z3)/r23**3

    return [vx1,vy1,vz1, ax1,ay1,az1,
            vx2,vy2,vz2, ax2,ay2,az2,
            vx3,vy3,vz3, ax3,ay3,az3]

def total_energy(s, G, m1, m2, m3):
    x1,y1,z1,vx1,vy1,vz1, x2,y2,z2,vx2,vy2,vz2, x3,y3,z3,vx3,vy3,vz3 = s
    r12 = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    r13 = np.sqrt((x3-x1)**2 + (y3-y1)**2 + (z3-z1)**2)
    r23 = np.sqrt((x3-x2)**2 + (y3-y2)**2 + (z3-z2)**2)
    K = 0.5*m1*(vx1**2+vy1**2+vz1**2) + 0.5*m2*(vx2**2+vy2**2+vz2**2) + 0.5*m3*(vx3**2+vy3**2+vz3**2)
    U = -G*m1*m2/r12 - G*m1*m3/r13 - G*m2*m3/r23
    return K+U

# ---------- 초기 조건 ----------
def make_ic(mode="exp1", alpha=1.0):
    # 중심 1개(정지), 좌우 2개가 반대 방향 속도로 공전 시작
    if mode == "exp1":      # 비교적 안정
        s = [0,0,0, 0,0,0,
             1,0,0, 0, 0.6, 0.1,
            -1,0,0, 0,-0.6,-0.1]
    elif mode == "exp2":    # 약한 혼돈
        s = [0,0,0, 0,0,0,
             1,0,0, 0, 0.8, 0.2,
            -1,0,0, 0,-0.5,-0.05]
    elif mode == "exp3":    # 강한 혼돈
        s = [0,0,0, 0,0,0,
             1,0,0, 0, 1.0, 0.4,
            -1,0,0, 0,-0.2, 0.0]
    else:
        raise ValueError("ic mode must be exp1|exp2|exp3")
    # 속도만 스케일
    s[3:6]   = [alpha*v for v in s[3:6]]
    s[9:12]  = [alpha*v for v in s[9:12]]
    s[15:18] = [alpha*v for v in s[15:18]]
    return np.array(s, dtype=float)

# ---------- 실행 ----------
def run(ic_mode, alpha, t_max, dt, out_root):
    G, m1, m2, m3 = 1.0, 1.0, 1.0, 1.0
    s0 = make_ic(ic_mode, alpha)

    t_eval = np.arange(0.0, t_max + 1e-12, dt)
    sol = solve_ivp(
        fun=lambda t, s: three_body_rhs(t, s, G, m1, m2, m3),
        t_span=(0.0, t_max),
        y0=s0,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-9, atol=1e-12,
    )

    # 데이터프레임 저장
    df = pd.DataFrame({
        "t": sol.t,
        "x1": sol.y[0], "y1": sol.y[1], "z1": sol.y[2],
        "x2": sol.y[6], "y2": sol.y[7], "z2": sol.y[8],
        "x3": sol.y[12], "y3": sol.y[13], "z3": sol.y[14],
    })
    csv_path = os.path.join(out_root, "data", f"threebody3d_{ic_mode}_a{alpha}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    # 3D 플롯
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(df["x1"], df["y1"], df["z1"], label="Body 1")
    ax.plot(df["x2"], df["y2"], df["z2"], label="Body 2")
    ax.plot(df["x3"], df["y3"], df["z3"], label="Body 3")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"3D Three-Body (ic={ic_mode}, α={alpha})")
    ax.legend()
    fig_path = os.path.join(out_root, "figures", f"threebody3d_{ic_mode}_a{alpha}.png")
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    # 에너지 드리프트(신뢰도 체크)
    E0 = total_energy(sol.y[:,0], G, m1, m2, m3)
    E = np.array([ total_energy(sol.y[:,i], G, m1, m2, m3) for i in range(sol.y.shape[1]) ])
    drift = (E - E0)/max(1e-12, abs(E0))
    plt.figure(figsize=(7,4))
    plt.plot(sol.t, drift)
    plt.xlabel("Time"); plt.ylabel("Relative Energy Drift")
    plt.title("Total Energy Drift (lower is better)")
    drift_path = os.path.join(out_root, "figures", f"energy_drift_{ic_mode}_a{alpha}.png")
    plt.savefig(drift_path, dpi=160)
    plt.close()

    print(f"[OK] CSV  : {csv_path}")
    print(f"[OK] FIG  : {fig_path}")
    print(f"[OK] DRIFT: {drift_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ic", choices=["exp1","exp2","exp3"], default="exp1")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--tmax", type=float, default=10.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--out", default=".")
    args = p.parse_args()
    run(args.ic, args.alpha, args.tmax, args.dt, args.out)