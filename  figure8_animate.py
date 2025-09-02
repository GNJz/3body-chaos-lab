#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# ---------------- 기본 파라미터 ----------------
EPS = 1e-12
masses = np.array([1.0, 1.0, 1.0])
G = 1.0

# ---------------- 상태 벡터 ----------------
def unpack_state(s, N=3):
    s = np.asarray(s, float).reshape(-1)
    pos = s[:3*N].reshape(3, N, order="F")
    vel = s[3*N:6*N].reshape(3, N, order="F")
    return pos, vel

def pack_state(pos, vel):
    return np.concatenate([pos.reshape(-1, order="F"), vel.reshape(-1, order="F")])

# ---------------- 가속도 계산 ----------------
def accelerations(pos, G, masses, eps=EPS):
    dr = pos[:, None, :] - pos[:, :, None]
    r2 = np.sum(dr * dr, axis=0) + eps
    np.fill_diagonal(r2, np.inf)
    inv_r3 = r2 ** (-1.5)
    w = masses[None, :] * inv_r3
    return G * np.einsum("kij,ij->ki", dr, w)

def rhs(t, s, G=1.0, masses=masses):
    pos, vel = unpack_state(s)
    acc = accelerations(pos, G, np.asarray(masses, float))
    return pack_state(vel, acc)

# ---------------- 초기조건 ----------------
def make_ic(mode="figure8", alpha=1.0):
    if mode == "figure8":
        raw = [
            -0.97000436,  0.24308753, 0,   0.466203685,  0.43236573, 0,
             0.97000436, -0.24308753, 0,   0.466203685,  0.43236573, 0,
             0.0,         0.0,        0,  -0.93240737, -0.86473146, 0
        ]
    else:
        raise ValueError("Only figure8 mode supported")
    s = np.array(raw, float)
    pos = np.vstack([s[[0,6,12]], s[[1,7,13]], s[[2,8,14]]])
    vel = np.vstack([s[[3,9,15]], s[[4,10,16]], s[[5,11,17]]])
    vel *= alpha
    return pack_state(pos, vel)

# ---------------- 시뮬레이션 ----------------
def simulate(tmax=20, dt=0.01):
    s0 = make_ic("figure8", 1.0)
    t_eval = np.arange(0.0, tmax + dt, dt)
    sol = solve_ivp(lambda t,s: rhs(t,s,G,masses),
                    (0.0, tmax), s0, t_eval=t_eval,
                    method="DOP853", rtol=1e-9, atol=1e-12)
    pos, _ = unpack_state(sol.y, N=3)
    return sol.t, pos.reshape(3, 3, -1)

# ---------------- 애니메이션 ----------------
def make_animation():
    t, pos = simulate(tmax=12, dt=0.01)

    # 좌표 분리
    x, y, z = pos

    # ---------- 3D 애니메이션 ----------
    fig3d = plt.figure(figsize=(7,6))
    ax3d = fig3d.add_subplot(111, projection="3d")
    lines3d = [ax3d.plot([], [], [], lw=2)[0] for _ in range(3)]

    ax3d.set_xlim(-1.5, 1.5)
    ax3d.set_ylim(-1.5, 1.5)
    ax3d.set_zlim(-0.1, 0.1)
    ax3d.set_title("3D Three-Body Figure-8")
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")

    def update3d(frame):
        for i, line in enumerate(lines3d):
            line.set_data(x[i,:frame], y[i,:frame])
            line.set_3d_properties(z[i,:frame])
        return lines3d

    ani3d = FuncAnimation(fig3d, update3d, frames=len(t), interval=20, blit=True)
    ani3d.save("figure8_3d.mp4", fps=30)
    plt.close(fig3d)

    # ---------- XY 애니메이션 ----------
    figxy, axxy = plt.subplots(figsize=(6,6))
    linesxy = [axxy.plot([], [], lw=2)[0] for _ in range(3)]

    axxy.set_xlim(-1.2, 1.2)
    axxy.set_ylim(-1.2, 1.2)
    axxy.set_title("XY Projection of Figure-8")
    axxy.set_xlabel("X"); axxy.set_ylabel("Y")

    def updatexy(frame):
        for i, line in enumerate(linesxy):
            line.set_data(x[i,:frame], y[i,:frame])
        return linesxy

    anixy = FuncAnimation(figxy, updatexy, frames=len(t), interval=20, blit=True)
    anixy.save("figure8_xy.mp4", fps=30)
    plt.close(figxy)

    print("[OK] figure8_3d.mp4 저장 완료 ✅")
    print("[OK] figure8_xy.mp4 저장 완료 ✅")

if __name__ == "__main__":
    make_animation()