#!/usr/bin/env python3
# animate_threebody.py — create 3D & XY animations (GIF/MP4) from CSV

import os
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # no GUI backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# -------------------- IO --------------------
def load_csv(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    t  = df["t"].to_numpy()
    P1 = df[["x1", "y1", "z1"]].to_numpy()
    P2 = df[["x2", "y2", "z2"]].to_numpy()
    P3 = df[["x3", "y3", "z3"]].to_numpy()
    return t, P1, P2, P3


# -------------------- Anim builders (return ani & fig) --------------------
def make_3d_animation(t, P1, P2, P3, fps=60, step=2, trail=300):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(7.5, 6.5), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("3D Three-Body Trajectory", pad=14)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    allP = np.vstack([P1, P2, P3])
    mins = allP.min(axis=0)
    maxs = allP.max(axis=0)
    pad  = 0.1 * (maxs - mins + 1e-9)
    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
    ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])

    l1, = ax.plot([], [], [], lw=2, color="#1f77b4")
    l2, = ax.plot([], [], [], lw=2, color="#ff7f0e")
    l3, = ax.plot([], [], [], lw=2, color="#2ca02c")
    p1 = ax.scatter([], [], [], s=30, color="#1f77b4", label="Body 1")
    p2 = ax.scatter([], [], [], s=30, color="#ff7f0e", label="Body 2")
    p3 = ax.scatter([], [], [], s=30, color="#2ca02c", label="Body 3")
    tt = ax.text2D(0.02, 0.93, "", transform=ax.transAxes)
    ax.legend(loc="upper right")

    idxs = np.arange(0, len(t), step)

    def update(k):
        i = idxs[k]
        j0 = max(0, i - trail)
        l1.set_data(P1[j0:i, 0], P1[j0:i, 1]); l1.set_3d_properties(P1[j0:i, 2])
        l2.set_data(P2[j0:i, 0], P2[j0:i, 1]); l2.set_3d_properties(P2[j0:i, 2])
        l3.set_data(P3[j0:i, 0], P3[j0:i, 1]); l3.set_3d_properties(P3[j0:i, 2])
        p1._offsets3d = ([P1[i, 0]], [P1[i, 1]], [P1[i, 2]])
        p2._offsets3d = ([P2[i, 0]], [P2[i, 1]], [P2[i, 2]])
        p3._offsets3d = ([P3[i, 0]], [P3[i, 1]], [P3[i, 2]])
        tt.set_text(f"t = {t[i]:.2f}")
        return l1, l2, l3, p1, p2, p3, tt

    ani = FuncAnimation(fig, update, frames=len(idxs), interval=1000 / fps, blit=False)
    return ani, fig


def make_xy_animation(t, P1, P2, P3, fps=60, step=2, trail=300):
    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=140)
    ax.set_title("XY Projection — Trajectories")
    ax.set_xlabel("X"); ax.set_ylabel("Y")

    allXY = np.vstack([P1[:, :2], P2[:, :2], P3[:, :2]])
    mins = allXY.min(axis=0)
    maxs = allXY.max(axis=0)
    pad  = 0.1 * (maxs - mins + 1e-9)
    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
    ax.set_aspect("equal", adjustable="box")

    l1, = ax.plot([], [], lw=2, color="#1f77b4")
    l2, = ax.plot([], [], lw=2, color="#ff7f0e")
    l3, = ax.plot([], [], lw=2, color="#2ca02c")
    p1, = ax.plot([], [], "o", ms=5, color="#1f77b4", label="Body 1")
    p2, = ax.plot([], [], "o", ms=5, color="#ff7f0e", label="Body 2")
    p3, = ax.plot([], [], "o", ms=5, color="#2ca02c", label="Body 3")
    tt = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    ax.legend(loc="upper right")

    idxs = np.arange(0, len(t), step)

    def update(k):
        i = idxs[k]
        j0 = max(0, i - trail)
        l1.set_data(P1[j0:i, 0], P1[j0:i, 1])
        l2.set_data(P2[j0:i, 0], P2[j0:i, 1])
        l3.set_data(P3[j0:i, 0], P3[j0:i, 1])
        p1.set_data(P1[i, 0], P1[i, 1])
        p2.set_data(P2[i, 0], P2[i, 1])
        p3.set_data(P3[i, 0], P3[i, 1])
        tt.set_text(f"t = {t[i]:.2f}")
        return l1, l2, l3, p1, p2, p3, tt

    ani = FuncAnimation(fig, update, frames=len(idxs), interval=1000 / fps, blit=False)
    return ani, fig


# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",  choices=["exp1", "exp2", "exp3", "figure8"], default="figure8")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--data",  default="./data")
    ap.add_argument("--out",   default=".")
    ap.add_argument("--fps",   type=int, default=60)   # 빠른 재생 기본값
    ap.add_argument("--step",  type=int, default=2)    # 프레임 샘플링 간격
    ap.add_argument("--trail", type=int, default=300)  # 꼬리 길이(표시 프레임 수)
    args = ap.parse_args()

    csv = os.path.join(args.data, f"threebody3d_{args.mode}_a{args.alpha}.csv")
    t, P1, P2, P3 = load_csv(csv)
    os.makedirs(args.out, exist_ok=True)

    # build animations
    ani3d, fig3d = make_3d_animation(t, P1, P2, P3, fps=args.fps, step=args.step, trail=args.trail)
    anixy, figxy = make_xy_animation(t, P1, P2, P3, fps=args.fps, step=args.step, trail=args.trail)

    # file paths
    path_3d_mp4 = os.path.join(args.out, f"{args.mode}_3d.mp4")
    path_xy_mp4 = os.path.join(args.out, f"{args.mode}_xy.mp4")
    path_3d_gif = os.path.join(args.out, f"{args.mode}_3d.gif")
    path_xy_gif = os.path.join(args.out, f"{args.mode}_xy.gif")

    # save MP4 (ffmpeg) — 안전하게 try/except
    try:
        ani3d.save(path_3d_mp4, writer="ffmpeg", fps=args.fps)
        anixy.save(path_xy_mp4, writer="ffmpeg", fps=args.fps)
        print(f"[OK] 3D MP4 : {path_3d_mp4}")
        print(f"[OK] XY MP4 : {path_xy_mp4}")
    except Exception as e:
        print(f"[WARN] MP4 저장 실패 (ffmpeg?): {e}")

    # save GIF (Pillow)
    ani3d.save(path_3d_gif, writer=PillowWriter(fps=args.fps))
    anixy.save(path_xy_gif, writer=PillowWriter(fps=args.fps))
    print(f"[OK] 3D GIF : {path_3d_gif}")
    print(f"[OK] XY GIF : {path_xy_gif}")

    plt.close(fig3d); plt.close(figxy)


if __name__ == "__main__":
    main()