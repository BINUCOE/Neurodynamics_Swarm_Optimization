# -*- coding: utf-8 -*-
"""方案 B：A 场热图（膜电位场成像）。
数据：e4_obs_*.npz 的 A_full（gens×N，instrument_e4 v2 起）+ ev_gen/ev_pid 叠加。
选 run 规则与 fig_raster 一致：3 种子中取最终误差中位者。
设计准则：两身共享色标 [0, 1]（A 的阈值语义相同，各自归一化是学术不端级错误）；
         踢事件以白色 tick 叠加（与 A 图互验：白点必落在 A 归零的行上）。
用法：python3 fig_afield.py            # F03/F05/F11 × D10/D20
      python3 fig_afield.py --all      # 全 12 函数 × 2 维度
"""
import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = ROOT.parent / "结果_E2E3E4" / "afield"
E4DIR = HERE / "e4_results"
ALGOS = [("ADPSO", "AD-PSO"), ("ADSCA", "AD-SCA")]
MAIN_FUNCS = [3, 5, 11]


def pick_run(algo: str, fid: int, dim: int) -> int:
    df = pd.read_csv(E4DIR / "e4_summary.csv", encoding="utf-8")
    sub = df[(df.algo == algo) & (df.func_id == fid) & (df.dim == dim)].sort_values("error")
    return int(sub.iloc[len(sub) // 2].run_id) if len(sub) % 2 == 0 else int(sub.iloc[1].run_id)


def load(algo: str, fid: int, dim: int, run: int):
    p = E4DIR / f"e4_obs_{algo}_F{fid:02d}_D{dim}_r{run}.npz"
    with np.load(p) as d:
        if "A_full" not in d.files:
            raise RuntimeError(f"{p.name} 缺 A_full——请重跑 instrument_e4.py（v2）")
        return d["A_full"], d["ev_gen"], d["ev_pid"]


def draw(fid: int, dim: int) -> pathlib.Path:
    fig = plt.figure(figsize=(11, 5.6))
    gs = GridSpec(2, 3, width_ratios=[1, 1, 0.025], height_ratios=[2.2, 1.4],
                  hspace=0.12, wspace=0.08, left=0.06, right=0.97, top=0.90, bottom=0.09)
    data, gens_max = {}, 0
    for short, _ in ALGOS:
        r = pick_run(short, fid, dim)
        A, eg, ep = load(short, fid, dim, r)
        data[short] = (A, eg, ep, r)
        gens_max = max(gens_max, A.shape[0])
    zoom_c = gens_max // 2
    z0, z1 = max(0, zoom_c - 125), zoom_c + 125

    im = None
    for col, (short, pretty) in enumerate(ALGOS):
        A, eg, ep, r = data[short]
        ax_f = fig.add_subplot(gs[0, col])
        ax_z = fig.add_subplot(gs[1, col])
        # 全程场热图（纵轴粒子按环索引；A 为行向量序列 → 转置为 N×gens）
        im = ax_f.imshow(A.T, aspect="auto", origin="lower", cmap="viridis",
                         vmin=0.0, vmax=1.0, extent=[1, A.shape[0], -0.5, 49.5],
                         interpolation="nearest", rasterized=True)
        ax_f.set_xlim(1, gens_max)
        ax_f.set_ylim(-0.5, 49.5)
        ax_f.set_ylabel("particle (ring index)", fontsize=9)
        ax_f.tick_params(labelbottom=False, labelsize=8)
        ax_f.set_title(f"{pretty}  (run r{r})", fontsize=10)
        # 缩放窗
        W = A[z0:z1]
        m = (eg > z0) & (eg <= z1)
        ax_z.imshow(W.T, aspect="auto", origin="lower", cmap="viridis",
                    vmin=0.0, vmax=1.0, extent=[z0 + 1, z0 + len(W), -0.5, 49.5],
                    interpolation="nearest")
        ax_z.scatter(eg[m], ep[m], s=6, c="white", marker="|", lw=0.7)
        ax_z.set_xlim(z0, z1)
        ax_z.set_ylim(-0.5, 49.5)
        ax_z.set_ylabel("zoom", fontsize=8)
        ax_z.set_xlabel(f"generation (window {z0}–{z1})", fontsize=9)
        ax_z.tick_params(labelsize=8)
    cax = fig.add_subplot(gs[:, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("accumulator A", fontsize=8)
    cb.ax.tick_params(labelsize=8)
    fig.suptitle(f"CEC2022 F{fid}, D={dim}", fontsize=11)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"fig_afield_F{fid:02d}_D{dim}.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="全 12 函数（附录图谱）")
    args = ap.parse_args()
    funcs = range(1, 13) if args.all else MAIN_FUNCS
    for fid in funcs:
        for dim in (10, 20):
            print("saved:", draw(fid, dim))
