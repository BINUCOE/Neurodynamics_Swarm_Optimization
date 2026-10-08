# -*- coding: utf-8 -*-
"""方案 A：踢事件栅格图（spike raster）。
数据：e4_obs_*.npz 的 ev_gen / ev_pid（零新实验）。
选 run 规则（防挑图）：3 个种子中取最终误差为中位者。
布局：每函数一图 = 左右两面板（AD-PSO | AD-SCA），共享横轴；
     每面板上方对齐触发数边际直方图（bin=25 代）。
用法：python3 fig_raster.py            # 只出 F03/F05/F11 × D10/D20（正文候选）
      python3 fig_raster.py --all      # 全 12 函数 × 2 维度（附录图谱）
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
OUT = ROOT.parent / "结果_E2E3E4" / "raster"
E4DIR = HERE / "e4_results"
ALGOS = [("ADPSO", "AD-PSO"), ("ADSCA", "AD-SCA")]
MAIN_FUNCS = [3, 5, 11]   # F3：SCA 通道承重；F5：PSO 基质承重；F11：两侧均承重


def pick_run(algo: str, fid: int, dim: int) -> int:
    """3 个种子中取 error 中位者（固定规则，caption 需声明）。"""
    df = pd.read_csv(E4DIR / "e4_summary.csv", encoding="utf-8")
    sub = df[(df.algo == algo) & (df.func_id == fid) & (df.dim == dim)].sort_values("error")
    return int(sub.iloc[len(sub) // 2].run_id) if len(sub) % 2 == 0 else int(sub.iloc[1].run_id)


def load_events(algo: str, fid: int, dim: int, run: int):
    p = E4DIR / f"e4_obs_{algo}_F{fid:02d}_D{dim}_r{run}.npz"
    d = np.load(p)
    return d["ev_gen"], d["ev_pid"], int(d["gen"].max())


def draw(fid: int, dim: int) -> pathlib.Path:
    # 版式（v2）：上 = 全程压缩背景条（窄），下 = 纪元缩放窗（主图）。
    # 图内不放任何说明文字，只有轴标签与列标题。
    fig = plt.figure(figsize=(11, 5.2))
    gs = GridSpec(2, 2, height_ratios=[1, 4], hspace=0.10, wspace=0.10,
                  left=0.06, right=0.99, top=0.90, bottom=0.09)
    gens_max = 0
    data = {}
    for short, _ in ALGOS:
        r = pick_run(short, fid, dim)
        eg, ep, gmax = load_events(short, fid, dim, r)
        data[short] = (eg, ep, r)
        gens_max = max(gens_max, gmax)
    zoom_c = gens_max // 2          # 缩放窗：以中点踢纪元为中心 ±125 代
    z0, z1 = zoom_c - 125, zoom_c + 125
    for col, (short, pretty) in enumerate(ALGOS):
        eg, ep, r = data[short]
        ax_s = fig.add_subplot(gs[0, col])   # 全程背景条
        ax_z = fig.add_subplot(gs[1, col])   # 缩放主图
        ax_s.scatter(eg, np.zeros_like(eg), s=1, c="#888888", marker="|",
                     lw=0.3, rasterized=True)
        ax_s.set_xlim(0, gens_max)
        ax_s.set_ylim(-1, 1)
        ax_s.set_yticks([])
        ax_s.set_xlabel("generation (full run)", fontsize=8)
        ax_s.tick_params(labelsize=7)
        ax_s.set_title(f"{pretty}  (run r{r})", fontsize=10)
        m = (eg >= z0) & (eg <= z1)
        ax_z.scatter(eg[m], ep[m], s=14, c="#222222", marker="|", lw=1.0)
        ax_z.set_xlim(z0, z1)
        ax_z.set_ylim(-1, 50)
        ax_z.set_ylabel("particle (ring index)", fontsize=9)
        ax_z.set_xlabel(f"generation (zoom {z0}–{z1})", fontsize=9)
        ax_z.tick_params(labelsize=8)
        ax_z.grid(axis="x", alpha=0.15)
    fig.suptitle(f"CEC2022 F{fid}, D={dim}", fontsize=11)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"fig_raster_F{fid:02d}_D{dim}.png"
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
