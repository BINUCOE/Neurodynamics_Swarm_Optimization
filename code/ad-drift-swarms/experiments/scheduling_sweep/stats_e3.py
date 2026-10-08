#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stats_e3.py — E3 调度扫描统计与出图（实验配置规范 v3 §七）

产出（--out 目录）：
  tsweep_doseresponse.csv      每 (algo, func, dim, T) 的中位误差与对 T=100 的对数比
  fig_tsweep_aggregate.png     主图：剂量-响应聚合曲线（两算法，24 数据集 IQR）
  fig_tsweep_panels.png        分面图：12 函数 × 2 维度逐格曲线
  tsweep_contrasts.csv         逐数据集 Wilcoxon 对照（T25 vs T100，T200 vs T100）

口径：error 中位数（EPS=1e-10 封底）；T=100 格子取自 E1（v1.8 / ADSCAv2.1），
同种子公式逐 run 配对；Δ_T = log10(med_T / med_100)，>0 表示该档节律有害。
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

EPS = 1e-10
_HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve_csv(path, what):
    """归一化并校验输入 CSV：折叠 '..' 段规避 Windows MAX_PATH 字面超长误判；
    不存在时给出清晰报错，不再抛原始 Errno 2 长栈。"""
    p = os.path.normpath(os.path.abspath(path))
    if not os.path.isfile(p):
        raise SystemExit(
            "[{}] 找不到{}：\n  尝试路径：{}\n  请检查文件是否存在，或用命令行参数指定。".format(os.path.basename(__file__),
                what, p))
    return p


def _norm_dir(path):
    """输出目录归一化（同样折叠 '..'）。"""
    return os.path.normpath(os.path.abspath(path))


def _find_repo():
    """向上定位仓库根（同时含 core/ 与 experiments/ 的目录），与脚本所在深度和仓库外层命名无关。"""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        for cand in (d, os.path.join(d, "ad-drift-swarms")):
            if os.path.isdir(os.path.join(cand, "core")) and os.path.isdir(os.path.join(cand, "experiments")):
                return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise SystemExit("找不到仓库根目录（需同时包含 core/ 与 experiments/），请把脚本放回仓库内任意位置运行。")


REPO = _find_repo()


def _find_out():
    """结果目录优先仓库根同级，其次仓库根内部；都不存在则取仓库根内部（调用方自建）。"""
    for cand in (os.path.join(REPO, "结果_E2E3E4"), os.path.join(os.path.dirname(REPO), "结果_E2E3E4")):
        if os.path.isdir(cand):
            return cand
    return os.path.join(REPO, "结果_E2E3E4")


OUT = _find_out()

E1_NAME = {"ADPSO": "v1.8", "ADSCA": "ADSCAv2.1"}
T_ALL = [25, 50, 100, 200]


def load_data(e3_csv, e1_csv):
    e3 = pd.read_csv(_resolve_csv(e3_csv, "E3 扫描原始成绩（e3_tsweep_raw.csv）"))
    e1 = pd.read_csv(_resolve_csv(e1_csv, "E1 基准原始成绩（cec2022_raw_results.csv）"))
    frames = [e3]
    for short, e1name in E1_NAME.items():
        sub = e1[e1.algo == e1name].copy()
        sub["algo"] = short
        sub["T"] = 100
        frames.append(sub)
    df = pd.concat(frames, ignore_index=True)
    df["algo"] = df["algo"].str.replace(r"_T\d+$", "", regex=True)
    df["error"] = df["error"].clip(lower=EPS)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e3", default=os.path.join(_HERE, "e3_results", "e3_tsweep_raw.csv"))
    ap.add_argument("--e1", default=os.path.join(_HERE, os.pardir, "benchmark_cec2022", "phase4_results",
        "cec2022_raw_results.csv"))
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    args.out = _norm_dir(args.out)
    os.makedirs(args.out, exist_ok=True)
    
    from scipy.stats import wilcoxon
    df = load_data(args.e3, args.e1)
    
    # 每格中位数与对数比
    med = (df.groupby(["algo", "func_id", "dim", "T"]).error.median().reset_index().rename(columns={"error": "med"}))
    ref = med[med["T"] == 100].rename(columns={"med": "med100", "T": "T0"})
    med = med.merge(ref[["algo", "func_id", "dim", "med100"]], on=["algo", "func_id", "dim"])
    med["delta"] = np.log10(med.med / med.med100)
    med.to_csv(os.path.join(args.out, "tsweep_doseresponse.csv"), index=False, encoding="utf-8-sig")
    
    # 逐数据集配对 Wilcoxon：T25 vs T100、T200 vs T100
    rows = []
    for (algo, fid, d), g in df.groupby(["algo", "func_id", "dim"]):
        for T in (25, 200):
            a = g[g["T"] == T].sort_values("run_id").error.values
            b = g[g["T"] == 100].sort_values("run_id").error.values
            if len(a) == 0 or len(b) == 0:
                continue
            try:
                p = wilcoxon(a, b).pvalue
            except ValueError:
                p = float("nan")
            rows.append(dict(algo=algo, func_id=fid, dim=d, contrast="T{}_vs_T100".format(T), med_T=float(np.median(a)),
                             med_100=float(np.median(b)), delta=float(np.log10(np.median(a) / np.median(b))),
                             wilcoxon_p=p))
    con = pd.DataFrame(rows)
    con.to_csv(os.path.join(args.out, "tsweep_contrasts.csv"), index=False, encoding="utf-8-sig")
    
    # 主图：逐数据集散点 + 中位线（聚合计量会抵消符号，改用逐点呈现）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), sharey=True)
    Ts = [25, 50, 200]
    for ax, algo, color in [(axes[0], "ADPSO", "#2166ac"), (axes[1], "ADSCA", "#b2182b")]:
        for T in T_ALL:
            v = med[(med.algo == algo) & (med["T"] == T)].delta.values
            x = T * 10.0 ** np.random.uniform(-0.015, 0.015, len(v))
            ax.scatter(x, v, s=10, color=color, alpha=0.45, zorder=2)
            if len(v):
                ax.plot(T, np.median(v), marker="_", ms=18, mew=2.5, color="black", zorder=3)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xticks(T_ALL)
        ax.set_xticklabels([str(t) for t in T_ALL])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(algo.replace("AD", "AD-"))
        ax.set_xlabel("channel period T")
    axes[0].set_ylabel("log10(median error / median error at T=100)")
    fig.suptitle("Dose-response of channel period: each dot = one dataset (24 per panel)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(args.out, "fig_tsweep_aggregate.png"), dpi=200)
    plt.close(fig)
    
    # 分面图：12 函数 × 2 维度
    fig, axes = plt.subplots(3, 4, figsize=(11, 7.5), sharex=True)
    for idx, fid in enumerate(range(1, 13)):
        ax = axes[idx // 4][idx % 4]
        for algo, color in [("ADPSO", "#2166ac"), ("ADSCA", "#b2182b")]:
            for d, ls in [(10, "-"), (20, "--")]:
                sub = med[(med.algo == algo) & (med.func_id == fid) & (med.dim == d)].set_index("T").reindex(T_ALL)
                ax.plot(T_ALL, sub.delta.values, ls, color=color, lw=1.2, marker=".", markersize=4)
        ax.axhline(0, color="gray", lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_xticks(T_ALL)
        ax.set_xticklabels(["25", "50", "100", "200"], fontsize=7)
        ax.set_title("F{}".format(fid), fontsize=9)
    fig.text(0.5, 0.02, "channel period T (log scale)", ha="center")
    fig.text(0.02, 0.5, "log10(median error ratio vs T=100)", va="center", rotation="vertical")
    handles = [plt.Line2D([], [], color="#2166ac", label="AD-PSO"), plt.Line2D([], [], color="#b2182b", label="AD-SCA"),
               plt.Line2D([], [], color="k", ls="-", label="D=10"),
               plt.Line2D([], [], color="k", ls="--", label="D=20")]
    fig.legend(handles=handles, loc="lower right", fontsize=8, bbox_to_anchor=(0.98, 0.06))
    fig.tight_layout(rect=[0.03, 0.04, 1, 1])
    fig.savefig(os.path.join(args.out, "fig_tsweep_panels.png"), dpi=200)
    plt.close(fig)
    
    # 控制台摘要：反向预言的逐数据集清点
    print("=== 调度律反向清点（24 数据集） ===")
    for algo in ("ADPSO", "ADSCA"):
        g25 = con[(con.algo == algo) & (con.contrast == "T25_vs_T100")]
        g200 = con[(con.algo == algo) & (con.contrast == "T200_vs_T100")]
        hurt25 = ((g25.delta > 0) & (g25.wilcoxon_p < 0.05)).sum()
        help25 = ((g25.delta < 0) & (g25.wilcoxon_p < 0.05)).sum()
        hurt200 = ((g200.delta > 0) & (g200.wilcoxon_p < 0.05)).sum()
        help200 = ((g200.delta < 0) & (g200.wilcoxon_p < 0.05)).sum()
        print("{}: T25 显著有害 {}/24 显著有利 {}/24 | T200 显著有害 {}/24 显著有利 {}/24".format(algo, hurt25, help25,
            hurt200, help200))
    print("输出 ->", args.out)


if __name__ == "__main__":
    main()
