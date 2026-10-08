#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stats_e2.py — E2 消融链统计与出图（实验配置规范 v3 §四/§七）

产出（--out 目录）：
  table_ablation_adpso.md / .csv   AD-PSO 链大表（24 数据集 × 4 节点，mean±std）
  table_ablation_adsca.md / .csv   AD-SCA 链大表（24 数据集 × 5 节点）
  chain_steps.csv                  步级 Δ 明细（中位数比 + 配对 Wilcoxon）
  fig_ablation_heatmap.png         联合热力图（行=24 数据集，列=同源机制两身对齐）
  fig_ablation_heatmap_D10/D20.png 分维度两张（附录）

口径：error = |best − f_global|；EPS=1e-10 封底；步级 Δ = log10(med_子/med_父)；
Wilcoxon 配对（同种子 30 runs）。端点并表自 E1：P0=v1.8, P3=PSO, S0=ADSCAv2.1, S4=SCA。
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
        raise SystemExit("[{}] 找不到{}：\n  尝试路径：{}\n  "
                         "请检查文件是否存在，或用命令行参数指定。".format(os.path.basename(__file__), what, p))
    return p


def _norm_dir(path):
    """输出目录归一化（同样折叠 '..'）。"""
    return os.path.normpath(os.path.abspath(path))


CHAINS = {"AD-PSO": [("P0", "v1.8"), ("P1", None), ("P2", None), ("P3", "PSO")],
          "AD-SCA": [("S0", "ADSCAv2.1"), ("S1", None), ("S2", None), ("S3", None), ("S4", "SCA")], }
STEP_NAMES = {"AD-PSO": ["−K自适应", "−环+WTA", "−通道"], "AD-SCA": ["−K自适应", "−通道", "−环+WTA", "−核层修正"], }

# 定价步定义：AD-PSO 侧 −通道 采用干净单拆步 P1→P0r（逆序节点，与 SCA 侧 S1→S2 同构对称）；
# 复合端点步（P2→P3，含标准参数集与方向发生器惯例）仅作审计，写入 chain_steps_compound.csv。
STEP_PAIRS = {  # 注：load_data 已按 CHAINS 将 E1 端点重命名为链内节点名
    # （v1.8→P0，ADSCAv2.1→S0，PSO→P3，SCA→S4），故此处用节点名。
    "AD-PSO": [("P0", "P1"), ("P1", "P2"), ("P1", "P0r")],
    "AD-SCA": [("S0", "S1"), ("S1", "S2"), ("S2", "S3"), ("S3", "S4")], }

NODE_LABELS = {"P0": "AD-PSO（全配置）", "P1": "−K自适应", "P2": "−K−环", "P3": "原始 PSO", "S0": "AD-SCA（全配置）",
               "S1": "−K自适应", "S2": "−K−通道", "S3": "−K−通道−环", "S4": "原始 SCA", }


def load_data(e2_csv, e1_csv):
    e2 = pd.read_csv(_resolve_csv(e2_csv, "E2 消融链原始成绩（e2_chain_raw.csv）"))
    e1 = pd.read_csv(_resolve_csv(e1_csv, "E1 基准原始成绩（cec2022_raw_results.csv）"))
    frames = [e2]
    for chain in CHAINS.values():
        for node, e1name in chain:
            if e1name:
                sub = e1[e1.algo == e1name].copy()
                sub["algo"] = node
                frames.append(sub)
    df = pd.concat(frames, ignore_index=True)
    df["error"] = df["error"].clip(lower=EPS)
    return df


def fmt_cell(mean, std):
    if mean <= 1.01 * EPS:
        return "0†"
    
    def sci(v):
        if v <= 1.01 * EPS:
            return "0†"
        e = int(np.floor(np.log10(abs(v))))
        return "{:.2f}e{:+d}".format(v / 10 ** e, e)
    
    if max(mean, std) >= 1e3 or mean < 1e-2:
        return "{}±{}".format(sci(mean), sci(std))
    return "{:.3g}±{:.2g}".format(mean, std)


def _safe_to_markdown(df, path, index=False, disable_numparse=True, **_kw):
    """写 markdown 表格。
    """
    try:
        cols = list(df.columns)
        lines = []
        if index:
            cols = [df.index.name or ""] + cols
        lines.append("| " + " | ".join(str(c) for c in cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for idx, row in df.iterrows():
            vals = ([idx] if index else []) + [row[c] for c in df.columns]
            lines.append("| " + " | ".join(str(v) for v in vals) + " |")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except OSError as e:
        print("[stats_e2] markdown 写出失败（CSV 不受影响）：", path, e)


def build_tables(df, out):
    tables = {}
    for algo, chain in CHAINS.items():
        rows = []
        for fid in range(1, 13):
            for d in (10, 20):
                row = {"func": "F{}".format(fid), "D": d}
                for node, _ in chain:
                    sub = df[(df.algo == node) & (df.func_id == fid) & (df.dim == d)]
                    row[node] = fmt_cell(sub.error.mean(), sub.error.std())
                rows.append(row)
        t = pd.DataFrame(rows)
        cols = [n for n, _ in chain]
        t = t[["func", "D"] + cols].rename(columns=NODE_LABELS)
        tag = "adpso" if algo == "AD-PSO" else "adsca"
        _safe_to_markdown(t, os.path.join(out, "table_ablation_{}.md".format(tag)), index=False, tablefmt="pipe",
                          disable_numparse=True)
        t.to_csv(os.path.join(out, "table_ablation_{}.csv".format(tag)), index=False, encoding="utf-8-sig")
        tables[algo] = t
    return tables


def build_steps(df, out):
    from scipy.stats import wilcoxon
    
    def _pair_rows(algo, step_name, parent, child):
        rs = []
        for fid in range(1, 13):
            for d in (10, 20):
                ep = df[(df.algo == parent) & (df.func_id == fid) & (df.dim == d)].sort_values("run_id").error.values
                ec = df[(df.algo == child) & (df.func_id == fid) & (df.dim == d)].sort_values("run_id").error.values
                mp, mc = np.median(ep), np.median(ec)
                delta = np.log10(mc / mp)
                try:
                    p = wilcoxon(ep, ec).pvalue
                except ValueError:
                    p = float("nan")
                rs.append(dict(algo=algo, step=step_name, parent=parent, child=child, func_id=fid, dim=d, med_parent=mp,
                               med_child=mc, delta=delta, wilcoxon_p=p, floored=(mp == EPS or mc == EPS)))
        return rs
    
    rows = []
    for algo, pairs in STEP_PAIRS.items():
        for si, (parent, child) in enumerate(pairs):
            rows += _pair_rows(algo, STEP_NAMES[algo][si], parent, child)
    steps = pd.DataFrame(rows)
    steps.to_csv(os.path.join(out, "chain_steps.csv"), index=False, encoding="utf-8-sig")
    # 审计行：AD-PSO 侧复合端点步（不入主定价、不入热力图）
    comp = pd.DataFrame(_pair_rows("AD-PSO", "−通道（端点复合）", "P2", "P3"))
    comp.to_csv(os.path.join(out, "chain_steps_compound.csv"), index=False, encoding="utf-8-sig")
    return steps


def heatmap(steps, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    # 联合图：列按同源机制两身对齐（图件标签用英文，终稿直接可用）
    EN = {"−K自适应": "-K-adaptive", "−通道": "-channel", "−环+WTA": "-ring+WTA", "−核层修正": "-kernel-fixes"}
    colspec = [("−K自适应", "AD-PSO"), ("−K自适应", "AD-SCA"), ("−通道", "AD-PSO"), ("−通道", "AD-SCA"),
               ("−环+WTA", "AD-PSO"), ("−环+WTA", "AD-SCA"), ("−核层修正", "AD-SCA")]
    datasets = [(fid, d) for fid in range(1, 13) for d in (10, 20)]
    M = np.full((len(datasets), len(colspec)), np.nan)
    for j, (sname, algo) in enumerate(colspec):
        for i, (fid, d) in enumerate(datasets):
            r = steps[(steps.algo == algo) & (steps.step == sname) & (steps.func_id == fid) & (steps.dim == d)]
            if len(r):
                M[i, j] = r.delta.iloc[0]
    vmax = np.nanmax(np.abs(M))
    vmax = min(vmax, 8.0)
    
    def draw(mat, ds_rows, fname, title):
        fig, ax = plt.subplots(figsize=(7.2, 8.5))
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(colspec)))
        ax.set_xticklabels(["{}\n{}".format(EN[s], a) for s, a in colspec], fontsize=8, rotation=0)
        ax.set_yticks(range(len(ds_rows)))
        ax.set_yticklabels(["F{} D{}".format(f, d) for f, d in ds_rows], fontsize=7)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, "{:+.1f}".format(mat[i, j]), ha="center", va="center", fontsize=6,
                            color="white" if abs(mat[i, j]) > vmax * 0.6 else "black")
        cb = fig.colorbar(im, ax=ax, shrink=0.6)
        cb.set_label("step Δ = log10(median error ratio), >0: mechanism helps")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), dpi=200)
        plt.close(fig)
    
    draw(M, datasets, "fig_ablation_heatmap.png",
         "Ablation chains: step-level deltas (both bodies aligned by mechanism)")
    for d in (10, 20):
        idx = [i for i, (_, dd) in enumerate(datasets) if dd == d]
        draw(M[idx], [datasets[i] for i in idx], "fig_ablation_heatmap_D{}.png".format(d),
             "Ablation chains, D={}".format(d))


# ===================== IDE 运行配置（按需修改） =====================
# 若运行时提示 FileNotFoundError，把下方路径改成你机器上文件的实际位置。
# None 表示使用脚本默认路径（相对于本文件所在目录，通常无需修改）。
E2_CSV = None  # e2_chain_raw.csv 路径
E1_CSV = None  # cec2022_raw_results.csv 路径（E1 基准原始成绩）
OUT_DIR = None  # 输出目录（chain_steps.csv 与热力图写入其中）


# ====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2", default=os.path.join(_HERE, "e2_results", "e2_chain_raw.csv"))
    ap.add_argument("--e1", default=os.path.join(_HERE, os.pardir, "benchmark_cec2022", "phase4_results",
                                                 "cec2022_raw_results.csv"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.e2 = _resolve_csv(args.e2, "E2 消融链原始成绩（--e2）")
    args.e1 = _resolve_csv(args.e1, "E1 基准原始成绩（--e1）")
    args.out = _norm_dir(args.out)
    os.makedirs(args.out, exist_ok=True)
    
    df = load_data(args.e2, args.e1)
    tables = build_tables(df, args.out)
    steps = build_steps(df, args.out)
    heatmap(steps, args.out)
    
    # 控制台摘要：每步的 24 数据集 Δ 中位与显著格数
    print("=== 步级摘要（24 数据集） ===")
    for (algo, sname), g in steps.groupby(["algo", "step"], sort=False):
        sig = ((g.wilcoxon_p < 0.05) & (g.delta > 0)).sum()
        sig_neg = ((g.wilcoxon_p < 0.05) & (g.delta < 0)).sum()
        print("{} {}: Δ中位 {:+.2f} | 显著有功 {}/24 | 显著有害 {}/24".format(algo, sname, g.delta.median(), sig,
                                                                              sig_neg))
    print("输出 ->", args.out)


if __name__ == "__main__":
    # IDE 一键运行入口：路径优先取上方配置块，未配置时用脚本默认路径。
    import sys
    
    argv = [sys.argv[0]]
    argv += ["--e2", E2_CSV or os.path.join(_HERE, "e2_results", "e2_chain_raw.csv")]
    argv += ["--e1",
             E1_CSV or os.path.join(_HERE, os.pardir, "benchmark_cec2022", "phase4_results", "cec2022_raw_results.csv")]
    argv += ["--out", OUT_DIR or os.path.join(_HERE, "e2_results")]
    sys.argv = argv
    main()
