#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stats_e4.py — E4 映射核验统计（实验配置规范 v3 §七）

产出（--out 目录）：
  table_verification.md / .csv   核验表：预测 ↔ 全基准实测（每算法 24 数据集 × 3 runs 聚合）
  activity_value.csv             活性×价值逐数据集明细
  fig_activity_value.png         活性×价值散点（两算法，跨 24 数据集 Spearman）

口径：E4 每格 3 runs 取中位；活性 = trig_count/gens（每代触发率）；
价值 = E2 链上 −通道步级 Δ（PSO: P2→P3；SCA: S1→S2），取自 chain_steps.csv。
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))


def _safe_to_markdown(df, path, index=False, disable_numparse=True, **_kw):
    """写 markdown 表格。

    不用 df.to_markdown：py3.7 旧版 pandas 会把 index/disable_numparse 等
    kwargs 原样透传给 tabulate，而旧版 tabulate 不认识这些参数，报
    TypeError: tabulate() got an unexpected keyword argument 'index'。
    这里手写 pipe 表格，只依赖 pandas 本身。
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

PREDICTIONS = [# (预测, 观测列, 判定函数, 读者口径期望)
    ("驻留时间 = T·A_thresh = 100 代", "dwell_p50", None, "100 代"),
    ("踢幅度 ∈ (1.0, 1.01]", "kick_mag", None, "(1.0, 1.01]"),
    ("近周期阈值过程：Fano = O(1)，无幂律尾", "fano_w100", None, "O(1)"),
    ("逃逸发生在坍缩后：踢幅/散布 ≫ 1", "kick_spread_p50", None, "≫ 1"),
    ("环上传播延迟小", "prop_delay_p50", None, "1–2 代"), ("K 与 mean(A) 反相关", "corr_K_meanA", None, "显著为负"),
    ("振幅增益调制结构性失活", "boost_hits", None, "恒 0"), ]


def agg_range(s):
    return "{:.3g}（{:.3g}–{:.3g}）".format(s.median(), s.min(), s.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e4", default=os.path.join(_HERE, "e4_results", "e4_summary.csv"))
    ap.add_argument("--steps", default=os.path.join(OUT, "chain_steps.csv"))
    ap.add_argument("--steps-csv", default=None, help="chain_steps.csv 路径（缺省取 --out 同目录）")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    args.out = _norm_dir(args.out)
    os.makedirs(args.out, exist_ok=True)
    steps_path = _resolve_csv(args.steps_csv or os.path.join(args.out, "chain_steps.csv"),
                              "E2 步级明细（chain_steps.csv）")
    
    e4 = pd.read_csv(_resolve_csv(args.e4, "E4 汇总（e4_summary.csv）"))
    steps = pd.read_csv(steps_path)
    
    # ---------- 核验表 ----------
    rows = []
    for algo in ("ADPSO", "ADSCA"):
        g = e4[e4.algo == algo]
        rows.append(dict(预测="驻留时间 = T·A_thresh = 100 代", 算法=algo, 实测=agg_range(g.dwell_p50),
                         判定="一致" if abs(g.dwell_p50.median() - 100) <= 2 else "偏离"))
        rows.append(dict(预测="踢幅度 ∈ (1.0, 1.01]", 算法=algo,
                         实测="[{:.4f}, {:.4f}]".format(g.kick_mag_min.min(), g.kick_mag_max.max()),
                         判定="一致" if g.kick_mag_min.min() > 1.0 and g.kick_mag_max.max() <= 1.01 else "偏离"))
        rows.append(dict(预测="Fano = O(1)，无幂律尾", 算法=algo, 实测=agg_range(g.fano_w100),
                         判定="一致" if g.fano_w100.max() < 100 else "待查"))
        rows.append(dict(预测="踢幅/散布 ≫ 1", 算法=algo, 实测=agg_range(g.kick_spread_p50),
                         判定="一致" if g.kick_spread_p50.median() > 1 else "偏离"))
        rows.append(dict(预测="环上传播延迟 1–2 代", 算法=algo, 实测=agg_range(g.prop_delay_p50),
                         判定="一致" if g.prop_delay_p50.median() <= 2 else "偏离"))
        neg = (g.corr_K_meanA < 0).mean()
        rows.append(dict(预测="K 与 mean(A) 反相关", 算法=algo,
                         实测="{:.3f}（{:.3f}–{:.3f}），负相关占 {}/{}".format(g.corr_K_meanA.median(),
                             g.corr_K_meanA.min(), g.corr_K_meanA.max(), int((g.corr_K_meanA < 0).sum()), len(g)),
                         判定="一致" if neg == 1.0 else "部分"))
        rows.append(dict(预测="振幅增益调制结构性失活", 算法=algo,
                         实测="激活合计 {}（{} 次运行）".format(int(g.boost_hits.sum()), len(g)),
                         判定="一致" if g.boost_hits.sum() == 0 else "偏离"))
    vt = pd.DataFrame(rows)
    _safe_to_markdown(vt, os.path.join(args.out, "table_verification.md"), index=False)
    vt.to_csv(os.path.join(args.out, "table_verification.csv"), index=False, encoding="utf-8-sig")
    
    # ---------- 活性×价值一致性 ----------
    e4["activity"] = e4.trig_count / e4.gens
    act = (e4.groupby(["algo", "func_id", "dim"]).activity.median().reset_index())
    # AD-PSO 侧价值取端点复合步 P2→P3（chain_steps_compound.csv），AD-SCA 侧取单拆步 S1→S2，
    # 与脚本头部 docstring 声明及手稿 Fig.8 数值（rho=-0.28 / -0.41）口径一致；
    # 若误用单拆步 P1→P0r，AD-PSO 侧将得到 rho=-0.18（p=0.41），与手稿不符。
    comp_path = os.path.normpath(
        os.path.abspath(os.path.join(_HERE, os.pardir, "ablation_chain", "e2_results", "chain_steps_compound.csv")))
    comp = pd.read_csv(_resolve_csv(comp_path, "消融复合步定价（chain_steps_compound.csv）"))
    ch = pd.concat([comp[comp.algo == "AD-PSO"][["algo", "func_id", "dim", "delta"]],
        steps[(steps.algo == "AD-SCA") & (steps.step == "−通道")][["algo", "func_id", "dim", "delta"]], ],
        ignore_index=True)
    ch["algo"] = ch.algo.map({"AD-PSO": "ADPSO", "AD-SCA": "ADSCA"})
    av = act.merge(ch, on=["algo", "func_id", "dim"])
    av.to_csv(os.path.join(args.out, "activity_value.csv"), index=False, encoding="utf-8-sig")
    
    from scipy.stats import spearmanr
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    print("=== 活性×价值一致性（Spearman，跨 24 数据集） ===")
    for algo, color, mk in [("ADPSO", "#2166ac", "o"), ("ADSCA", "#b2182b", "s")]:
        g = av[av.algo == algo]
        rho, p = spearmanr(g.activity, g.delta)
        print("{}: rho={:.3f} p={:.4g}".format(algo, rho, p))
        ax.scatter(g.activity, g.delta, c=color, marker=mk, s=22, alpha=0.8,
                   label="{} (rho={:.2f}, p={:.1g})".format(algo.replace("AD", "AD-"), rho, p))
    rho, p = spearmanr(av.activity, av.delta)
    print("合并 48 格: rho={:.3f} p={:.4g}".format(rho, p))
    ax.set_xscale("log")
    ax.set_xlabel("channel activity (triggers per generation)")
    ax.set_ylabel("channel value (step delta, log10)")
    ax.legend(fontsize=8)
    ax.set_title("Activity-value consistency across 24 datasets")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_activity_value.png"), dpi=200)
    plt.close(fig)
    
    print("输出 ->", args.out)


if __name__ == "__main__":
    main()
