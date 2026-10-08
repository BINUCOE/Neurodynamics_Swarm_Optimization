# -*- coding: utf-8 -*-
"""延迟 × 基质定价相关性：逐数据集「传播延迟」与「−环+WTA 步 Δ」的 Spearman 相关。
零新实验：e4_summary.csv（延迟） × chain_steps.csv（基质定价）。
产出：delay_vs_ringdelta.csv（逐数据集合并表）、fig_delay_vs_ringdelta.png（散点）。
"""
import pathlib
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _os


def _find_repo():
    d = _os.path.dirname(_os.path.abspath(__file__))
    for _ in range(8):
        for cand in (d, _os.path.join(d, "ad-drift-swarms")):
            if _os.path.isdir(_os.path.join(cand, "core")) and _os.path.isdir(_os.path.join(cand, "experiments")):
                return cand
        parent = _os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise SystemExit("找不到仓库根目录（需同时包含 core/ 与 experiments/）")


REPO = pathlib.Path(_find_repo())
HERE = pathlib.Path(__file__).resolve().parent


def _find_out():
    for cand in (REPO / "结果_E2E3E4", REPO.parent / "结果_E2E3E4"):
        if cand.is_dir():
            return cand
    return REPO / "结果_E2E3E4"


OUT = _find_out()
E4 = HERE / "e4_results" / "e4_summary.csv"
STEPS = OUT / "chain_steps.csv"
CAP = 10  # 传播延迟的截断上限（插桩口径）

FAMILY = {1: "unimodal", 2: "basic", 3: "basic", 4: "basic", 5: "basic", 6: "hybrid", 7: "hybrid", 8: "hybrid",
          9: "composition", 10: "composition", 11: "composition", 12: "composition"}


def load_delay() -> pd.DataFrame:
    df = pd.read_csv(E4, encoding="utf-8")
    df["algo"] = df["algo"].map({"ADPSO": "AD-PSO", "ADSCA": "AD-SCA"})  # 对齐 chain_steps 口径
    g = (df.groupby(["algo", "func_id", "dim"], as_index=False).agg(delay_p50=("prop_delay_p50", "median"),
                                                                    delay_max=("prop_delay_p50", "max"),
                                                                    trig_med=("trig_count", "median")))
    g["censored"] = g["delay_max"] >= CAP  # 任一种子触及截断 → 标记
    return g


def load_ring_delta() -> pd.DataFrame:
    df = pd.read_csv(STEPS, encoding="utf-8-sig")
    d = df[df["step"] == "−环+WTA"][["algo", "func_id", "dim", "delta", "floored", "wilcoxon_p"]]
    return d.rename(columns={"delta": "ring_delta"})


def main() -> None:
    m = load_delay().merge(load_ring_delta(), on=["algo", "func_id", "dim"], how="inner")
    m["family"] = m["func_id"].map(FAMILY)
    m.to_csv(OUT / "delay_vs_ringdelta.csv", index=False, encoding="utf-8-sig")
    
    # ---- 统计 ----
    lines = ["# 延迟 × 基质定价相关性\n", "延迟 = E4 逐格 prop_delay_p50 的 3 种子中位数；基质定价 = E2 −环+WTA 步 Δ。",
             "截断口径：延迟上限 10 代，任一种子触及即标 censored。\n"]
    for algo, sub in m.groupby("algo"):
        rho, p = stats.spearmanr(sub["delay_p50"], sub["ring_delta"])
        keep = sub[~sub["censored"]]
        rho2, p2 = (stats.spearmanr(keep["delay_p50"], keep["ring_delta"]) if len(keep) >= 6 else (np.nan, np.nan))
        lines.append(f"- **{algo}**：全部 24 点 rho={rho:+.3f}, p={p:.4f}；"
                     f"剔除截断格 n={len(keep)} 后 rho={rho2:+.3f}, p={p2:.4f}")
        for fam, fs in sub.groupby("family"):
            if len(fs) >= 4:
                r_f, p_f = stats.spearmanr(fs["delay_p50"], fs["ring_delta"])
                lines.append(f"  - 族内 {fam}（n={len(fs)}）：rho={r_f:+.3f}, p={p_f:.3f}（描述性）")
    rho, p = stats.spearmanr(m["delay_p50"], m["ring_delta"])
    lines.append(f"- **合并 48 点**：rho={rho:+.3f}, p={p:.2e}")
    (OUT / "delay_corr_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    
    # ---- 图 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    fam_color = {"unimodal": "#4c72b0", "basic": "#55a868", "hybrid": "#dd8452", "composition": "#c44e52"}
    dim_marker = {10: "o", 20: "s"}
    for ax, (algo, sub) in zip(axes, m.groupby("algo")):
        for (fam, dm), g in sub.groupby(["family", "dim"]):
            ax.scatter(g["delay_p50"], g["ring_delta"], c=fam_color[fam], marker=dim_marker[dm], s=42, alpha=0.85,
                       edgecolors="white", linewidths=0.5)
        cens = sub[sub["censored"]]
        ax.scatter(cens["delay_p50"], cens["ring_delta"], facecolors="none", edgecolors="black", s=90, linewidths=1.1,
                   label="censored (delay>=10)")
        rho, p = stats.spearmanr(sub["delay_p50"], sub["ring_delta"])
        ax.set_title(f"{algo}: Spearman rho={rho:+.2f}, p={p:.3f}")
        ax.set_xlabel("propagation delay (median generations)")
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("ring+WTA step delta (log10 median error ratio)")
    axes[1].legend(loc="lower right", fontsize=8)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f) for f, c in fam_color.items()]
    handles += [plt.Line2D([], [], marker=m_, ls="", color="gray", label=f"D{d}") for d, m_ in dim_marker.items()]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "fig_delay_vs_ringdelta.png", dpi=300)
    print("saved:", OUT / "fig_delay_vs_ringdelta.png")


if __name__ == "__main__":
    main()
