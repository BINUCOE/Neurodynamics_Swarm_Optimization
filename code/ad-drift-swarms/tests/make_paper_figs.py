"""make_paper_figs.py — 论文新图总装（台账 T1–T8，零新实验，全部基于既有数据）。
用法: python make_paper_figs.py cd|conv|box|e2table|heatmap|agg|mapping|all
分块说明见各函数 docstring 与 图件重做行动登记.md。
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs


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
    """自仓库根向上搜索 结果_E2E3E4（任意祖先层级）；找不到则在仓库根内创建。"""
    d = REPO
    for _ in range(6):
        cand = os.path.join(d, "结果_E2E3E4")
        if os.path.isdir(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    cand = os.path.join(REPO, "结果_E2E3E4")
    os.makedirs(cand, exist_ok=True)
    return cand


OUT = _find_out()
P4 = os.path.join(REPO, "experiments/benchmark_cec2022/phase4_results")


def _need(paths, what):
    """输入齐全返回 True；缺失则打印指引并返回 False。"""
    missing = [x for x in paths if not os.path.exists(x)]
    if missing:
        print("[skip] %s 缺少: %s" % (what, " ".join(missing)))
        print("       把 结果_E2E3E4 文件夹放到仓库同级或仓库内部后重跑（全量包内有）。")
        return False
    return True


READER_NAME = {"v1.8": "AD-PSO", "ADSCAv2.1": "AD-SCA", "PSO": "PSO", "SCA": "SCA", "wDecay": "wDecay"}


# ---------------- T1 CD 图 ----------------
def fig_cd():
    """CD 图重画：读者口径名，两维度合并 24 数据集，Nemenyi 双栏图。"""
    raw = pd.read_csv(os.path.join(P4, "cec2022_raw_results.csv"))
    raw["reader"] = raw["algo"].map(READER_NAME)
    # 每数据集（func×dim）内按 30 run 中位误差排名
    med = raw.groupby(["func_id", "dim", "reader"])["error"].mean().reset_index()
    med["rank"] = med.groupby(["func_id", "dim"])["error"].rank(method="average")
    avg = med.groupby("reader")["rank"].mean().sort_values()
    k, N = len(avg), med.groupby(["func_id", "dim"]).ngroups
    q_alpha = 2.728  # Nemenyi, alpha=0.05
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * N))
    
    fig, ax = fs.newfig(180, 55)
    y = {name: i for i, name in enumerate(avg.index)}
    for name, r in avg.items():
        color, lw = fs.ALGO_STYLE[name]
        ax.plot([r], [y[name]], "o", color=color, ms=5)
        ax.text(r, y[name] + 0.28, name, ha="center", va="bottom", fontsize=8, color="black")
    xmin = 0.4
    ax.set_xlim(xmin, k + 0.6)
    ax.set_ylim(-0.6, k - 0.2)
    ax.set_yticks([])
    ax.set_xlabel("average rank (lower is better)")
    ax.set_title("Nemenyi critical-difference diagram "
                 "(CEC2022, 12 functions × 2 dims, 30 runs, "
                 "CD = 1.245)")
    # 顶部 CD 标尺
    yb = k - 0.9
    ax.plot([xmin + 0.1, xmin + 0.1 + cd], [yb, yb], color="black", lw=1.4)
    ax.text(xmin + 0.1 + cd / 2, yb + 0.12, "CD", ha="center", fontsize=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.savefig(os.path.join(OUT, "fig1_cd_v2.png"))
    plt.close(fig)
    print("fig1_cd_v2.png | avg ranks:", dict(avg.round(3)), "| CD=%.3f | N=%d" % (cd, N))


# ---------------- T2 收敛曲线 ----------------
def fe_grid(budget):
    """与 run_formal_cec2022.py 完全同构的对数 FE 网格（121 点）。"""
    g = np.unique(np.round(np.logspace(np.log10(50), np.log10(budget), 121)).astype(int))
    g[-1] = budget
    return g


EPS = 1e-10  # 零误差封底，与统计口径一致


def _load_panel(func_id, dim):
    """读一个 (F,D) 的全部曲线，返回 {读者名: (grid, err[30,121])}。"""
    z = np.load(os.path.join(P4, "curves_F{:02d}_D{:d}.npz".format(func_id, dim)))
    fg = pd.read_csv(os.path.join(P4, "cec2022_raw_results.csv"))
    fg = fg[(fg.func_id == func_id) & (fg.dim == dim)]["f_global"].iloc[0]
    budget = 200000 if dim == 10 else 500000
    grid = fe_grid(budget)
    out = {}
    for algo_raw, reader in READER_NAME.items():
        runs = sorted(k for k in z.files if k.startswith(algo_raw + "__"))
        err = np.clip(np.abs(np.stack([z[k] for k in runs]) - fg), EPS, None)
        out[reader] = (grid, err)
    return out


def fig_conv():
    """收敛曲线：D=10/D=20 各一张 12 格，log 误差 vs FE，中位曲线+IQR 带。"""
    for dim in (10, 20):
        fig, axes = plt.subplots(3, 4, figsize=(180 / 25.4, 120 / 25.4), sharex=True)
        for i, ax in enumerate(axes.flat):
            fid = i + 1
            panel = _load_panel(fid, dim)
            for name in ("PSO", "wDecay", "SCA", "AD-PSO", "AD-SCA"):
                grid, err = panel[name]
                med = np.median(err, axis=0)
                q25, q75 = np.percentile(err, [25, 75], axis=0)
                color, lw = fs.ALGO_STYLE[name]
                ax.plot(grid, med, color=color, lw=lw, label=name)
                ax.fill_between(grid, q25, q75, color=color, alpha=0.15, lw=0)
            ax.relim()
            ax.autoscale_view()
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_title("F%d" % fid, fontsize=8)
            ax.grid(True, which="major", lw=0.3, color="0.9")
            if i % 4 == 0:
                ax.set_ylabel("error")
            if i >= 8:
                ax.set_xlabel("FE")
        handles = [Line2D([], [], color=fs.ALGO_STYLE[n][0], lw=fs.ALGO_STYLE[n][1], label=n) for n in
                   ("AD-PSO", "AD-SCA", "PSO", "SCA", "wDecay")]
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=True, bbox_to_anchor=(0.5, -0.05))
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_conv_D%d.png" % dim))
        plt.close(fig)
        print("fig_conv_D%d.png" % dim)


# ---------------- T3 箱体图 ----------------
def fig_box():
    """箱体图：同 12 格网格，每格 5 算法 30 runs 的 log10 误差箱线。"""
    raw = pd.read_csv(os.path.join(P4, "cec2022_raw_results.csv"))
    raw["reader"] = raw["algo"].map(READER_NAME)
    raw["logerr"] = np.log10(np.clip(raw["error"], EPS, None))
    names = ["AD-PSO", "AD-SCA", "PSO", "SCA", "wDecay"]
    for dim in (10, 20):
        fig, axes = plt.subplots(3, 4, figsize=(180 / 25.4, 120 / 25.4), sharey=True)
        for i, ax in enumerate(axes.flat):
            fid = i + 1
            sub = raw[(raw.func_id == fid) & (raw.dim == dim)]
            data = [sub[sub.reader == n]["logerr"].values for n in names]
            bp = ax.boxplot(data, vert=True, widths=0.55, showfliers=False, patch_artist=True,
                            medianprops=dict(color="black", lw=1.0), whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7))
            for patch, n in zip(bp["boxes"], names):
                patch.set(facecolor=fs.ALGO_STYLE[n][0], alpha=0.75, lw=0.7)
            ax.set_title("F%d" % fid, fontsize=8)
            ax.set_xticks([])
            ax.grid(True, axis="y", lw=0.3, color="0.9")
            if i % 4 == 0:
                ax.set_ylabel("$\\log_{10}$ error")
            ymin = min(np.percentile(d, 0) for d in data if len(d))
        handles = [Line2D([], [], marker="s", ls="", ms=7, color=fs.ALGO_STYLE[n][0], label=n) for n in names]
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_box_D%d.png" % dim))
        plt.close(fig)
        print("fig_box_D%d.png" % dim)


# ---------------- T5 消融热力图 ----------------
HEAT_COLS = [  # (algo, step, 显示名) —— 按机制同源对齐两身
    ("AD-PSO", "−K自适应", "− K adapt.\nAD-PSO"), ("AD-SCA", "−K自适应", "− K adapt.\nAD-SCA"),
    ("AD-PSO", "−通道", "− channel\nAD-PSO"), ("AD-SCA", "−通道", "− channel\nAD-SCA"),
    ("AD-PSO", "−环+WTA", "− ring + WTA\nAD-PSO"), ("AD-SCA", "−环+WTA", "− ring + WTA\nAD-SCA"),
    ("AD-SCA", "−核层修正", "− kernel fixes\nAD-SCA"), ]
HEAT_GROUPS = [("interface", 0, 1), ("core: channel", 2, 3), ("substrate", 4, 5), ("kernel fix", 6, 6)]


def fig_heatmap():
    """消融热力图 v2：机制对齐 7 列 × 24 数据集；封底格灰底†、|Δ|<0.2 不标数、
    显著格粗体；标题写结论。"""
    cs_path = os.path.join(OUT, "chain_steps.csv")
    if not os.path.exists(cs_path):
        cs_path = os.path.join(REPO, "experiments/ablation_chain/e2_results/chain_steps.csv")
    if not _need([cs_path], "heatmap"):
        return
    cs = pd.read_csv(cs_path)
    funcs = sorted(cs.func_id.unique())
    dims = sorted(cs.dim.unique())
    nR = len(funcs) * len(dims)
    M = np.zeros((nR, len(HEAT_COLS)))
    F = np.zeros_like(M, bool)
    P = np.ones_like(M)
    for j, (algo, step, _) in enumerate(HEAT_COLS):
        g = cs[(cs.algo == algo) & (cs.step == step)].set_index(["func_id", "dim"])
        for i, (fid, dm) in enumerate([(f, d) for f in funcs for d in dims]):
            r = g.loc[(fid, dm)]
            M[i, j] = r.delta
            F[i, j] = bool(r.floored)
            P[i, j] = r.wilcoxon_p if np.isfinite(r.wilcoxon_p) else 1.0
    Mplot = np.where(F, np.nan, M)
    
    fig, ax = fs.newfig(180, 150)
    vmax = 8.0
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(np.clip(np.where(F, 0, M), -vmax, vmax), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(nR):
        for j in range(len(HEAT_COLS)):
            if F[i, j]:
                ax.text(j, i, "†", ha="center", va="center", fontsize=7, color="0.45")
            elif abs(M[i, j]) >= 0.2:
                w = "bold" if P[i, j] < 0.05 else "normal"
                ax.text(j, i, "%+.1f" % M[i, j], ha="center", va="center", fontsize=6.5, weight=w,
                        color="white" if abs(M[i, j]) > 4.5 else "black")
    ax.set_xticks(range(len(HEAT_COLS)))
    ax.set_xticklabels([c[2] for c in HEAT_COLS], fontsize=7)
    ylabels = ["F%d, D%d" % (f, d) for f in funcs for d in dims]
    ax.set_yticks(range(nR))
    ax.set_yticklabels(ylabels, fontsize=7)
    for i in range(1, nR):
        if dims[0] == 10 and (i % 2 == 0):
            ax.axhline(i - 0.5, color="white", lw=1.2)
    # 机制分组括注（矩阵上方）
    for name, j0, j1 in HEAT_GROUPS:
        x0, x1 = j0 - 0.42, j1 + 0.42
        ax.plot([x0, x0, x1, x1], [-0.55, -0.9, -0.9, -0.55], color="0.3", lw=0.8, clip_on=False)
        ax.text((x0 + x1) / 2, -1.05, name, ha="center", va="bottom", fontsize=7.5, style="italic", clip_on=False)
    ax.set_xlim(-0.5, len(HEAT_COLS) - 0.5)
    ax.set_ylim(nR - 0.5, -2.0)
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.015)
    cb.set_label("Δ log10 error (removed − full)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(os.path.join(OUT, "fig_ablation_heatmap_v2.png"))
    plt.close(fig)
    print("fig_ablation_heatmap_v2.png")


E4DIR = os.path.join(REPO, "experiments/neuro_mapping/e4_results")


def compute_agg():
    """T6 聚合指标：以 e4_summary.csv（已核验口径）为权威列，
    附 exploratory 的纪元内散布（事件日志现算，纯探索不入正文）。返回 144 行。"""
    import glob, re
    if not _need([os.path.join(E4DIR, "e4_summary.csv")], "agg"):
        return
    summ = pd.read_csv(os.path.join(E4DIR, "e4_summary.csv"))
    rows = []
    for f in sorted(glob.glob(os.path.join(E4DIR, "e4_obs_*.npz"))):
        m = re.match(r"e4_obs_(ADPSO|ADSCA)_F(\d+)_D(\d+)_r(\d+)", os.path.basename(f))
        d = np.load(f)
        ev = d["ev_gen"]
        spreads = []
        for e0 in range(0, int(d["gen"].max()) + 1, 100):
            sel = (ev >= e0) & (ev < e0 + 100)
            if sel.sum() >= 5:
                spreads.append(np.std(ev[sel].astype(float)))
        rows.append(dict(algo=m.group(1), func_id=int(m.group(2)), dim=int(m.group(3)), run_id=int(m.group(4)),
                         epoch_spread=float(np.median(spreads)) if spreads else np.nan))
    esp = pd.DataFrame(rows)
    df = summ.merge(esp, on=["algo", "func_id", "dim", "run_id"], how="left")
    df.to_csv(os.path.join(OUT, "e4_aggregate.csv"), index=False)
    print("e4_aggregate.csv:", df.shape)
    print(df.groupby("algo")[["dwell_p50", "fano_w100", "kick_spread_p50", "prop_delay_p50", "corr_K_meanA",
                              "epoch_spread"]].median().T.round(2))


# ---------------- T8 神经映射复合 figure ----------------
# T7 选格（规则：该身 D10 内指标最接近全体中位数的 run）：
EXEMPLAR = {"A": {"ADSCA": (10, 10, 1), "ADPSO": (1, 10, 0)},  # dwell=100
            "B": {"ADPSO": (10, 10, 0), "ADSCA": (10, 10, 1)},  # F10 双身
            "C": {"ADSCA": (11, 10, 1), "ADPSO": (11, 10, 0)}}  # prop delay 5/4
ZOOM = {"A": (1500, 2100), "B": (1700, 2300), "C": (1750, 2050)}


def _load_e4(body, fid, dim, rid):
    return np.load(os.path.join(E4DIR, "e4_obs_%s_F%02d_D%d_r%d.npz" % (body, fid, dim, rid)))


def _panel_traces(ax):
    """行A左：两身各 5 条单粒子 A 值曲线（蓝 AD-PSO / 红 AD-SCA）。"""
    g0, g1 = ZOOM["A"]
    handles, labels = [], []
    for body, c, name in [("ADPSO", fs.C_ADPSO, "AD-PSO"), ("ADSCA", fs.C_ADSCA, "AD-SCA")]:
        fid, dim, rid = EXEMPLAR["A"][body]
        A = _load_e4(body, fid, dim, rid)["A_full"][g0:g1]
        h = None
        for pp in range(0, 50, 10):
            h, = ax.plot(range(g0, g1), A[:, pp], lw=0.7, color=c, alpha=0.8)
        handles.append(h)
        labels.append("%s (F%d)" % (name, fid))
    ax.axhline(1.0, color="0.2", ls="--", lw=0.9)
    ax.text(g0 + 8, 1.03, "threshold $A$ = 1", fontsize=7.5, va="bottom")
    # 顶部预留标注带（1.05–1.22），dwell 双箭头与 threshold 线齐平展示，永不被曲线遮挡
    ax.annotate("", xy=(1768, 1.11), xytext=(1868, 1.11), arrowprops=dict(arrowstyle="<->", lw=0.9))
    ax.text(1818, 1.135, "dwell ≈ T = 100 generations", fontsize=7.5, ha="center", va="bottom")
    ax.legend(handles, labels, loc="lower right", frameon=True, framealpha=0.95, edgecolor="0.5", fontsize=7.5)
    ax.set_ylim(-0.03, 1.24)
    ax.set_ylabel("$A$ (5 of 50 particles per body)", fontsize=8)
    ax.set_title("Leaky integrate-and-fire channel, both bodies", fontsize=9.5, loc="left")


def _panel_raster(ax):
    g0, g1 = ZOOM["B"]
    for k, (body, c) in enumerate([("ADSCA", fs.C_ADSCA), ("ADPSO", fs.C_ADPSO)]):
        off = k * 55
        fid, dim, rid = EXEMPLAR["B"][body]
        d = _load_e4(body, fid, dim, rid)
        sel = (d["ev_gen"] >= g0) & (d["ev_gen"] < g1)
        ax.plot(d["ev_gen"][sel], d["ev_pid"][sel] + off, "|", color=c, ms=3.5, mew=0.8)
        name = "AD-SCA" if body == "ADSCA" else "AD-PSO"
        ax.text(g1 - 8, off + 46, name, color=c, fontsize=8, weight="bold", ha="right", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c, lw=0.6))
    ax.axhline(52, color="0.6", lw=0.7)
    for e in range((g0 // 100 + 1) * 100, g1, 100):
        ax.axvline(e, color="0.88", lw=0.6, zorder=0)
    ax.set_ylim(-3, 108)
    ax.set_yticks([])
    ax.set_ylabel("particle", fontsize=8)
    ax.set_title("Kicks follow the search, not a global clock (F10)", fontsize=8.5, loc="left")
    ax.set_xlabel("generation", fontsize=8)


def _draw_waves(fig, gs_cell):
    """行C左实际绘制：两个叠放热图 + 各自标注。"""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    sub = gs_cell.subgridspec(2, 1, hspace=0.32)
    g0, g1 = ZOOM["C"]
    axes = []
    for k, (body, c, lab) in enumerate([("ADPSO", fs.C_ADPSO, "AD-PSO"), ("ADSCA", fs.C_ADSCA, "AD-SCA")]):
        ax = fig.add_subplot(sub[k])
        fid, dim, rid = EXEMPLAR["C"][body]
        A = _load_e4(body, fid, dim, rid)["A_full"][g0:g1].T
        im = ax.imshow(A, aspect="auto", origin="lower", cmap="viridis", extent=[g0, g1, 0, 50], vmin=0, vmax=1.01)
        ax.set_yticks([0, 25, 50])
        t = ax.text(g0 + 6, 44, lab, color="white", fontsize=7.5, weight="bold")
        t.set_path_effects([pe.withStroke(linewidth=2.5, foreground="black")])
        if k == 0:
            ax.annotate("", xy=(1985, 42), xytext=(1785, 4), arrowprops=dict(arrowstyle="->", lw=1.4, color="white",
                                                                             path_effects=[pe.withStroke(linewidth=2.5,
                                                                                                         foreground="black")]))
            t2 = ax.text(1885, 12, "wavefront sweeps the ring", fontsize=7, color="white", ha="center")
            t2.set_path_effects([pe.withStroke(linewidth=2.5, foreground="black")])
            ax.set_title("Kicks propagate along the ring (F11)", fontsize=8.5, loc="left")
        else:
            ax.set_xlabel("generation", fontsize=8)
        ax.set_ylabel("particle", fontsize=7.5)
        axes.append((ax, im))
    return axes


def _panel_agg(ax, col, ylab, ref, reflab, ylim, ref2=None, ref2lab=None):
    rng = np.random.default_rng(0)
    for k, (b, c) in enumerate([("ADPSO", fs.C_ADPSO), ("ADSCA", fs.C_ADSCA)]):
        v = AGG[AGG.algo == b][col].values
        ax.scatter(np.full(len(v), k) + rng.uniform(-0.16, 0.16, len(v)), v, s=5, color=c, alpha=0.4, lw=0)
        ax.hlines(np.median(v), k - 0.28, k + 0.28, color="black", lw=1.1)
    if ref is not None:
        ax.axhline(ref, color="0.25", ls="--", lw=0.9)
        ax.text(0.97, 0.07, reflab, fontsize=7, va="bottom", ha="right", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", lw=0.5))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["AD-PSO", "AD-SCA"], fontsize=7.5)
    ax.set_ylabel(ylab, fontsize=8)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(*ylim)


def fig_mapping():
    """目标3复合figure：左列机制直观图（两身并列），右列全基准聚合（144 runs）。
    面板字母：(a)A轨迹 (b)驻留 (c)踢栅格 (d)Fano (e)波前 (f)传播延迟。"""
    global AGG
    agg_path = os.path.join(OUT, "e4_aggregate.csv")
    if not os.path.exists(agg_path):
        print("[hint] 先运行: python make_paper_figs.py agg")
        return
    AGG = pd.read_csv(agg_path)
    fig = plt.figure(figsize=(180 / 25.4, 215 / 25.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1], hspace=0.45, wspace=0.30, left=0.08, right=0.985, top=0.955,
                          bottom=0.06)
    axA = fig.add_subplot(gs[0, 0])
    _panel_traces(axA)
    axA.set_xlabel("generation", fontsize=8)
    axA.text(-0.11, 1.04, "(a)", transform=axA.transAxes, fontsize=9, weight="bold")
    axA2 = fig.add_subplot(gs[0, 1])
    _panel_agg(axA2, "dwell_p50", "dwell time (generations)", 100, "clock period $T$ = 100", (85, 115))
    axA2.set_title("All 144 runs: dwell = T", fontsize=8.5, loc="left")
    axA2.text(-0.13, 1.04, "(b)", transform=axA2.transAxes, fontsize=9, weight="bold")
    axB = fig.add_subplot(gs[1, 0])
    _panel_raster(axB)
    axB.text(-0.11, 1.04, "(c)", transform=axB.transAxes, fontsize=9, weight="bold")
    axB2 = fig.add_subplot(gs[1, 1])
    _panel_agg(axB2, "fano_w100", "Fano factor of kick counts\nper 100-generation window", 1.0, "Poisson = 1", (0, 12))
    axB2.set_title("All 144 runs: bursts are Poisson-like", fontsize=8.5, loc="left")
    axB2.text(-0.13, 1.04, "(d)", transform=axB2.transAxes, fontsize=9, weight="bold")
    wave_axes = _draw_waves(fig, gs[2, 0])
    wave_axes[0][0].text(-0.11, 1.30, "(e)", transform=wave_axes[0][0].transAxes, fontsize=9, weight="bold")
    cb = fig.colorbar(wave_axes[-1][1], ax=[a for a, _ in wave_axes], shrink=0.9, pad=0.02, aspect=30)
    cb.set_label("$A$", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    axC2 = fig.add_subplot(gs[2, 1])
    _panel_agg(axC2, "prop_delay_p50", "ring propagation delay (generations)", None, None, (0, 10))
    axC2.set_title("All 144 runs: propagation delay = 4-5 generations", fontsize=8.5, loc="left")
    axC2.text(-0.13, 1.04, "(f)", transform=axC2.transAxes, fontsize=9, weight="bold")
    fig.savefig(os.path.join(OUT, "fig_neuro_mapping.png"))
    plt.close(fig)
    print("fig_neuro_mapping.png")


# ---------------- T10 E3/tsweep、activity、kernel 重画 ----------------
def fig_tsweep():
    """E3 剂量-响应 v2：T 对数轴、抖动散点、显著恶化深色、面板内计数。"""
    if not _need([os.path.join(OUT, "tsweep_doseresponse.csv"), os.path.join(OUT, "tsweep_contrasts.csv")], "tsweep"):
        return
    dr = pd.read_csv(os.path.join(OUT, "tsweep_doseresponse.csv"))
    ct = pd.read_csv(os.path.join(OUT, "tsweep_contrasts.csv"))
    harm = {}
    for (a, fid, dm, T), g in ct[ct.contrast.str.startswith("T")].groupby(["algo", "func_id", "dim", "contrast"]):
        Tval = int(contrast_T(T))
        harm[(a, fid, dm, Tval)] = bool((g.wilcoxon_p < 0.05).any() and (g.delta > 0).any())
    Ts = [25, 50, 100, 200]
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 78 / 25.4))
    for k, (body, cname) in enumerate([("ADPSO", "AD-PSO"), ("ADSCA", "AD-SCA")]):
        ax = axes[k]
        sub = dr[dr.algo == body]
        for i, T in enumerate(Ts):
            v = np.clip(sub[sub["T"] == T]["delta"].values, -8, 8)
            x = i + rng.uniform(-0.22, 0.22, len(v))
            sig = np.array([harm.get((body, r.func_id, r.dim, T), False) for r in sub[sub["T"] == T].itertuples()])
            ax.scatter(x[~sig], v[~sig], s=8, color="0.75", lw=0)
            ax.scatter(x[sig], v[sig], s=11, color="#b2182b", lw=0)
            med = np.median(v)
            ax.plot([i - 0.3, i + 0.3], [med, med], color="black", lw=1.2)
            if T != 100:
                n = int(sig.sum())
                ax.text(i, -7.6, "%d/24" % n, ha="center", fontsize=7, color="#b2182b" if n >= 6 else "0.3")
        ax.axhline(0, color="0.3", ls="--", lw=0.8)
        ax.set_xticks(range(4))
        ax.set_xticklabels([str(t) for t in Ts])
        ax.set_xlabel("channel period $T$ (generations)", fontsize=8)
        if k == 0:
            ax.set_ylabel("Δ log10 error vs $T$ = 100", fontsize=8)
        ax.set_title(cname, fontsize=9)
        ax.set_ylim(-8.3, 8.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_tsweep_aggregate_v2.png"))
    plt.close(fig)
    print("fig_tsweep_aggregate_v2.png")


def contrast_T(s):
    import re
    return re.match(r"T(\d+)_", s).group(1)


def fig_activity():
    """activity-value v2：规范化排版 + 结论标题 + 零参考线。"""
    if not _need([os.path.join(OUT, "activity_value.csv")], "activity"):
        return
    av = pd.read_csv(os.path.join(OUT, "activity_value.csv"))
    from scipy import stats as st
    fig, ax = fs.newfig(88, 70)
    labs = {}
    for body, c in [("ADPSO", fs.C_ADPSO), ("ADSCA", fs.C_ADSCA)]:
        g = av[av.algo == body]
        ax.scatter(g.activity, g.delta, s=26, color=c, lw=0, alpha=0.85)
        rho, p = st.spearmanr(g.activity, g.delta)
        labs[body] = (rho, p)
    ax.axhline(0, color="0.3", ls="--", lw=0.8)
    ax.set_xlabel("channel activity (kicks per generation)", fontsize=8)
    ax.set_ylabel("channel value (Δ log10 error)", fontsize=8)
    txt = "AD-PSO: ρ = %.2f, p = %.2f\nAD-SCA: ρ = %.2f, p = %.2f" % (
        labs["ADPSO"][0], labs["ADPSO"][1], labs["ADSCA"][0], labs["ADSCA"][1])
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.6))
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=fs.C_ADPSO, label="AD-PSO"),
                       Line2D([], [], marker="o", ls="", color=fs.C_ADSCA, label="AD-SCA")], loc="lower left",
              frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_activity_value_v2.png"))
    plt.close(fig)
    print("fig_activity_value_v2.png")


def fig_kernel():
    """核解析三联 v2：常数与 algorithms/ad_pso.py、ad_sca.py 对齐
    (w=0.7, c1=c2=1.4 → φ∈(0,2.8), E[φ]=1.4 c=0.7, a∈(0,2], a*=2.34)。"""
    from matplotlib.patches import Circle
    w, c1 = 0.7, 1.4
    fig, axes = plt.subplots(1, 2, figsize=(120 / 25.4, 58 / 25.4))
    # (a) PSO 根轨迹
    ax = axes[0]
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), ls="--", color="0.6", lw=0.9)
    phi = np.linspace(0.05, 2.8, 300)
    lam = (1 - phi + w + np.sqrt(np.clip((1 - phi + w) ** 2 - 4 * w, 0, None))) / 2
    lam2 = (1 - phi + w - np.sqrt(np.clip((1 - phi + w) ** 2 - 4 * w, 0, None))) / 2
    disc = (1 - phi + w) ** 2 - 4 * w
    re, im = (1 - phi + w) / 2, np.sqrt(np.clip(-disc, 0, None)) / 2
    ax.plot(re[disc < 0], im[disc < 0], color=fs.C_ADPSO, lw=1.4)
    ax.plot(re[disc < 0], -im[disc < 0], color=fs.C_ADPSO, lw=1.4)
    lamE_re, lamE_im = (1 - 1.4 + w) / 2, np.sqrt(4 * w - (1 - 1.4 + w) ** 2) / 2
    ax.plot([lamE_re], [lamE_im], "*", color="black", ms=9)
    ax.annotate("$E[\\varphi] = 1.4$\n$|\\lambda| = \\sqrt{w} \\approx 0.84$", xy=(lamE_re, lamE_im),
                xytext=(0.60, 0.95), fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.text(0.02, -0.12, "unit circle", fontsize=7, color="0.4")
    ax.set_xlabel("$Re$", fontsize=8)
    ax.set_ylabel("$Im$", fontsize=8)
    ax.set_title("(a) second-order inertial kernel:\n$|\\lambda| = \\sqrt{w} < 1$ (spiral sink)", fontsize=8.5)
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    # (b) SCA m(a)
    ax = axes[1]
    a = np.linspace(0, 3, 200)
    ax.plot(a, 1 + a ** 2 / 6, color=fs.C_ADSCA, lw=1.4, label="$c$ = 0")
    ax.plot(a, (1 - 0.7) ** 2 + a ** 2 / 6, color="#e08214", lw=1.4, label="$c$ = 0.7 (anchor pull)")
    ax.axhline(1, color="0.3", ls="--", lw=0.9)
    astar = np.sqrt(6 * (1 - (1 - 0.7) ** 2))
    ax.plot([astar], [1], "o", color="black", ms=5)
    ax.annotate("$a^* \\approx 2.34$", xy=(astar, 1), xytext=(2.42, 0.60), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.text(1.90, 1.05, "$\\rho$ = 1 (neutral)", fontsize=7, color="0.3")
    ax.text(0.06, 0.56, "$c$ = 0: $\\rho(a) > 1$ for all $a$\n(no intrinsic sink)", fontsize=7)
    ax.set_xlabel("oscillation amplitude $a$", fontsize=8)
    ax.set_ylabel("mean-square multiplier $\\rho(a)$", fontsize=8)
    ax.set_title("(b) first-order oscillatory kernel:\nno sink without the anchor", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_xlim(0, 3)
    ax.set_ylim(0.5, 2.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_kernel_v2.png"))
    plt.close(fig)
    print("fig4_kernel_v2.png")


# ---------------- T11 消融热力图 v3（含 --with-extras 方向对照与纯净 −通道） ----------------
def fig_heatmap_v3():
    """热力图 v3：7 条链步（AD-PSO −通道 为纯净单拆 P1→P0r）+ 2 列方向对照。
    数据：e2_results/e2_chain_raw.csv（含 extras）+ E1 端点。"""
    from scipy import stats as st
    E2RAW = os.path.join(REPO, "experiments/ablation_chain/e2_results/e2_chain_raw.csv")
    E1RAW = os.path.join(REPO, "experiments/benchmark_cec2022/phase4_results/cec2022_raw_results.csv")
    if not _need([E2RAW, E1RAW], "heatmap3"):
        return
    rn = pd.read_csv(E2RAW)
    e1 = pd.read_csv(E1RAW)
    COLS = [("AD-PSO", "− K adapt.\nAD-PSO", "v1.8", "P1", e1, rn),
            ("AD-PSO", "− ring + WTA\nAD-PSO", "P1", "P2", rn, rn),
            ("AD-PSO", "− channel\nAD-PSO", "P1", "P0r", rn, rn),
            ("AD-PSO", "direction:\ngauss AD-PSO", "v1.8", "P0d", e1, rn),
            ("AD-SCA", "− kernel fixes\nAD-SCA", "S3", "SCA", rn, e1),
            ("AD-SCA", "− channel\nAD-SCA", "S1", "S2", rn, rn), ("AD-SCA", "− ring + WTA\nAD-SCA", "S2", "S3", rn, rn),
            ("AD-SCA", "− K adapt.\nAD-SCA", "ADSCAv2.1", "S1", e1, rn),
            ("AD-SCA", "direction:\ngauss AD-SCA", "ADSCAv2.1", "S0d", e1, rn), ]
    nR = 24
    nC = len(COLS)
    M = np.zeros((nR, nC))
    P = np.ones((nR, nC))
    F = np.zeros((nR, nC), bool)
    for j, (algo, lab, pa, ch, sp, sc) in enumerate(COLS):
        for i, (fid, dm) in enumerate([(f, d) for f in range(1, 13) for d in (10, 20)]):
            gp = sp[(sp.algo == pa) & (sp.func_id == fid) & (sp.dim == dm)]
            gc = sc[(sc.algo == ch) & (sc.func_id == fid) & (sc.dim == dm)]
            m = gp.merge(gc, on="run_id", suffixes=("_p", "_c"))
            ep = np.maximum(m.error_p.values, EPS)
            ec = np.maximum(m.error_c.values, EPS)
            M[i, j] = np.median(np.log10(ec / ep))
            F[i, j] = bool((np.median(m.error_p.values) <= EPS) and (np.median(m.error_c.values) <= EPS))
            if len(m) > 5 and not F[i, j]:
                try:
                    P[i, j] = st.wilcoxon(m.error_c.values, m.error_p.values).pvalue
                except ValueError:
                    P[i, j] = float("nan")
    fig, ax = fs.newfig(180, 160)
    vmax = 8.0
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    ax.imshow(np.where(F, 0, np.clip(M, -vmax, vmax)), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(nR):
        for j in range(nC):
            if F[i, j]:
                ax.text(j, i, "\u2020", ha="center", va="center", fontsize=7, color="0.45")
            elif abs(M[i, j]) >= 0.2:
                w = "bold" if P[i, j] < 0.05 else "normal"
                ax.text(j, i, "%+.1f" % M[i, j], ha="center", va="center", fontsize=6.5, weight=w,
                        color="white" if abs(M[i, j]) > 4.5 else "black")
    ax.set_xticks(range(nC))
    ax.set_xticklabels([c[1] for c in COLS], fontsize=7)
    ax.set_yticks(range(nR))
    ax.set_yticklabels(["F%d, D%d" % (f, d) for f in range(1, 13) for d in (10, 20)], fontsize=7)
    for i in (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22):
        ax.axhline(i - 0.5, color="white", lw=1.0)
    ax.set_xlim(-0.5, nC - 0.5)
    ax.set_ylim(nR - 0.5, -0.6)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-vmax, vmax)), ax=ax, shrink=0.75, pad=0.015)
    cb.set_label("Δ log10 error (removed − full)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(os.path.join(OUT, "fig_ablation_heatmap_v3.png"))
    plt.close(fig)
    print("fig_ablation_heatmap_v3.png")


def fig_rankmap():
    """分函数名次图（候选，2026-10-06）：逐 (函数, 维度) 格内按 30 run 均值误差排名，
    色块 = 名次（1 绿 = 最优，5 红 = 最差），底部行为该维度平均名次（与 Table 4 口径同源）。"""
    raw = pd.read_csv(os.path.join(P4, "cec2022_raw_results.csv"))
    raw["reader"] = raw["algo"].map(READER_NAME)
    med = raw.groupby(["func_id", "dim", "reader"])["error"].mean().reset_index()
    med["rank"] = med.groupby(["func_id", "dim"])["error"].rank(method="average")
    order = ["AD-PSO", "AD-SCA", "PSO", "wDecay", "SCA"]
    fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 62 / 25.4))
    for ax, dim, lab in [(axes[0], 10, "(a) D = 10"), (axes[1], 20, "(b) D = 20")]:
        sub = med[med.dim == dim].pivot_table(index="func_id", columns="reader", values="rank")[order]
        M = sub.values
        im = ax.imshow(M, cmap="RdYlGn_r", vmin=1, vmax=5, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, "%g" % v, ha="center", va="center", fontsize=6.5,
                        color="white" if (v <= 1.5 or v >= 4.5) else "0.15")
        # 列间白网格
        ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 12, 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.2)
        ax.tick_params(which="minor", length=0)
        ax.set_xticks(range(5))
        ax.set_xticklabels(order, fontsize=7.5)
        ax.set_yticks(range(12))
        ax.set_yticklabels(["F%d" % f for f in range(1, 13)], fontsize=7.5)
        # 底部平均名次行（与格区粗线分隔，行下留白再放算法列标签，互不挤压）
        avg = sub.mean()
        for j, name in enumerate(order):
            ax.text(j, 12.1, "%.3f" % avg[name], ha="center", va="center", fontsize=7, weight="bold")
        ax.text(-0.62, 12.2, "mean\nrank", ha="right", va="center", fontsize=7)
        ax.axhline(11.5, color="0.2", lw=1.0)
        ax.set_ylim(12.6, -0.5)
        ax.set_title(lab, fontsize=9, loc="center")
        ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_rankmap.png"))
    plt.close(fig)
    print("fig_rankmap.png")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["all"]
    # 2026-10-06 备案：fig_box（箱线图）已由 fig_rankmap（逐函数名次图）取代退役，
    # fig_box 函数保留存档但不再注册出图。
    for name, fn in [("cd", fig_cd), ("conv", fig_conv), ("heatmap", fig_heatmap), ("agg", compute_agg),
                     ("mapping", fig_mapping), ("tsweep", fig_tsweep), ("activity", fig_activity),
                     ("kernel", fig_kernel), ("heatmap3", fig_heatmap_v3), ("rankmap", fig_rankmap)]:
        if name in todo or "all" in todo:
            fn()
