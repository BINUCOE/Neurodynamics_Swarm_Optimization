#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stats_formal.py — Phase 4 统计与表格生成（备忘录 v1.4 第五部分，R7 口径）

输入（--results 目录，run_formal_cec2022.py 的落盘）：
  cec2022_raw_results.csv     逐 run 原始结果
  curves_F{02d}_D{d}.npz      逐 run FE 索引收敛曲线

输出（写回同一目录）：
  cec2022_summary_by_function.csv   func_id, dim, algo, f_global, mean_error
                                    （visualization.py loader 约定）
  cec2022_stats_full.csv            mean/std/best/worst/median/n 全表
  cec2022_wilcoxon.csv              每 (func,dim)：v1.8 / ADSCAv2.1 对其余各算法
                                    的成对 Wilcoxon 符号秩 p 值与胜负判定
  cec2022_friedman.txt              每 dim：平均名次 + Friedman p + Nemenyi CD
                                    及 CD 意义下的显著对列表
  conv_curves.npz                   key "cec2022_F{02d}_D{d}__{algo}" → 30 runs
                                    平均收敛曲线（visualization.py 约定）

统计口径（R7）：Wilcoxon 符号秩（成对，run_id 配对）；Friedman + Nemenyi
（k=5, q_0.05=2.728，CD = q·sqrt(k(k+1)/(6N_funcs))，按 12 函数平均名次）。
部分数据（n<30）可算但标注。

用法：python3 stats_formal.py [--results DIR]
依赖：numpy、scipy。
"""

import os
import csv
import argparse
from collections import defaultdict

import numpy as np
from scipy import stats as sst


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


# Nemenyi q_alpha (alpha=0.05, two-tailed)，与 visualization.py 同源
Q_ALPHA_05 = {2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
FLAGSHIPS = ["v1.8", "ADSCAv2.1"]


def load_raw(path):
    rows = []
    with open(path, "r") as fp:
        for r in csv.DictReader(fp):
            rows.append(dict(algo=r["algo"], func_id=int(r["func_id"]), dim=int(r["dim"]), run_id=int(r["run_id"]),
                             error=float(r["error"]), best_fitness=float(r["best_fitness"]),
                             f_global=float(r["f_global"])))
    return rows


def summarize(rows, results_dir):
    cells = defaultdict(list)
    for r in rows:
        cells[(r["func_id"], r["dim"], r["algo"])].append(r)
    fg = {}
    for (fid, d, a), rs in cells.items():
        fg[(fid, d, a)] = rs[0]["f_global"]
    
    full_path = os.path.join(results_dir, "cec2022_stats_full.csv")
    sum_path = os.path.join(results_dir, "cec2022_summary_by_function.csv")
    with open(full_path, "w", newline="") as fp, open(sum_path, "w", newline="") as fs:
        w = csv.writer(fp)
        w.writerow(["func_id", "dim", "algo", "n", "mean", "std", "best", "worst", "median"])
        ws = csv.writer(fs)
        ws.writerow(["func_id", "dim", "algo", "f_global", "mean_error"])
        for (fid, d, a) in sorted(cells):
            errs = np.array([r["error"] for r in cells[(fid, d, a)]])
            w.writerow([fid, d, a, len(errs), "{:.6e}".format(errs.mean()),
                        "{:.6e}".format(errs.std(ddof=1) if len(errs) > 1 else 0.0), "{:.6e}".format(errs.min()),
                        "{:.6e}".format(errs.max()), "{:.6e}".format(np.median(errs))])
            ws.writerow([fid, d, a, "{:.10e}".format(fg[(fid, d, a)]), "{:.6e}".format(errs.mean())])
    print("[stats] {} / {}".format(full_path, sum_path))
    return cells


def wilcoxon_table(rows, results_dir):
    by = defaultdict(dict)  # (func,dim,algo) -> {run_id: error}
    for r in rows:
        by[(r["func_id"], r["dim"], r["algo"])][r["run_id"]] = r["error"]
    out_path = os.path.join(results_dir, "cec2022_wilcoxon.csv")
    with open(out_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(
            ["func_id", "dim", "flagship", "opponent", "n_pairs", "median_flag", "median_opp", "p_value", "verdict"])
        for (fid, d, flag) in sorted(by):
            if flag not in FLAGSHIPS:
                continue
            algos = sorted({a for (f2, d2, a) in by if f2 == fid and d2 == d})
            for opp in algos:
                if opp == flag:
                    continue
                ra, rb = by[(fid, d, flag)], by[(fid, d, opp)]
                common = sorted(set(ra) & set(rb))
                if not common:
                    continue
                xa = np.array([ra[i] for i in common])
                xb = np.array([rb[i] for i in common])
                diff = xa - xb
                if np.all(diff == 0):
                    p, verdict = float("nan"), "="
                else:
                    p = float(sst.wilcoxon(xa, xb)[1])  # [1] 索引：兼容新旧 scipy 返回类型
                    ma, mb = np.median(xa), np.median(xb)
                    verdict = ("+" if ma < mb else ("-" if ma > mb else "="))
                    if p >= 0.05:
                        verdict = "="  # 不显著记平
                w.writerow(
                    [fid, d, flag, opp, len(common), "{:.6e}".format(np.median(xa)), "{:.6e}".format(np.median(xb)),
                     "{:.4e}".format(p), verdict])
    print("[stats] {}".format(out_path))


def friedman_nemenyi(rows, results_dir):
    cells = defaultdict(list)
    for r in rows:
        cells[(r["func_id"], r["dim"], r["algo"])].append(r["error"])
    dims = sorted({d for (_, d, _) in cells})
    lines = []
    for d in dims:
        funcs = sorted({f for (f, d2, _) in cells if d2 == d})
        algos = sorted({a for (_, d2, a) in cells if d2 == d})
        M = np.array([[np.mean(cells[(f, d, a)]) for a in algos] for f in funcs])
        ranks = np.array([sst.rankdata(row) for row in M])  # 每函数内名次（小优）
        avg = ranks.mean(axis=0)
        k, nfn = len(algos), len(funcs)
        stat, p = sst.friedmanchisquare(*[M[:, j] for j in range(k)])
        q = Q_ALPHA_05.get(k, 3.164)
        cd = q * np.sqrt(k * (k + 1) / (6.0 * nfn))
        lines.append(
            "D={}: Friedman chi2={:.3f} p={:.4e} | Nemenyi CD={:.3f} (k={}, N={})".format(d, stat, p, cd, k, nfn))
        order = np.argsort(avg)
        for j in order:
            lines.append("  {:<10s} avg_rank={:.3f}".format(algos[j], avg[j]))
        sig = []
        for i in range(k):
            for j in range(i + 1, k):
                if abs(avg[i] - avg[j]) > cd:
                    sig.append("{} vs {} (|Δ|={:.2f})".format(algos[i], algos[j], abs(avg[i] - avg[j])))
        lines.append("  CD 显著对: " + ("; ".join(sig) if sig else "无"))
    out_path = os.path.join(results_dir, "cec2022_friedman.txt")
    with open(out_path, "w") as fp:
        fp.write("\n".join(lines) + "\n")
    print("[stats] {}".format(out_path))
    for ln in lines:
        print("    " + ln)


def aggregate_curves(rows, results_dir):
    cells = defaultdict(list)  # (func,dim,algo) -> [curves]
    dims_funcs = set()
    for r in rows:
        dims_funcs.add((r["func_id"], r["dim"]))
    for (fid, d) in sorted(dims_funcs):
        path = os.path.join(results_dir, "curves_F{:02d}_D{:d}.npz".format(fid, d))
        if not os.path.exists(path):
            continue
        with np.load(path) as z:
            for k in z.files:
                algo = k.split("__run")[0]
                cells[(fid, d, algo)].append(z[k])
    store = {}
    for (fid, d, algo), curves in sorted(cells.items()):
        store["cec2022_F{:02d}_D{:d}__{}".format(fid, d, algo)] = np.mean(curves, axis=0)
    out_path = os.path.join(results_dir, "conv_curves.npz")
    np.savez(out_path, **store)
    print("[stats] {}（{} 条平均曲线）".format(out_path, len(store)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase4_results"))
    args = ap.parse_args()
    args.results = _norm_dir(args.results)
    raw_path = _resolve_csv(os.path.join(args.results, "cec2022_raw_results.csv"),
                            "E1 基准原始成绩（cec2022_raw_results.csv）")
    rows = load_raw(raw_path)
    n30 = len(rows)
    print("[stats] 载入 {} 行（{}）".format(n30, raw_path))
    summarize(rows, args.results)
    wilcoxon_table(rows, args.results)
    friedman_nemenyi(rows, args.results)
    aggregate_curves(rows, args.results)


if __name__ == "__main__":
    main()
