#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_formal_cec2022.py — Phase 4 正式批次驱动（备忘录 v1.4 第五部分，R2 口径）

阵容（paper 五人组）：PSO / wDecay / SCA / v1.8 / ADSCAv2.1
  - PSO / SCA / v1.8 / ADSCAv2.1 由 ad_switchboard 预设承载（旗舰与冻结类逐位一致，
    回归门槛 21 PASS；wDecay 直接用 pso_family 冻结类 + FE 计数包装）
矩阵：5 算法 × F1–F12 × D∈{10,20} × 30 runs = 3600 任务
预算：D=10 → 200,000 FE；D=20 → 500,000 FE（FE 预算终止，退火地平线 = budget//N 外设）
种子：seed = 666 + run_id + func_id*100 + dim*1000（跨算法配对）

落盘（results_dir，默认 ./phase4_results/）——与 visualization.py loader 约定兼容：
  cec2022_raw_results.csv     逐 run 追加（断点续跑锚点）：algo, func_id, dim, run_id,
                              seed, budget, best_fitness, error, f_global, fe, gens, runtime
  curves_F{02d}_D{d}.npz      每 (func,dim) 一个：key "{algo}__run{run_id:02d}" →
                              FE 索引收敛曲线（121 点对数网格，best_fitness 原始值）
  manifest.json               配置矩阵 + 代码 sha256 快照 + 环境版本 + 起止时间（R9）

后续：stats_formal.py 汇总统计表 / Wilcoxon / Friedman+Nemenyi / conv_curves.npz。

用法：python3 run_formal_cec2022.py [--results DIR] [--algos a,b] [--dims 10] [--runs 30]
py3.7 兼容；依赖 numpy、opfunu 1.0.1、pso_family.py / sca_family.py /
cec_schwefel_fix.py / ad_switchboard.py（同目录或 PYTHONPATH）。
"""

import os
import sys
import csv

# 仓库布局兼容：平铺（本目录）优先，其次 <repo>/core/（两层开源组织）
_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_HERE, os.pardir, os.pardir),
               os.path.join(_HERE, os.pardir, os.pardir, "core"), _HERE):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

import json
import time
import argparse
import hashlib

import numpy as np
import math as _math
if not hasattr(np, "math"):
    np.math = _math

# 注：模块搜索路径由文件顶部引导段设置（平铺=本目录，仓库=<repo>/core/），
# 不含任何机器相关的硬编码路径。

import core.cec_schwefel_fix  # noqa: F401  Schwefel 补丁（必须先于 cec2022 使用）
from opfunu.cec_based import cec2022
from core.ad_switchboard import make
from core.pso_family import PSO_wDecay

N = 50
FUNCS = list(range(1, 13))
BUDGET = {10: 200000, 20: 500000}
SEED_BASE = 666
ALGOS_ALL = ["PSO", "wDecay", "SCA", "v1.8", "ADSCAv2.1"]
N_GRID = 121  # FE 对数网格点数（收敛曲线降采样）

RAW_CSV = "cec2022_raw_results.csv"
CSV_FIELDS = ["algo", "func_id", "dim", "run_id", "seed", "budget",
              "best_fitness", "error", "f_global", "fe", "gens", "runtime"]


def fe_grid(budget):
    g = np.unique(np.round(np.logspace(np.log10(N), np.log10(budget), N_GRID)).astype(int))
    g[-1] = budget
    return g


def run_task(task):
    algo_name, func_id, dim, run_id = task
    budget = BUDGET[dim]
    seed = SEED_BASE + run_id + func_id * 100 + dim * 1000
    np.random.seed(seed)
    f = getattr(cec2022, "F{:d}2022".format(func_id))(ndim=dim)
    bounds = (float(f.lb[0]), float(f.ub[0]))
    fg = float(f.f_global)
    t0 = time.perf_counter()

    if algo_name == "wDecay":
        def cnt(x):
            cnt.n += 1
            return float(f.evaluate(x))
        cnt.n = 0
        algo = PSO_wDecay(D=dim, N=N, bounds=bounds, func=cnt)
        fe = lambda: cnt.n
    else:
        algo = make(algo_name, D=dim, N=N, bounds=bounds,
                    func=lambda x: float(f.evaluate(x)))
        fe = lambda: algo.fe_count
    algo.max_iter = budget // N  # 退火地平线外设（FE 口径）

    grid = fe_grid(budget)
    curve = np.full(len(grid), np.nan)
    k = 0
    curve[0] = float(algo.g_fit)  # 初始化后（fe = N）
    gens = 0
    while fe() < budget:
        algo.step()
        gens += 1
        now = fe()
        while k + 1 < len(grid) and now >= grid[k + 1]:
            k += 1
            curve[k] = float(algo.g_fit)
    # 尾部前向填充（FE 超出预算的末代）
    last = curve[0]
    for i in range(len(curve)):
        if np.isnan(curve[i]):
            curve[i] = last
        else:
            last = curve[i]

    row = dict(algo=algo_name, func_id=func_id, dim=dim, run_id=run_id,
               seed=seed, budget=budget,
               best_fitness=float(algo.g_fit), error=abs(float(algo.g_fit) - fg),
               f_global=fg, fe=fe(), gens=gens,
               runtime=round(time.perf_counter() - t0, 2))
    return row, curve


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, "r") as fp:
            for r in csv.DictReader(fp):
                done.add((r["algo"], int(r["func_id"]), int(r["dim"]), int(r["run_id"])))
    return done


def update_curve_npz(results_dir, func_id, dim, algo, run_id, curve):
    path = os.path.join(results_dir, "curves_F{:02d}_D{:d}.npz".format(func_id, dim))
    store = {}
    if os.path.exists(path):
        with np.load(path) as z:
            store = {k: z[k] for k in z.files}
    store["{}__run{:02d}".format(algo, run_id)] = curve
    np.savez(path, **store)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        h.update(fp.read())
    return h.hexdigest()


def write_manifest(results_dir, algos, dims, runs, funcs):
    # 哈希"实际加载的模块"（__file__），而非按路径猜——记录的是真正跑起来的代码
    snap = {os.path.basename(__file__): sha256_of(os.path.abspath(__file__))}
    import importlib
    for name in ("ad_switchboard", "pso_family", "sca_family", "cec_schwefel_fix"):
        try:
            m = importlib.import_module(name)
            snap[name + ".py"] = sha256_of(m.__file__)
        except Exception:
            pass
    import opfunu
    manifest = dict(
        phase="Phase 4 formal CEC2022",
        algos=algos, funcs=funcs, dims=dims, runs=runs, N=N,
        budget=BUDGET, seed_base=SEED_BASE,
        seed_formula="base + run_id + func_id*100 + dim*1000",
        fe_grid_points=N_GRID,
        env=dict(python=sys.version.split()[0], numpy=np.__version__,
                 opfunu=getattr(opfunu, "__version__", "unknown"),
                 patch="cec_schwefel_fix loaded"),
        code_sha256=snap,
        started=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    with open(os.path.join(results_dir, "manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(_HERE, "phase4_results"))
    ap.add_argument("--algos", default=",".join(ALGOS_ALL))
    ap.add_argument("--dims", default="10,20")
    ap.add_argument("--funcs", default=",".join(str(f) for f in FUNCS))
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 4) - 2))  # 默认用满核（留 2 核），IDE 一键运行友好
    args = ap.parse_args()

    algos = [a for a in args.algos.split(",") if a]
    dims = [int(d) for d in args.dims.split(",") if d]
    funcs = [int(f) for f in args.funcs.split(",") if f]
    os.makedirs(args.results, exist_ok=True)
    raw_path = os.path.join(args.results, RAW_CSV)

    tasks = [(a, fid, d, r) for a in algos for fid in funcs for d in dims
             for r in range(args.runs)]
    done = load_done(raw_path)
    todo = [t for t in tasks if t not in done]
    print("[P4] 总任务 {} | 已完成 {} | 待跑 {}".format(len(tasks), len(done), len(todo)),
          flush=True)
    if not todo:
        return
    write_manifest(args.results, algos, dims, args.runs, funcs)

    write_header = not os.path.exists(raw_path)
    fp = open(raw_path, "a", newline="")
    writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()
    from multiprocessing import Pool
    t_start = time.perf_counter()
    n0 = len(done)
    with Pool(processes=args.workers) as pool:
        for i, (row, curve) in enumerate(pool.imap_unordered(run_task, todo), 1):
            writer.writerow(row)
            fp.flush()
            update_curve_npz(args.results, row["func_id"], row["dim"],
                             row["algo"], row["run_id"], curve)
            if i % 20 == 0 or i == len(todo):
                el = time.perf_counter() - t_start
                eta = el / i * (len(todo) - i)
                print("[P4] {}/{}（累计 {}/{}） 已用 {:.0f}s 预计剩余 {:.0f}s".format(
                    i, len(todo), n0 + i, len(tasks), el, eta), flush=True)
    fp.close()
    print("[P4] 完成 -> {}".format(raw_path), flush=True)


if __name__ == "__main__":
    main()
