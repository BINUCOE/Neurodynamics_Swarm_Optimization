#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_e3_tsweep.py — E3 调度扫描正式批次驱动（实验配置规范 v3 §二）

全配置两旗舰 × T ∈ {25, 50, 200}（T=100 即 E1 已有格子，不重复跑）。
T 经 switchboard 构造参数传入，α=1/T、β=0.1/T 自动联动，其余 flag 与全配置一致。
呈现形态：剂量-响应曲线（横轴 T 对数刻度，纵轴 log10(error_T/error_100) 中位数）。

矩阵：2 算法 × 3 档 T × F1–F12 × D∈{10,20} × 30 runs = 4320 任务
预算/种子/终止口径与 E1 完全一致（FE 预算终止，退火地平线外设 budget//N，
seed = 666 + run_id + func_id*100 + dim*1000），与 E1 的 T=100 格子逐 run 配对。

落盘（results_dir，默认 ./e3_results/）：
  e3_tsweep_raw.csv       逐 run 追加（断点续跑锚点），字段 = E1 同构 + trig_count + T
  curves_F{02d}_D{d}.npz  每 (func,dim) 一个，key "{algo}__run{run_id:02d}"
  manifest.json           配置矩阵 + 代码 sha256 + 环境版本

用法：python3 run_e3_tsweep.py           # 正式全量（IDE 一键）
      python3 run_e3_tsweep.py --smoke   # 冒烟：F1，D=10/20，预算 500/300，2 runs
py3.7 兼容；依赖 numpy、opfunu 1.0.1、core/ 的 switchboard 与冻结族。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_HERE, os.pardir, os.pardir),
               os.path.join(_HERE, os.pardir, os.pardir, "core"), _HERE):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

import csv
import json
import time
import argparse
import hashlib

import numpy as np
import math as _math
if not hasattr(np, "math"):
    np.math = _math

import core.cec_schwefel_fix  # noqa: F401  Schwefel 补丁（必须先于 cec2022 使用）
from opfunu.cec_based import cec2022
from core.ad_switchboard import SB_PSO, SB_SCA

N = 50
FUNCS = list(range(1, 13))
BUDGET = {10: 200000, 20: 500000}
SEED_BASE = 666
N_GRID = 121
T_GRID = [25, 50, 200]  # T=100 为 E1 已有格子

FLAGSHIP_FLAGS = {
    "ADPSO": (SB_PSO, dict(topology="ring", channel=True, k_adaptive=True,
                           direction="mantegna")),
    "ADSCA": (SB_SCA, dict(reference="ti", topology="ring", channel=True,
                           boost=True, kick_timing="same_gen", clip_after_kick=True,
                           k_adaptive=True, c=0.7)),
}

RAW_CSV = "e3_tsweep_raw.csv"
CSV_FIELDS = ["algo", "T", "func_id", "dim", "run_id", "seed", "budget",
              "best_fitness", "error", "f_global", "fe", "gens",
              "trig_count", "runtime"]


def fe_grid(budget):
    g = np.unique(np.round(np.logspace(np.log10(N), np.log10(budget), N_GRID)).astype(int))
    g[-1] = budget
    return g


def run_task(task):
    algo_name, T, func_id, dim, run_id = task
    cls, flags = FLAGSHIP_FLAGS[algo_name]
    budget = BUDGET[dim]
    seed = SEED_BASE + run_id + func_id * 100 + dim * 1000
    np.random.seed(seed)
    f = getattr(cec2022, "F{:d}2022".format(func_id))(ndim=dim)
    bounds = (float(f.lb[0]), float(f.ub[0]))
    fg = float(f.f_global)
    t0 = time.perf_counter()

    algo = cls(D=dim, N=N, bounds=bounds,
               func=lambda x: float(f.evaluate(x)), T=T, **flags)
    algo.max_iter = budget // N  # 退火地平线外设（FE 口径）

    grid = fe_grid(budget)
    curve = np.full(len(grid), np.nan)
    k = 0
    curve[0] = float(algo.g_fit)
    gens = 0
    while algo.fe_count < budget:
        algo.step()
        gens += 1
        now = algo.fe_count
        while k + 1 < len(grid) and now >= grid[k + 1]:
            k += 1
            curve[k] = float(algo.g_fit)
    last = curve[0]
    for i in range(len(grid)):
        if np.isnan(curve[i]):
            curve[i] = last
        else:
            last = curve[i]

    row = dict(algo="{}_T{}".format(algo_name, T), T=T, func_id=func_id, dim=dim,
               run_id=run_id, seed=seed, budget=budget,
               best_fitness=float(algo.g_fit), error=abs(float(algo.g_fit) - fg),
               f_global=fg, fe=algo.fe_count, gens=gens,
               trig_count=algo.trig_count,
               runtime=round(time.perf_counter() - t0, 2))
    return row, curve


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fp:
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
        phase="E3 scheduling sweep (规范 v3)",
        algos=algos, T_grid=T_GRID, T100_ref="E1 格子（v1.8 / ADSCAv2.1）",
        funcs=funcs, dims=dims, runs=runs, N=N,
        budget=BUDGET, seed_base=SEED_BASE,
        seed_formula="base + run_id + func_id*100 + dim*1000",
        env=dict(python=sys.version.split()[0], numpy=np.__version__,
                 opfunu=getattr(opfunu, "__version__", "unknown"),
                 patch="cec_schwefel_fix loaded"),
        code_sha256=snap,
        started=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    with open(os.path.join(results_dir, "manifest.json"), "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(_HERE, "e3_results"))
    ap.add_argument("--dims", default="10,20")
    ap.add_argument("--funcs", default=",".join(str(f) for f in FUNCS))
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟：F1，D=10 预算500 / D=20 预算300，2 runs，单进程")
    args = ap.parse_args()

    global BUDGET
    if args.smoke:
        BUDGET = {10: 500, 20: 300}
        args.funcs, args.runs, args.workers = "1", 2, 1

    algos = ["{}_T{}".format(a, T) for a in FLAGSHIP_FLAGS for T in T_GRID]
    dims = [int(d) for d in args.dims.split(",") if d]
    funcs = [int(f) for f in args.funcs.split(",") if f]
    os.makedirs(args.results, exist_ok=True)
    raw_path = os.path.join(args.results, RAW_CSV)

    tasks = []
    for a in FLAGSHIP_FLAGS:
        for T in T_GRID:
            for fid in funcs:
                for d in dims:
                    for r in range(args.runs):
                        tasks.append((a, T, fid, d, r))
    done = load_done(raw_path)
    todo = [t for t in tasks
            if ("{}_T{}".format(t[0], t[1]), t[2], t[3], t[4]) not in done]
    print("[E3] 配置 {} | 总任务 {} | 已完成 {} | 待跑 {}".format(
        len(algos), len(tasks), len(done), len(todo)), flush=True)
    if not todo:
        return
    write_manifest(args.results, algos, dims, args.runs, funcs)

    write_header = not os.path.exists(raw_path)
    fp = open(raw_path, "a", newline="", encoding="utf-8")
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
                print("[E3] {}/{}（累计 {}/{}） 已用 {:.0f}s 预计剩余 {:.0f}s".format(
                    i, len(todo), n0 + i, len(tasks), el, eta), flush=True)
    fp.close()
    print("[E3] 完成 -> {}".format(raw_path), flush=True)


if __name__ == "__main__":
    main()
