#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
instrument_e4.py — E4 全基准插桩（实验配置规范 v3 §三）

全配置两旗舰 × F1–F12 × D∈{10,20} × 3 seeds（run_id 0,1,2，与 E1 同种子公式配对）
= 144 次插桩运行。预算/种子/终止口径与 E1 一致；退火地平线显式外设 budget//N
（历史教训：不显式设置会使观测量偏差，本脚本在 run_task 内硬设置）。

观测量（不改变算法行为与 RNG 消耗，纯外部读取）：
  逐代：fe, mean(A), K, 群体散布, 触发粒子数, 逐粒子 A 矩阵 A_full（gens×N, float32）
  逐事件：触发代号, 粒子索引, 踢幅度（A×ds，ds=1）
  触发事件的外部检测：A 仅经复位归零（漏电衰减永不精确到 0），
    故 mask = (A_prev > 0) & (A_now == 0)；事件计数须与 algo.trig_count
    逐 run 对账（不一致即报错，对账纪律沿用）。
派生量（写入 e4_summary.csv）：
  dwell_p10/p50/p90   同粒子相邻触发的间隔（代），舍弃末端删失
  fano_w100           100 代窗触发计数的 var/mean
  fano_gen            逐代触发计数的 var/mean
  kick_mag_min/max    踢幅度（理论界 (1.0, 1.01]）
  kick_spread_p50     踢幅度 / 当代群体散布 中位
  prop_delay_p50      环上传播延迟：触发粒子 i±1（mod N）的下一次触发间隔中位（上限 10 代）
  corr_K_meanA        K 与 mean(A) 的 Pearson 相关
  boost_hits          振幅调制激活粒子次（预期恒 0）

落盘（results_dir，默认 ./e4_results/）：
  e4_obs_{algo}_F{02d}_D{d}_r{rid}.npz   逐 run 原始序列
  e4_summary.csv                         逐 run 派生量汇总（断点续跑锚点）
  manifest.json                          配置矩阵 + 代码 sha256 + 环境版本

用法：python3 instrument_e4.py           # 正式全量（IDE 一键）
      python3 instrument_e4.py --smoke   # 冒烟：F1，D=10/20，预算 500/300，2 runs
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
RUN_IDS = [0, 1, 2]
PROP_CAP = 10

ALGOS = {
    "ADPSO": (SB_PSO, dict(topology="ring", channel=True, k_adaptive=True,
                           direction="mantegna")),
    "ADSCA": (SB_SCA, dict(reference="ti", topology="ring", channel=True,
                           boost=True, kick_timing="same_gen", clip_after_kick=True,
                           k_adaptive=True, c=0.7)),
}

RAW_CSV = "e4_summary.csv"
CSV_FIELDS = ["algo", "func_id", "dim", "run_id", "seed", "budget",
              "fe", "gens", "trig_count", "boost_hits",
              "best_fitness", "error",
              "dwell_p10", "dwell_p50", "dwell_p90",
              "fano_w100", "fano_gen",
              "kick_mag_min", "kick_mag_max", "kick_spread_p50",
              "prop_delay_p50", "corr_K_meanA", "runtime"]


def run_task(task):
    algo_name, func_id, dim, run_id = task
    cls, flags = ALGOS[algo_name]
    budget = BUDGET[dim]
    seed = SEED_BASE + run_id + func_id * 100 + dim * 1000
    np.random.seed(seed)
    f = getattr(cec2022, "F{:d}2022".format(func_id))(ndim=dim)
    bounds = (float(f.lb[0]), float(f.ub[0]))
    fg = float(f.f_global)
    t0 = time.perf_counter()

    algo = cls(D=dim, N=N, bounds=bounds, func=lambda x: float(f.evaluate(x)), **flags)
    algo.max_iter = budget // N  # 退火地平线显式外设（历史教训，勿删）

    is_pso = algo_name == "ADPSO"
    get_A = lambda: (algo.A if is_pso else algo.A_drift).copy()

    gens = 0
    g_fe, g_meanA, g_K, g_spread, g_ntrig = [], [], [], [], []
    A_rows = []  # 逐代逐粒子 A（方案 B 场热图用；只读快照，不影响算法）
    ev_gen, ev_pid, ev_mag = [], [], []
    A_prev = get_A()
    mag_cursor = 0  # kick_mags 游标：逐代按粒子索引序消费（vec/loop 均为索引序追加）

    while algo.fe_count < budget:
        algo.step()
        gens += 1
        A_now = get_A()
        # 触发外部检测：A 仅经复位精确归零
        mask = (A_prev > 0.0) & (A_now == 0.0)
        pids = np.nonzero(mask)[0]
        nd = len(pids)
        mags = algo.kick_mags[mag_cursor:mag_cursor + nd]
        if len(mags) != nd:
            raise RuntimeError("踢幅度游标失步：{} vs {}".format(len(mags), nd))
        mag_cursor += nd
        for p, m in zip(pids.tolist(), mags):
            ev_gen.append(gens)
            ev_pid.append(int(p))
            ev_mag.append(float(m))
        centroid = algo.pos.mean(axis=0)
        spread = float(np.linalg.norm(algo.pos - centroid, axis=1).mean())
        g_fe.append(algo.fe_count)
        g_meanA.append(float(A_now.mean()))
        g_K.append(int(algo.K))
        g_spread.append(spread)
        g_ntrig.append(nd)
        A_rows.append(A_now)
        A_prev = A_now

    # 对账纪律：外部检测事件数 == 算法自报触发数 == kick_mags 长度
    if not (len(ev_gen) == algo.trig_count == mag_cursor):
        raise RuntimeError("触发对账失败：detected={} trig_count={} mags={}".format(
            len(ev_gen), algo.trig_count, mag_cursor))

    ev_gen = np.asarray(ev_gen)
    ev_pid = np.asarray(ev_pid)
    ev_mag = np.asarray(ev_mag, dtype=float)
    g_ntrig = np.asarray(g_ntrig, dtype=float)
    g_K = np.asarray(g_K, dtype=float)
    g_meanA = np.asarray(g_meanA, dtype=float)
    g_spread = np.asarray(g_spread, dtype=float)

    # 驻留：同粒子相邻触发间隔（舍弃末端删失）
    dwells = []
    for p in range(N):
        gs = np.sort(ev_gen[ev_pid == p])
        if len(gs) >= 2:
            dwells.extend(np.diff(gs).tolist())
    dwells = np.asarray(dwells, dtype=float)

    # Fano：100 代窗计数 + 逐代计数
    def fano(counts):
        m = counts.mean()
        return float(counts.var() / m) if m > 0 else float("nan")
    if gens >= 100:
        w = counts_w = g_ntrig[:gens // 100 * 100].reshape(-1, 100).sum(axis=1)
        fano_w100 = fano(w)
    else:
        fano_w100 = float("nan")
    fano_gen = fano(g_ntrig)

    # 踢幅/散布
    ks = ev_mag / g_spread[ev_gen - 1] if len(ev_gen) else np.asarray([])

    # 环上传播延迟：粒子 i 触发后，邻居 i±1（mod N）下次触发的间隔
    delays = []
    ev_by_particle = {p: np.sort(ev_gen[ev_pid == p]) for p in range(N)}
    for g, p in zip(ev_gen.tolist(), ev_pid.tolist()):
        best = None
        for q in ((p - 1) % N, (p + 1) % N):
            later = ev_by_particle[q][ev_by_particle[q] > g]
            if len(later):
                d = int(later[0] - g)
                if best is None or d < best:
                    best = d
        if best is not None and best <= PROP_CAP:
            delays.append(best)

    corr = float("nan")
    if gens > 2 and g_K.std() > 0 and g_meanA.std() > 0:
        corr = float(np.corrcoef(g_K, g_meanA)[0, 1])

    row = dict(algo=algo_name, func_id=func_id, dim=dim, run_id=run_id,
               seed=seed, budget=budget, fe=algo.fe_count, gens=gens,
               trig_count=algo.trig_count,
               boost_hits=int(getattr(algo, "boost_hits", 0)),
               best_fitness=float(algo.g_fit), error=abs(float(algo.g_fit) - fg),
               dwell_p10=float(np.percentile(dwells, 10)) if len(dwells) else float("nan"),
               dwell_p50=float(np.median(dwells)) if len(dwells) else float("nan"),
               dwell_p90=float(np.percentile(dwells, 90)) if len(dwells) else float("nan"),
               fano_w100=fano_w100, fano_gen=fano_gen,
               kick_mag_min=float(ev_mag.min()) if len(ev_mag) else float("nan"),
               kick_mag_max=float(ev_mag.max()) if len(ev_mag) else float("nan"),
               kick_spread_p50=float(np.median(ks)) if len(ks) else float("nan"),
               prop_delay_p50=float(np.median(delays)) if delays else float("nan"),
               corr_K_meanA=corr,
               runtime=round(time.perf_counter() - t0, 2))

    arrays = dict(gen=np.arange(1, gens + 1), fe=np.asarray(g_fe),
                  meanA=g_meanA, K=g_K, spread=g_spread, ntrig=g_ntrig,
                  A_full=np.stack(A_rows).astype(np.float32),
                  ev_gen=ev_gen, ev_pid=ev_pid, ev_mag=ev_mag)
    return row, arrays


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fp:
            for r in csv.DictReader(fp):
                done.add((r["algo"], int(r["func_id"]), int(r["dim"]), int(r["run_id"])))
    return done


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        h.update(fp.read())
    return h.hexdigest()


def write_manifest(results_dir, dims, runs, funcs):
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
        phase="E4 full-benchmark instrumentation (规范 v3)",
        algos=list(ALGOS), funcs=funcs, dims=dims, run_ids=runs, N=N,
        budget=BUDGET, seed_base=SEED_BASE,
        seed_formula="base + run_id + func_id*100 + dim*1000",
        detection="trigger mask = (A_prev > 0) & (A_now == 0)；逐 run 三方对账",
        prop_cap=PROP_CAP,
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
    ap.add_argument("--results", default=os.path.join(_HERE, "e4_results"))
    ap.add_argument("--dims", default="10,20")
    ap.add_argument("--funcs", default=",".join(str(f) for f in FUNCS))
    ap.add_argument("--runs", type=int, default=len(RUN_IDS),
                    help="run_id 0..runs-1（默认 3，与 E1 前三个种子配对）")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟：F1，D=10 预算500 / D=20 预算300，2 runs，单进程")
    args = ap.parse_args()

    global BUDGET
    if args.smoke:
        BUDGET = {10: 500, 20: 300}
        args.funcs, args.runs, args.workers = "1", 2, 1

    dims = [int(d) for d in args.dims.split(",") if d]
    funcs = [int(f) for f in args.funcs.split(",") if f]
    run_ids = list(range(args.runs))
    os.makedirs(args.results, exist_ok=True)
    raw_path = os.path.join(args.results, RAW_CSV)

    tasks = [(a, fid, d, r) for a in ALGOS for fid in funcs for d in dims
             for r in run_ids]
    done = load_done(raw_path)

    # 陈旧格检测：csv 标完成但 npz 缺 A_full（v2 新增的逐粒子 A 矩阵）→ 重跑并改写 csv
    def _npz_path(t):
        a, fid, d, r = t
        return os.path.join(args.results, "e4_obs_{}_F{:02d}_D{}_r{}.npz".format(a, fid, d, r))

    def _has_afull(t):
        p = _npz_path(t)
        if not os.path.exists(p):
            return False
        try:
            with np.load(p) as z:
                return "A_full" in z.files
        except Exception:
            return False

    stale = {t for t in done if not _has_afull(t)}
    if stale:
        kept = []
        with open(raw_path, "r", encoding="utf-8") as fp0:
            for r0 in csv.DictReader(fp0):
                t0 = (r0["algo"], int(r0["func_id"]), int(r0["dim"]), int(r0["run_id"]))
                if t0 not in stale:
                    kept.append(r0)
        with open(raw_path, "w", newline="", encoding="utf-8") as fp0:
            w0 = csv.DictWriter(fp0, fieldnames=CSV_FIELDS)
            w0.writeheader()
            w0.writerows(kept)
        done = done - stale
        print("[E4] 检测到 {} 个缺 A_full 的陈旧格，已从 csv 摘除并重排".format(len(stale)),
              flush=True)

    todo = [t for t in tasks if t not in done]
    print("[E4] 总任务 {} | 已完成 {} | 待跑 {}".format(
        len(tasks), len(done), len(todo)), flush=True)
    if not todo:
        return
    write_manifest(args.results, dims, run_ids, funcs)

    write_header = not os.path.exists(raw_path)
    fp = open(raw_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()
    from multiprocessing import Pool
    t_start = time.perf_counter()
    with Pool(processes=args.workers) as pool:
        for i, (row, arrays) in enumerate(pool.imap_unordered(run_task, todo), 1):
            writer.writerow(row)
            fp.flush()
            np.savez(os.path.join(
                args.results, "e4_obs_{}_F{:02d}_D{}_r{}.npz".format(
                    row["algo"], row["func_id"], row["dim"], row["run_id"])), **arrays)
            if i % 5 == 0 or i == len(todo):
                el = time.perf_counter() - t_start
                eta = el / i * (len(todo) - i)
                print("[E4] {}/{} 已用 {:.0f}s 预计剩余 {:.0f}s".format(
                    i, len(todo), el, eta), flush=True)
    fp.close()
    print("[E4] 完成 -> {}".format(raw_path), flush=True)


if __name__ == "__main__":
    main()
