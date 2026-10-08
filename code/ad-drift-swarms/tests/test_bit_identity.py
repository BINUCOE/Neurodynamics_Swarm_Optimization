#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_bit_identity.py — 三方 bit 一致性锚定（开源组织防漂移门）

对五个算法资产逐一同种子核对三条来源的**逐位一致**：
  ① 纯净类（algorithms/，本仓库第一层，供读者直接取用）
  ② 冻结类（core/pso_family.py / core/sca_family.py，历史冻结实现）
  ③ switchboard 旗舰/基线预设（core/ad_switchboard.py，谱系消融载体）

四重相等：收敛曲线、终位置、终适应度、FE 评估计数。
任何一方被意外改动（代码漂移），本门即 FAIL。

用法：python tests/test_bit_identity.py   （从仓库任意目录运行均可）
退出码：0 = 全部 PASS；1 = 存在 FAIL。
"""

import os
import sys

import numpy as np

# NumPy 2.x 兼容（冻结 levy_batch 使用 np.math）
import math as _math
if not hasattr(np, "math"):
    np.math = _math

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "algorithms"), os.path.join(_ROOT, "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ① 纯净类
from ad_pso import ADPSO
from ad_sca import ADSCA
from pso import PSO
from sca import SCA
from pso_wdecay import PSO_wDecay as PSO_wDecay_clean
# ② 冻结类
from pso_family import StandardPSO, ADPSOv18, PSO_wDecay as PSO_wDecay_frozen, rastrigin
from sca_family import StandardSCA, ADSCAv21
# ③ switchboard 预设
from ad_switchboard import make

D, N, ITERS, B = 10, 50, 200, (-5.12, 5.12)
SEEDS = [42, 7, 2024]

# (标签, 纯净类, 纯净kwargs, 冻结类, 冻结kwargs, switchboard预设或None)
TRIPLE = [
    ("AD-PSO v1.8", ADPSO, {"K_min": 5, "K_max": 7},
     ADPSOv18, {"K_min": 5, "K_max": 7}, "v1.8"),
    ("AD-SCA v2.1", ADSCA, {"c": 0.7, "K_min": 5, "K_max": 7},
     ADSCAv21, {"c": 0.7, "K_min": 5, "K_max": 7}, "ADSCAv2.1"),
    ("PSO",         PSO, {}, StandardPSO, {}, "PSO"),
    ("SCA",         SCA, {}, StandardSCA, {}, "SCA"),
    ("PSO-wDecay",  PSO_wDecay_clean, {}, PSO_wDecay_frozen, {}, None),
]


def _counted(func):
    def f(x):
        f.n += 1
        return func(x)
    f.n = 0
    return f


def _run(cls, kw, seed):
    np.random.seed(seed)
    f = _counted(rastrigin)
    algo = cls(D=D, N=N, bounds=B, func=f, **kw)
    _, g, conv = algo.run(ITERS)
    return np.asarray(conv, dtype=float), algo.pos.copy(), float(g), f.n


def _run_sb(preset, seed):
    np.random.seed(seed)
    algo = make(preset, D=D, N=N, bounds=B, func=rastrigin)
    _, g, conv = algo.run(ITERS)
    return np.asarray(conv, dtype=float), algo.pos.copy(), float(g), algo.fe_count


def _eq(a, b):
    return (np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
            and a[2] == b[2] and a[3] == b[3])


def main():
    print("=" * 78)
    print("  三方 bit 一致性锚定：纯净类 ≡ 冻结类 ≡ switchboard 预设")
    print("  D={} N={} iters={} func=Rastrigin seeds={}".format(D, N, ITERS, SEEDS))
    print("=" * 78)
    all_pass = True
    for label, cls_c, kw_c, cls_f, kw_f, preset in TRIPLE:
        for s in SEEDS:
            rc = _run(cls_c, kw_c, s)
            rf = _run(cls_f, kw_f, s)
            ok_cf = _eq(rc, rf)
            if preset is not None:
                rs = _run_sb(preset, s)
                ok = ok_cf and _eq(rc, rs)
                rel = "纯净≡冻结≡总线"
            else:
                ok = ok_cf
                rel = "纯净≡冻结（总线无此节点）"
            all_pass &= ok
            print("  {:<12s} seed={:<5d} {} 逐位一致: {}".format(
                label, s, rel, "PASS" if ok else "FAIL"))
    print("-" * 78)
    print("  总判定: {}".format("PASS —— 三层实现同源无漂移"
          if all_pass else "FAIL —— 存在代码漂移，禁止发布"))
    return all_pass


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
