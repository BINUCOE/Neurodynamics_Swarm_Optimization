#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pso_wdecay.py — 惯性权重线性衰减 PSO（PSO-wDecay，简单变体基线）

gbest 拓扑 + w 自 w_max=0.9 至 w_min=0.4 随 t/max_iter 线性退火——
论文阵容中的"经典改进 PSO"代表（单参数调度，无慢通道）。
注意：退火地平线 max_iter 由 run() 外设（与 FE 预算挂钩），
这是冻结口径的一部分，勿改。

来源与一致性：类体逐字拷贝自冻结实现 pso_family.PSO_wDecay；
与冻结类的 bit 一致性由 tests/test_bit_identity.py 锚定。

用法：
    import numpy as np
    from pso_wdecay import PSO_wDecay
    np.random.seed(42)
    algo = PSO_wDecay(D=10, N=50, bounds=(-5.12, 5.12), func=rastrigin)
    best_x, best_f, conv = algo.run(max_iter=1000)

仅依赖 numpy。py3.7+。
"""

import numpy as np


# ---- 以下为冻结同源代码（pso_family.py 逐字拷贝，勿动） ----

class PSO_wDecay:
    def __init__(self, D, N, bounds, func, w_max=0.9, w_min=0.4, c1=1.4, c2=1.4):
        self.D, self.N, self.func = D, N, func
        self.lb, self.ub = bounds
        self.w_max, self.w_min, self.c1, self.c2 = w_max, w_min, c1, c2
        span = self.ub - self.lb
        self.pos = np.random.uniform(self.lb, self.ub, (N, D))
        self.vel = np.random.uniform(-span * 0.1, span * 0.1, (N, D))
        self.pb_pos = self.pos.copy()
        self.pb_fit = np.array([func(x) for x in self.pos])
        idx = np.argmin(self.pb_fit)
        self.g_pos, self.g_fit = self.pb_pos[idx].copy(), self.pb_fit[idx]
        self.t = 0
        self.max_iter = 1000  # Phase 1 修复：退火地平线由 harness 按 FE 预算外设（原为 /1000 硬编码）

    def step(self):
        w = self.w_max - (self.w_max - self.w_min) * self.t / self.max_iter
        r1, r2 = np.random.random((self.N, self.D)), np.random.random((self.N, self.D))
        self.vel = w * self.vel + self.c1 * r1 * (self.pb_pos - self.pos) + self.c2 * r2 * (self.g_pos - self.pos)
        self.pos = np.clip(self.pos + self.vel, self.lb, self.ub)
        fit = np.array([self.func(x) for x in self.pos])
        imp = fit < self.pb_fit
        self.pb_pos[imp], self.pb_fit[imp] = self.pos[imp].copy(), fit[imp]
        bi = np.argmin(fit)
        if fit[bi] < self.g_fit:
            self.g_pos, self.g_fit = self.pos[bi].copy(), fit[bi]
        self.t += 1
        return self.g_fit

    def run(self, max_iter):
        self.max_iter = max_iter
        conv = [self.g_fit]
        for _ in range(max_iter):
            conv.append(self.step())
        return self.g_pos.copy(), self.g_fit, conv


def _demo():
    def rastrigin(x):
        return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

    np.random.seed(42)
    algo = PSO_wDecay(D=10, N=50, bounds=(-5.12, 5.12), func=rastrigin)
    _, g, _ = algo.run(1000)
    print("PSO-wDecay demo (Rastrigin D=10, seed=42): g = {:.6e}".format(g))


if __name__ == "__main__":
    _demo()
