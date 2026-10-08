#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pso.py — 原生 Standard PSO（基线算法资产）

全局最优（gbest）拓扑、固定惯性权重 w=0.7、c1=c2=1.4——
论文受控对比中的 PSO 侧裸内核基线（无慢通道、无环拓扑）。

来源与一致性：类体逐字拷贝自冻结实现 pso_family.StandardPSO
（仅类名改为 PSO，并附冻结名别名）；与 switchboard "PSO" 预设的
bit 一致性由 tests/test_bit_identity.py 锚定。

用法：
    import numpy as np
    from pso import PSO
    np.random.seed(42)
    algo = PSO(D=10, N=50, bounds=(-5.12, 5.12), func=rastrigin)
    best_x, best_f, conv = algo.run(max_iter=1000)

仅依赖 numpy。py3.7+。
"""

import numpy as np


# ---- 以下为冻结同源代码（pso_family.py 逐字拷贝，勿动） ----

class PSO:
    def __init__(self, D, N, bounds, func, w=0.7, c1=1.4, c2=1.4):
        self.D, self.N, self.func = D, N, func
        self.lb, self.ub = bounds
        self.w, self.c1, self.c2 = w, c1, c2
        span = self.ub - self.lb
        self.pos = np.random.uniform(self.lb, self.ub, (N, D))
        self.vel = np.random.uniform(-span * 0.1, span * 0.1, (N, D))
        self.pb_pos = self.pos.copy()
        self.pb_fit = np.array([func(x) for x in self.pos])
        idx = np.argmin(self.pb_fit)
        self.g_pos, self.g_fit = self.pb_pos[idx].copy(), self.pb_fit[idx]

    def step(self):
        r1, r2 = np.random.random((self.N, self.D)), np.random.random((self.N, self.D))
        self.vel = self.w * self.vel + self.c1 * r1 * (self.pb_pos - self.pos) + self.c2 * r2 * (self.g_pos - self.pos)
        self.pos = np.clip(self.pos + self.vel, self.lb, self.ub)
        fit = np.array([self.func(x) for x in self.pos])
        imp = fit < self.pb_fit
        self.pb_pos[imp], self.pb_fit[imp] = self.pos[imp].copy(), fit[imp]
        bi = np.argmin(fit)
        if fit[bi] < self.g_fit:
            self.g_pos, self.g_fit = self.pos[bi].copy(), fit[bi]
        return self.g_fit

    def run(self, max_iter):
        conv = [self.g_fit]
        for _ in range(max_iter):
            conv.append(self.step())
        return self.g_pos.copy(), self.g_fit, conv


StandardPSO = PSO  # 冻结名别名（回归门 / 旧脚本兼容）


def _demo():
    def rastrigin(x):
        return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

    np.random.seed(42)
    algo = PSO(D=10, N=50, bounds=(-5.12, 5.12), func=rastrigin)
    _, g, _ = algo.run(1000)
    print("PSO demo (Rastrigin D=10, seed=42): g = {:.6e}".format(g))


if __name__ == "__main__":
    _demo()
