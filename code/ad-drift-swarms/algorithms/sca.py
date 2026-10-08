#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sca.py — 原生 Standard SCA（基线算法资产）

Mirjalili (2016) 文献原味正弦余弦算法：gbest 引导、
参考项 |r2·x − g|（非平移等变形式，论文中作为"修复前"对照）、
幅度 a 自 a_max=2.0 线性退火、sin/cos 等概率抽签。
论文受控对比中的 SCA 侧裸内核基线（无慢通道、无环拓扑、无恢复项）。

来源与一致性：类体逐字拷贝自冻结实现 sca_family.StandardSCA
（仅类名改为 SCA，并附冻结名别名）；与 switchboard "SCA" 预设的
bit 一致性由 tests/test_bit_identity.py 锚定。

用法：
    import numpy as np
    from sca import SCA
    np.random.seed(42)
    algo = SCA(D=10, N=50, bounds=(-5.12, 5.12), func=rastrigin)
    best_x, best_f, conv = algo.run(max_iter=1000)

仅依赖 numpy。py3.7+。
"""

import numpy as np


# ---- 以下为冻结同源代码（sca_family.py 逐字拷贝，勿动） ----

class SCA:
    def __init__(self, D, N, bounds, func, a_max=2.0):
        self.D, self.N, self.func = D, N, func
        self.lb, self.ub = bounds
        self.a_max = a_max
        self.pos = np.random.uniform(self.lb, self.ub, (N, D))
        self.fit = np.array([func(x) for x in self.pos])
        self.pb_pos = self.pos.copy()
        self.pb_fit = self.fit.copy()
        idx = np.argmin(self.fit)
        self.g_pos, self.g_fit = self.pb_pos[idx].copy(), self.pb_fit[idx]
        self.t = 0
        self.max_iter = 1000

    def step(self):
        a = self.a_max * (1.0 - self.t / self.max_iter)
        for i in range(self.N):
            r1 = np.random.random() * 2.0 * np.pi
            r2 = np.random.random()
            r3 = np.random.random()
            term = np.abs(r2 * self.pos[i] - self.g_pos)
            if np.random.random() < 0.5:
                self.pos[i] += a * np.sin(r1) * term
            else:
                self.pos[i] += a * np.cos(r1) * term
        self.pos = np.clip(self.pos, self.lb, self.ub)
        self.fit = np.array([self.func(x) for x in self.pos])
        imp = self.fit < self.pb_fit
        self.pb_pos[imp] = self.pos[imp].copy()
        self.pb_fit[imp] = self.fit[imp]
        bi = np.argmin(self.fit)
        if self.fit[bi] < self.g_fit:
            self.g_pos = self.pos[bi].copy()
            self.g_fit = self.fit[bi]
        self.t += 1
        return self.g_fit

    def run(self, max_iter):
        self.max_iter = max_iter
        conv = [self.g_fit]
        for _ in range(max_iter):
            conv.append(self.step())
        return self.g_pos.copy(), self.g_fit, conv


StandardSCA = SCA  # 冻结名别名（回归门 / 旧脚本兼容）


def _demo():
    def rastrigin(x):
        return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

    np.random.seed(42)
    algo = SCA(D=10, N=50, bounds=(-5.12, 5.12), func=rastrigin)
    _, g, _ = algo.run(1000)
    print("SCA demo (Rastrigin D=10, seed=42): g = {:.6e}".format(g))


if __name__ == "__main__":
    _demo()
