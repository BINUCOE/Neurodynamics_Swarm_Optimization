#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ad_pso.py — AD-PSO v1.8（终版，可直接取用的纯净算法资产）

Adaptive-Drift Particle Swarm Optimization：
标准 PSO 惯性内核（二阶）+ 慢适应漂移通道（A 积累器 + 定长踢出）
+ 环形 K 邻域 lbest + K 自适应调度。

机制一览（与论文记号对应）：
  内核      v ← w·v + c1·r1·(pb − x) + c2·r2·(lbest_K − x)，x ← clip(x + v)
  慢通道    改进（fit < pb·(1−eps)）→ A ← A·(1−β)；否则 A ← A + α
            α = 1/T，β = 0.1/T（T=100），A > A_thresh=1.0 触发踢出并硬复位
  踢出      x ← clip(x + A·drift_scale·u)，u = Mantegna 采样的单位方向
            （幅度恒 ∈ (A_thresh, A_thresh+α]，仅触发者重估：N + n_trig/代）
  K 自适应  K ← K_min + (K_max−K_min)·(1 − mean(A)/A_thresh)，取奇数

论文配置（受控对比与 CEC2022 基准所用）：
    algo = ADPSO(D, N, bounds, func, K_min=5, K_max=7)
（类默认值保留冻结原样 K_min=3；论文全部实验显式传 K_min=5。）

来源与一致性：本文件类体逐字拷贝自冻结实现 pso_family.ADPSOv18
（仅类名改为 ADPSO，并附冻结名别名）。三方 bit 一致性
（纯净类 ≡ 冻结类 ≡ switchboard 旗舰预设）由 tests/test_bit_identity.py 锚定。

用法：
    import numpy as np
    from ad_pso import ADPSO
    np.random.seed(42)
    algo = ADPSO(D=10, N=50, bounds=(-5.12, 5.12), func=rastrigin,
                 K_min=5, K_max=7)
    best_x, best_f, conv = algo.run(max_iter=1000)

仅依赖 numpy。py3.7+。
"""

import numpy as np

# NumPy 2.x 兼容：冻结 levy_batch 使用 np.math（NumPy 1.x 遗留接口）
import math as _math
if not hasattr(np, "math"):
    np.math = _math


# ---- 以下为冻结同源代码（pso_family.py 逐字拷贝，勿动） ----

def _levy_sigma(beta):
    return (np.math.gamma(1 + beta) * np.math.sin(np.pi * beta / 2) / (
            np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)


def levy_batch(N, D, beta=1.5):
    """各向同性随机单位方向生成器（历史名 levy_batch，保留以兼容）。

    注意（Phase 0 探针 B 实测确认）：返回的是 Mantegna 采样经归一化后的
    **单位方向向量**（模长恒为 1），并不携带 Lévy 重尾步长分布。
    AD 机制的踢出位移 = A·drift_scale·u，A∈(A_thresh, A_thresh+α]，
    drift_scale=1.0 时恒 ≈1.0 绝对单位；其逃逸语义来自群体坍缩后的
    相对尺度优势（kick/spread ≈ 5–20×），而非重尾步长。
    """
    if beta >= 1.99:
        return np.random.randn(N, D)
    sigma_u = _levy_sigma(beta)
    u = np.random.randn(N, D) * sigma_u
    v = np.random.randn(N, D)
    step = u / (np.abs(v) ** (1 / beta) + 1e-10)
    norms = np.linalg.norm(step, axis=1, keepdims=True)
    return step / (norms + 1e-10)


class ADPSO:
    """AD-PSO v1.8：v1.7 的 FE 公平协议修订版（Phase 1 批次）。

    机制与 v1.7 完全一致（PSO 内核 + A 积累器 + 环 K 邻域 lbest + 单位方向
    定长踢出 + 自适应 K）。仅修订评估协议，两处：

    1. 重评估收窄：v1.7 每代先在未裁剪位置评估 N 次、clip 后对**全部**粒子
       再评 N 次（恒定 2N/代）；v1.8 改为 clip 后单次评估 + 仅对触发踢出的
       粒子重估（N + n_triggered/代），与 ADSCAv2.1 口径完全同构。
    2. 边界策略：位置裁剪统一为"内核移动后 clip + 踢出后 clip"，与
       ADSCAv2.1 对称（v1.7 第一次在越界位置评估；ADSCAv2 踢出后不 clip）。

    Phase 0 探针 A 已验证：v1.7 在半预算等 FE 切点上仍保持平均名次 1.33，
    故本协议修订不推翻既有主结论。v1.7 类保留冻结，供谱系对照。
    """

    def __init__(self, D, N, bounds, func, w=0.7, c1=1.4, c2=1.4, T=100, eps=1e-6, A_thresh=1.0, drift_scale=1.0,
                 K_min=3, K_max=7, levy_beta=1.5):
        self.D, self.N, self.func = D, N, func
        self.lb, self.ub = bounds
        self.w, self.c1, self.c2 = w, c1, c2
        self.alpha, self.beta = 1.0 / T, 0.1 / T
        self.eps, self.A_thresh, self.drift_scale = eps, A_thresh, drift_scale
        self.K_min, self.K_max = K_min, K_max
        if self.K_max % 2 == 0: self.K_max -= 1
        self.K = self.K_max
        self.levy_beta = levy_beta
        span = self.ub - self.lb
        self.pos = np.random.uniform(self.lb, self.ub, (N, D))
        self.vel = np.random.uniform(-span * 0.1, span * 0.1, (N, D))
        self.A = np.zeros(N)
        self.pb_pos = self.pos.copy()
        self.pb_fit = np.array([func(x) for x in self.pos])
        idx = np.argmin(self.pb_fit)
        self.g_pos, self.g_fit = self.pb_pos[idx].copy(), self.pb_fit[idx]
        self._arN = np.arange(N)

    def _update_K(self):
        r = 1.0 - min(np.mean(self.A) / self.A_thresh, 1.0)
        self.K = max(self.K_min, min(int(self.K_min + (self.K_max - self.K_min) * r), self.K_max))
        if self.K % 2 == 0: self.K += 1

    def _lbest_vec(self):
        half = self.K // 2
        off = np.arange(-half, half + 1)
        idx = (self._arN[:, None] + off) % self.N
        nf = self.pb_fit[idx]
        bi = np.argmin(nf, axis=1)
        return self.pb_pos[idx[self._arN, bi]]

    def step(self):
        self._update_K()
        l_pos = self._lbest_vec()
        r1, r2 = np.random.random((self.N, self.D)), np.random.random((self.N, self.D))
        self.vel = self.w * self.vel + self.c1 * r1 * (self.pb_pos - self.pos) + self.c2 * r2 * (l_pos - self.pos)
        self.pos += self.vel
        np.clip(self.pos, self.lb, self.ub, out=self.pos)  # 边界：内核移动后 clip
        fit = np.array([self.func(x) for x in self.pos])  # 唯一一次全群评估（N 次）
        imp = fit < self.pb_fit * (1.0 - self.eps)
        self.A = np.where(imp, self.A * (1.0 - self.beta), self.A + self.alpha)
        dm = self.A > self.A_thresh
        if np.any(dm):
            nd = np.sum(dm)
            d = levy_batch(nd, self.D, self.levy_beta)
            self.pos[dm] += self.A[dm, None] * self.drift_scale * d
            self.A[dm] = 0.0
            np.clip(self.pos, self.lb, self.ub, out=self.pos)  # 边界：踢出后 clip
            fit[dm] = np.array([self.func(x) for x in self.pos[dm]])  # 仅触发者重估（n_triggered 次）
        imp2 = fit < self.pb_fit
        self.pb_pos[imp2], self.pb_fit[imp2] = self.pos[imp2].copy(), fit[imp2]
        bi = np.argmin(fit)
        if fit[bi] < self.g_fit:
            self.g_pos, self.g_fit = self.pos[bi].copy(), fit[bi]
        return self.g_fit

    def run(self, max_iter):
        conv = [self.g_fit]
        for _ in range(max_iter):
            conv.append(self.step())
        return self.g_pos.copy(), self.g_fit, conv


ADPSOv18 = ADPSO  # 冻结名别名（回归门 / 旧脚本兼容）


def _demo():
    def rastrigin(x):
        return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

    np.random.seed(42)
    algo = ADPSO(D=10, N=50, bounds=(-5.12, 5.12), func=rastrigin,
                 K_min=5, K_max=7)  # 论文配置
    _, g, _ = algo.run(1000)
    print("AD-PSO v1.8 demo (Rastrigin D=10, seed=42): g = {:.6e}".format(g))


if __name__ == "__main__":
    _demo()
