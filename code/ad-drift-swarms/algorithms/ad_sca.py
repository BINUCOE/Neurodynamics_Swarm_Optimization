#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ad_sca.py — AD-SCA v2.1（终版，可直接取用的纯净算法资产）

Adaptive-Drift Sine Cosine Algorithm：
SCA 振荡内核（一阶，sin/cos 搜索 + 平移等变参考项 r2·|x − anchor|）
+ 与 AD-PSO 完全相同的慢适应漂移通道（A 积累器 + 定长踢出）
+ 环形 K 邻域 lbest + K 自适应 + lbest 恢复项（c=0.7）。

机制一览（与论文记号对应）：
  内核      x ← clip(x + a_mod·sin/cos(r1)·r2·|x − lbest_K| + c·(lbest_K − x))
            a = a_max·(1 − t/max_iter) 线性退火；sin/cos 等概率抽签
  慢通道    与 AD-PSO v1.8 完全同构：α = 1/T，β = 0.1/T（T=100），
            A > A_thresh=1.0 → 同代踢出（幅度 ∈ (1.0, 1.01]）并硬复位，
            踢出后 clip，仅触发者重估（N + n_trig/代）
  K 自适应  与 AD-PSO v1.8 相同（mean(A) 调度，取奇数）

论文配置（受控对比与 CEC2022 基准所用）：
    algo = ADSCA(D, N, bounds, func, c=0.7, K_min=5, K_max=7)
（类默认值保留冻结原样 c=0.5、K_min=3；论文全部实验显式传 c=0.7、K_min=5。）

来源与一致性：本文件类体逐字拷贝自冻结实现 sca_family.ADSCAv2 /
sca_family.ADSCAv21（后者唯一改动 = 踢出后补一次 clip，与 AD-PSO v1.8
边界策略对称；此处子类改名为 ADSCA，并附冻结名别名）。三方 bit 一致性
（纯净类 ≡ 冻结类 ≡ switchboard 旗舰预设）由 tests/test_bit_identity.py 锚定。

用法：
    import numpy as np
    from ad_sca import ADSCA
    np.random.seed(42)
    algo = ADSCA(D=10, N=50, bounds=(-5.12, 5.12), func=rastrigin,
                 c=0.7, K_min=5, K_max=7)
    best_x, best_f, conv = algo.run(max_iter=1000)

仅依赖 numpy。py3.7+。
"""

import numpy as np

# NumPy 2.x 兼容：冻结 levy_batch 使用 np.math（NumPy 1.x 遗留接口）
import math as _math
if not hasattr(np, "math"):
    np.math = _math


# ---- 以下为冻结同源代码（sca_family.py 逐字拷贝，勿动） ----

def _levy_sigma(beta):
    return (np.math.gamma(1 + beta) * np.math.sin(np.pi * beta / 2) / (
            np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)


def levy_batch(N, D, beta=1.5):
    """各向同性随机单位方向生成器（历史名 levy_batch，保留以兼容）。

    注意：返回的是 Mantegna 采样经归一化后的**单位方向向量**（模长恒为 1），
    并不携带 Lévy 重尾步长分布。AD 机制的踢出位移 = A·drift_scale·u，
    A∈(A_thresh, A_thresh+α]，drift_scale=1.0 时恒 ≈1.0 绝对单位。
    """
    if beta >= 1.99:
        return np.random.randn(N, D)
    sigma_u = _levy_sigma(beta)
    u = np.random.randn(N, D) * sigma_u
    v = np.random.randn(N, D)
    step = u / (np.abs(v) ** (1 / beta) + 1e-10)
    norms = np.linalg.norm(step, axis=1, keepdims=True)
    return step / (norms + 1e-10)


class ADSCAv2:
    """
    SCA-AD v2: Fully Vectorized SCA with Adaptive Drift

    Key improvements over SCA-AD:
    1. Vectorized K-neighborhood computation (3-10x faster)
    2. Vectorized drift accumulation (5-20x faster)
    3. Vectorized drift trigger (5-20x faster)
    4. Vectorized SCA update where possible

    Core mechanism unchanged:
    - SCA oscillation (sin/cos search)
    - K-neighborhood best guidance
    - Adaptive drift + Levy perturbation

    Parameters:
    - a_max: maximum amplitude (default 2.0)
    - c: guidance strength (default 0.5)
    - T: drift time constant (default 100)
    - eps: improvement threshold (default 1e-6)
    - A_thresh: drift trigger threshold (default 1.0)
    - drift_scale: drift scaling factor (default 1.0)
    - K_min, K_max: neighborhood range (default 3, 7)
    - levy_beta: Levy exponent (default 1.5)
    """

    def __init__(self, D, N, bounds, func, a_max=2.0, c=0.5, T=100, eps=1e-6, A_thresh=1.0, drift_scale=1.0, K_min=3,
                 K_max=7, levy_beta=1.5):
        self.D, self.N, self.func = D, N, func
        self.lb, self.ub = bounds
        self.a_max = a_max
        self.c = c

        # Drift parameters
        self.alpha_drift = 1.0 / T
        self.beta_drift = 0.1 / T
        self.eps = eps
        self.A_thresh = A_thresh
        self.drift_scale = drift_scale

        # K-neighborhood parameters
        self.K_min, self.K_max = K_min, K_max
        if self.K_max % 2 == 0:
            self.K_max -= 1
        self.K = self.K_max

        # Levy parameters
        self.levy_beta = levy_beta

        # Initialize population
        self.pos = np.random.uniform(self.lb, self.ub, (N, D))

        # Drift accumulator
        self.A_drift = np.zeros(N)

        # Personal best
        self.pb_pos = self.pos.copy()
        self.pb_fit = np.array([func(x) for x in self.pos])

        # Global best
        idx = np.argmin(self.pb_fit)
        self.g_pos, self.g_fit = self.pb_pos[idx].copy(), self.pb_fit[idx]

        # Iteration counter
        self.t = 0
        self.max_iter = 1000

        # Pre-computed index array (for vectorized K-neighborhood)
        self._arN = np.arange(N)

    def _update_K(self):
        """Update K-neighborhood size based on drift state"""
        r = 1.0 - min(np.mean(self.A_drift) / self.A_thresh, 1.0)
        self.K = max(self.K_min, min(int(self.K_min + (self.K_max - self.K_min) * r), self.K_max))
        if self.K % 2 == 0:
            self.K += 1

    def _lbest_vec(self):
        """Vectorized: Find best in K-neighborhood"""
        half = self.K // 2
        off = np.arange(-half, half + 1)
        # N x K index matrix
        idx = (self._arN[:, None] + off) % self.N
        # N x K fitness matrix
        nf = self.pb_fit[idx]
        # Index of minimum in each row
        bi = np.argmin(nf, axis=1)
        # Return best positions
        return self.pb_pos[idx[self._arN, bi]]

    def _sca_update_vec(self, lbest):
        """Vectorized SCA update for all particles"""
        # Amplitude decay
        a = self.a_max * (1.0 - self.t / self.max_iter)

        # Modulate amplitude based on drift state
        # High drift -> higher amplitude
        a_mod = np.where(self.A_drift > self.A_thresh, a + self.A_drift * self.drift_scale, a)

        # Random parameters for all particles
        r1 = np.random.random(self.N) * 2.0 * np.pi
        r2 = np.random.random(self.N)
        use_sin = np.random.random(self.N) < 0.5

        # SCA reference term: r2 * |pos - lbest|（平移不变，锚点=lbest）
        term = r2[:, None] * np.abs(self.pos - lbest)

        # Compute SCA moves
        sca_sin = a_mod[:, None] * np.sin(r1[:, None]) * term
        sca_cos = a_mod[:, None] * np.cos(r1[:, None]) * term

        # Select sin or cos for each particle
        sca_move = np.where(use_sin[:, None], sca_sin, sca_cos)

        # Local guidance
        local_move = self.c * (lbest - self.pos)

        # Combined movement
        self.pos += sca_move + local_move

    def _drift_accumulate_vec(self, fit):
        """Vectorized: Update drift accumulator for all particles"""
        imp = fit < self.pb_fit * (1.0 - self.eps)
        self.A_drift = np.where(imp, self.A_drift * (1.0 - self.beta_drift), self.A_drift + self.alpha_drift)

    def _drift_trigger_vec(self):
        """Vectorized: Apply Levy perturbation to particles with high drift. Returns trigger mask."""
        dm = self.A_drift > self.A_thresh
        if np.any(dm):
            nd = np.sum(dm)
            d = levy_batch(nd, self.D, self.levy_beta)
            self.pos[dm] += self.A_drift[dm, None] * self.drift_scale * d
            self.A_drift[dm] = 0.0
        return dm

    def step(self):
        # Step 1: Update K
        self._update_K()

        # Step 2: Find local best (vectorized)
        lbest = self._lbest_vec()

        # Step 3: SCA update (vectorized)
        self._sca_update_vec(lbest)

        # Step 4: Boundary handling
        np.clip(self.pos, self.lb, self.ub, out=self.pos)

        # Step 5: Evaluate
        fit = np.array([self.func(x) for x in self.pos])

        # Step 6: Drift accumulation (vectorized)
        self._drift_accumulate_vec(fit)

        # Step 7: Drift trigger (vectorized)
        dm = self._drift_trigger_vec()

        # Step 8: Re-evaluate only the particles that actually triggered drift
        if np.any(dm):
            fit[dm] = np.array([self.func(x) for x in self.pos[dm]])

        # Step 9: Update personal best
        imp = fit < self.pb_fit
        self.pb_pos[imp] = self.pos[imp].copy()
        self.pb_fit[imp] = fit[imp]

        # Step 10: Update global best
        bi = np.argmin(fit)
        if fit[bi] < self.g_fit:
            self.g_pos = self.pos[bi].copy()
            self.g_fit = fit[bi]

        self.t += 1
        return self.g_fit

    def run(self, max_iter):
        self.max_iter = max_iter
        conv = [self.g_fit]
        for _ in range(max_iter):
            conv.append(self.step())
        return self.g_pos.copy(), self.g_fit, conv


class ADSCA(ADSCAv2):
    """ADSCAv2.1：v2 的边界对称协议修订版（Phase 1 批次）。

    唯一改动：踢出后补一次 clip，与 ADPSOv18 的边界策略完全对称
    （v2 踢出后不 clip，被踢粒子可带一代越界自由度并以越界坐标参与重估）。
    机制、参数、FE 口径（N + n_triggered/代）均与 v2 一致。v2 类保留冻结。
    """

    def _drift_trigger_vec(self):
        dm = self.A_drift > self.A_thresh
        if np.any(dm):
            nd = np.sum(dm)
            d = levy_batch(nd, self.D, self.levy_beta)
            self.pos[dm] += self.A_drift[dm, None] * self.drift_scale * d
            self.A_drift[dm] = 0.0
            np.clip(self.pos, self.lb, self.ub, out=self.pos)  # 踢出后边界裁剪
        return dm


ADSCAv21 = ADSCA  # 冻结名别名（回归门 / 旧脚本兼容）


def _demo():
    def rastrigin(x):
        return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

    np.random.seed(42)
    algo = ADSCA(D=10, N=50, bounds=(-5.12, 5.12), func=rastrigin,
                 c=0.7, K_min=5, K_max=7)  # 论文配置
    _, g, _ = algo.run(1000)
    print("AD-SCA v2.1 demo (Rastrigin D=10, seed=42): g = {:.6e}".format(g))


if __name__ == "__main__":
    _demo()
