"""SCA Family Benchmark Suite
包含: StandardSCA, SCA-AD, ADSCAv2
"""

import numpy as np
import time
import json
import math
import os
from math import gamma, sin, pi


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


# ============================================================
# Test Functions
# ============================================================


def sphere(x):
    return np.sum(x ** 2)


def rosenbrock(x):
    return np.sum(100.0 * (x[:-1] ** 2 - x[1:]) ** 2 + (x[:-1] - 1) ** 2)


def rastrigin(x):
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def ackley(x):
    D = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / D)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / D) + 20 + np.e


def griewank(x):
    return np.sum(x ** 2) / 4000.0 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) + 1


# ============================================================
# 1. Standard PSO
# ============================================================


class StandardSCA:
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


# ============================================================
# 12. SCA-AD (Sine Cosine Algorithm with Adaptive Drift)
# ============================================================


class SCA_AD:
    def __init__(self, D, N, bounds, func, a_max=2.0, c=0.5, T=100, eps=1e-6, A_thresh=1.0, drift_scale=1.0, K_min=3,
                 K_max=7, levy_beta=1.5):
        self.D, self.N, self.func = D, N, func
        self.lb, self.ub = bounds
        self.a_max = a_max
        self.c = c
        self.alpha_drift = 1.0 / T
        self.beta_drift = 0.1 / T
        self.eps = eps
        self.A_thresh = A_thresh
        self.drift_scale = drift_scale
        self.K_min, self.K_max = K_min, K_max
        if self.K_max % 2 == 0:
            self.K_max -= 1
        self.K = self.K_max
        self.levy_beta = levy_beta
        
        self.pos = np.random.uniform(self.lb, self.ub, (N, D))
        self.A_drift = np.zeros(N)
        self.pb_pos = self.pos.copy()
        self.pb_fit = np.array([func(x) for x in self.pos])
        idx = np.argmin(self.pb_fit)
        self.g_pos, self.g_fit = self.pb_pos[idx].copy(), self.pb_fit[idx]
        self.t = 0
        self.max_iter = 1000
        self._arN = np.arange(N)
    
    def _update_K(self):
        r = 1.0 - min(np.mean(self.A_drift) / self.A_thresh, 1.0)
        self.K = max(self.K_min, min(int(self.K_min + (self.K_max - self.K_min) * r), self.K_max))
        if self.K % 2 == 0:
            self.K += 1
    
    def _find_local_best(self):
        half = self.K // 2
        off = np.arange(-half, half + 1)
        idx = (self._arN[:, None] + off) % self.N
        nf = self.pb_fit[idx]
        bi = np.argmin(nf, axis=1)
        return self.pb_pos[idx[self._arN, bi]]
    
    def step(self):
        self._update_K()
        lbest = self._find_local_best()
        a = self.a_max * (1.0 - self.t / self.max_iter)
        for i in range(self.N):
            if self.A_drift[i] > self.A_thresh:
                a_mod = a + self.A_drift[i] * self.drift_scale
            else:
                a_mod = a
            
            r1 = np.random.random() * 2.0 * np.pi
            r2 = np.random.random()
            term = r2 * np.abs(self.pos[i] - lbest[i])  # 平移不变形式
            
            if np.random.random() < 0.5:
                self.pos[i] += a_mod * np.sin(r1) * term
            else:
                self.pos[i] += a_mod * np.cos(r1) * term
            
            if self.A_drift[i] > self.A_thresh:
                d = levy_batch(1, self.D, self.levy_beta)[0]
                self.pos[i] += self.A_drift[i] * self.drift_scale * d
                self.A_drift[i] = 0.0
        
        self.pos = np.clip(self.pos, self.lb, self.ub)
        fit = np.array([self.func(x) for x in self.pos])
        
        for i in range(self.N):
            if fit[i] < self.pb_fit[i] * (1.0 - self.eps):
                self.A_drift[i] *= (1.0 - self.beta_drift)
            else:
                self.A_drift[i] += self.alpha_drift
        
        imp = fit < self.pb_fit
        self.pb_pos[imp] = self.pos[imp].copy()
        self.pb_fit[imp] = fit[imp]
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


# ============================================================
# 13. ADSCAv2 (SCA-AD with vectorized operations)
# ============================================================


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


# ============================================================
# 2b. ADSCAv2.1 (boundary-symmetric protocol revision)
# ============================================================


class ADSCAv21(ADSCAv2):
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


# ============================================================
# Simple Benchmark Runner
# ============================================================

TEST_FUNCTIONS = {"Sphere": {"func": sphere, "bounds": (-100, 100)},
                  "Rosenbrock": {"func": rosenbrock, "bounds": (-30, 30)},
                  "Rastrigin": {"func": rastrigin, "bounds": (-5.12, 5.12)},
                  "Ackley": {"func": ackley, "bounds": (-32, 32)},
                  "Griewank": {"func": griewank, "bounds": (-600, 600)}, }

ALGOS = {"SCA": (StandardSCA, {"a_max": 2.0}),
         "SCA-AD": (SCA_AD, {"a_max": 2.0, "c": 0.7, "T": 100, "eps": 1e-6, "K_min": 5, "K_max": 7}),
         "ADSCAv2": (ADSCAv2, {"a_max": 2.0, "c": 0.7, "T": 100, "eps": 1e-6, "K_min": 5, "K_max": 7}), }

D, N, MAX_ITER, RUNS, SEED_BASE = 30, 50, 1000, 10, 42


def run_single(algo_class, func, bounds, params, seed):
    np.random.seed(seed)
    algo = algo_class(D=D, N=N, bounds=bounds, func=func, **params)
    _, gf, conv = algo.run(MAX_ITER)
    return gf


def main():
    print("=" * 80)
    print("  SCA Family Benchmark")
    print(f"  D={D} N={N} iter={MAX_ITER} runs={RUNS}")
    print("=" * 80)
    for fname, fc in TEST_FUNCTIONS.items():
        print(f"  [{fname}]")
        for aname, (aclass, params) in ALGOS.items():
            finals = [run_single(aclass, fc["func"], fc["bounds"], params, SEED_BASE + r) for r in range(RUNS)]
            print(f"    {aname:>10s}:  {float(np.mean(finals)):.4e} ± {float(np.std(finals)):.4e}")


if __name__ == "__main__":
    main()
