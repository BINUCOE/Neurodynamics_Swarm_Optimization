"""PSO Family Benchmark Suite
包含: StandardPSO, PSO_wDecay, AdaptivePSO, EnergyLandscapePSO,
AD-PSO v1.3/v1.6/v1.7, DS-PSO
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


class StandardPSO:
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


# ============================================================
# 2. PSO with Weight Decay
# ============================================================


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


# ============================================================
# 3. Standard DE/rand/1/bin
# ============================================================


class ADPSOv13:
    def __init__(self, D, N, bounds, func, w=0.7, c1=1.4, c2=1.4, T=100, eps=1e-6, A_thresh=1.0, drift_scale=1.0, K=5):
        self.D, self.N, self.func = D, N, func
        self.lb, self.ub = bounds
        self.w, self.c1, self.c2 = w, c1, c2
        self.alpha, self.beta = 1.0 / T, 0.1 / T
        self.eps, self.A_thresh, self.drift_scale = eps, A_thresh, drift_scale
        self.K = K
        span = self.ub - self.lb
        self.pos = np.random.uniform(self.lb, self.ub, (N, D))
        self.vel = np.random.uniform(-span * 0.1, span * 0.1, (N, D))
        self.A = np.zeros(N)
        self.pb_pos = self.pos.copy()
        self.pb_fit = np.array([func(x) for x in self.pos])
        idx = np.argmin(self.pb_fit)
        self.g_pos, self.g_fit = self.pb_pos[idx].copy(), self.pb_fit[idx]
    
    def _lbest(self):
        l_pos = np.zeros_like(self.pos)
        half = self.K // 2
        for i in range(self.N):
            idxs = [(i + j) % self.N for j in range(-half, half + 1)]
            l_pos[i] = self.pb_pos[min(idxs, key=lambda j: self.pb_fit[j])]
        return l_pos
    
    def step(self):
        l_pos = self._lbest()
        r1, r2 = np.random.random((self.N, self.D)), np.random.random((self.N, self.D))
        self.vel = self.w * self.vel + self.c1 * r1 * (self.pb_pos - self.pos) + self.c2 * r2 * (l_pos - self.pos)
        self.pos += self.vel
        fit = np.array([self.func(x) for x in self.pos])
        for i in range(self.N):
            if fit[i] < self.pb_fit[i] * (1.0 - self.eps):
                self.A[i] *= (1.0 - self.beta)
            else:
                self.A[i] += self.alpha
            if self.A[i] > self.A_thresh:
                d = np.random.randn(self.D)
                self.pos[i] += self.A[i] * self.drift_scale * d / (np.linalg.norm(d) + 1e-10)
                self.A[i] = 0.0
        self.pos = np.clip(self.pos, self.lb, self.ub)
        for i in range(self.N):
            fit[i] = self.func(self.pos[i])
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


# ============================================================
# 5. AD-PSO v1.6 (+Levy)
# ============================================================


class ADPSOv16:
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
    
    def _update_K(self):
        r = 1.0 - min(np.mean(self.A) / self.A_thresh, 1.0)
        self.K = max(self.K_min, min(int(self.K_min + (self.K_max - self.K_min) * r), self.K_max))
        if self.K % 2 == 0: self.K += 1
    
    def _lbest(self):
        l_pos = np.zeros_like(self.pos)
        half = self.K // 2
        for i in range(self.N):
            idxs = [(i + j) % self.N for j in range(-half, half + 1)]
            l_pos[i] = self.pb_pos[min(idxs, key=lambda j: self.pb_fit[j])]
        return l_pos
    
    def step(self):
        self._update_K()
        l_pos = self._lbest()
        r1, r2 = np.random.random((self.N, self.D)), np.random.random((self.N, self.D))
        self.vel = self.w * self.vel + self.c1 * r1 * (self.pb_pos - self.pos) + self.c2 * r2 * (l_pos - self.pos)
        self.pos += self.vel
        fit = np.array([self.func(x) for x in self.pos])
        for i in range(self.N):
            if fit[i] < self.pb_fit[i] * (1.0 - self.eps):
                self.A[i] *= (1.0 - self.beta)
            else:
                self.A[i] += self.alpha
            if self.A[i] > self.A_thresh:
                d = levy_batch(1, self.D, self.levy_beta)[0]
                self.pos[i] += self.A[i] * self.drift_scale * d
                self.A[i] = 0.0
        self.pos = np.clip(self.pos, self.lb, self.ub)
        for i in range(self.N):
            fit[i] = self.func(self.pos[i])
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


# ============================================================
# 6. AD-PSO v1.7 (vectorized)
# ============================================================


class ADPSOv17:
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
        fit = np.array([self.func(x) for x in self.pos])
        imp = fit < self.pb_fit * (1.0 - self.eps)
        self.A = np.where(imp, self.A * (1.0 - self.beta), self.A + self.alpha)
        dm = self.A > self.A_thresh
        if np.any(dm):
            nd = np.sum(dm)
            d = levy_batch(nd, self.D, self.levy_beta)
            self.pos[dm] += self.A[dm, None] * self.drift_scale * d
            self.A[dm] = 0.0
        np.clip(self.pos, self.lb, self.ub, out=self.pos)
        for i in range(self.N):
            fit[i] = self.func(self.pos[i])
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


# ============================================================
# 6b. AD-PSO v1.8 (FE-fair protocol revision of v1.7)
# ============================================================


class ADPSOv18:
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


# ============================================================
# Simple Benchmark Runner
# ============================================================

TEST_FUNCTIONS = {"Sphere": {"func": sphere, "bounds": (-100, 100)},
                  "Rosenbrock": {"func": rosenbrock, "bounds": (-30, 30)},
                  "Rastrigin": {"func": rastrigin, "bounds": (-5.12, 5.12)},
                  "Ackley": {"func": ackley, "bounds": (-32, 32)},
                  "Griewank": {"func": griewank, "bounds": (-600, 600)}, }

ALGOS = {"PSO": (StandardPSO, {"w": 0.7, "c1": 1.4, "c2": 1.4}),
         "wDecay": (PSO_wDecay, {"w_max": 0.9, "w_min": 0.4, "c1": 1.4, "c2": 1.4}),
         "v1.3": (ADPSOv13, {"w": 0.7, "c1": 1.4, "c2": 1.4, "T": 100, "eps": 1e-6, "K": 5}),
         "v1.6": (ADPSOv16, {"w": 0.7, "c1": 1.4, "c2": 1.4, "T": 100, "eps": 1e-6, "K_min": 5, "K_max": 7}),
         "v1.7": (ADPSOv17, {"w": 0.7, "c1": 1.4, "c2": 1.4, "T": 100, "eps": 1e-6, "K_min": 5, "K_max": 7}),
         }

D, N, MAX_ITER, RUNS, SEED_BASE = 30, 50, 1000, 10, 42


def run_single(algo_class, func, bounds, params, seed):
    np.random.seed(seed)
    try:
        algo = algo_class(D=D, N=N, bounds=bounds, func=func, **params)
    except TypeError:
        algo = algo_class(D=D, N=N, bounds=bounds, func=func)
    _, gf, conv = algo.run(MAX_ITER)
    return gf


def main():
    print("=" * 80)
    print("  PSO Family Benchmark")
    print(f"  D={D} N={N} iter={MAX_ITER} runs={RUNS}")
    print("=" * 80)
    for fname, fc in TEST_FUNCTIONS.items():
        print(f"  [{fname}]")
        for aname, (aclass, params) in ALGOS.items():
            finals = [run_single(aclass, fc["func"], fc["bounds"], params, SEED_BASE + r) for r in range(RUNS)]
            print(f"    {aname:>10s}:  {float(np.mean(finals)):.4e} ± {float(np.std(finals)):.4e}")


if __name__ == "__main__":
    main()
