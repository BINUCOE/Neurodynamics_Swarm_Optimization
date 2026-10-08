#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ad_switchboard.py — AD-PSO / AD-SCA 谱系消融 switchboard 实现
（Route B，理论备忘录 v1.4 §2.5 / §7.2）

一个文件承载两条谱系链的全部节点，每个节点 = 一组 flag 组合：

  PSO 链:  PSO ─K4→ PSO-ring ─M1+M2→ v1.3-like ─M3→ v1.8(旗舰)
           协议边 P1+P2 由冻结类 ADPSOv17（旧 2N 口径）对照 v1.8 承担
  SCA 链:  SCA ─TI→ SCA-TI ─K4→ SCA-TI-ring ─M1+M2+M4→ SCA-AD ─K2→ SCA-AD+c
           ─M5(⟹M4失活)+P2→ ADSCAv2.1(旗舰)
           M4 量化格: SCA-AD-noboost（次代时序下 boost 为活，测振幅调制真实贡献）

回归门槛（__main__ 自测，备忘录 §7.2 实现纪律）：
  旗舰与锚点预设必须与 pso_family.py / sca_family.py 的冻结类
  **同种子逐位一致**（收敛曲线、终位置、终适应度、FE 计数四重相等）：
    PSO 侧: StandardPSO, ADPSOv18
    SCA 侧: StandardSCA, SCA_AD, ADSCAv2, ADSCAv21
  门槛不过，谱系消融不得开跑。

内置轻量插桩（不改变任何行为与 RNG 消耗）：
  fe_count    函数评估计数（FE 账本）
  trig_count  踢出次数
  boost_hits  振幅调制激活粒子次（M4 证据：同代时序下结构性为 0）
  kick_mags   每次踢出的位移幅度（理论界 (A_thresh, A_thresh+α]）

接口与冻结类完全一致: __init__(D, N, bounds, func, **kw) / .run(max_iter)。
py3.7 兼容；依赖 numpy 与同级目录 pso_family.py / sca_family.py
（levy_batch 从 pso_family 原样复用，保证与冻结代码同源）。
"""

import sys
import numpy as np

# NumPy 2.x 兼容：pso_family 的 levy_batch 使用 np.math（与 benchmark_suite 同口径）
import math as _math
if not hasattr(np, "math"):
    np.math = _math

from .pso_family import levy_batch, _levy_sigma  # 冻结同源：单位方向生成器（备忘录 §3.2 冻结口径）


def gauss_unit_batch(N, D):
    """各向同性高斯单位方向（v1.3 历史口径的向量化等价物，仅供 v1.3-like 节点）。

    注意：与 levy_batch 的 RNG 消耗模式不同（一次 randn(N,D) vs 两次），
    因此 v1.3-like 节点不与冻结 ADPSOv13 逐位一致——该节点定义即
    "v1.3 机制集（固定 K + 高斯方向）在新协议下的等价体"，无锚点需求。
    """
    d = np.random.randn(N, D)
    return d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-10)


def levy_raw_batch(N, D, beta=1.5):
    """kraw 对照格（hold6 Stage B）：Mantegna 采样但不归一化——真重尾步长。

    RNG 消耗与 levy_batch β<1.99 分支一致（两次 randn(N,D)），σ 同源；
    仅跳过行归一化。备忘录 §3.2：kraw 在 dev6 仅 1/6 函数占优（重尾无收益）。
    """
    sigma_u = _levy_sigma(beta)
    u = np.random.randn(N, D) * sigma_u
    v = np.random.randn(N, D)
    return u / (np.abs(v) ** (1.0 / beta) + 1e-10)


# ============================================================
# PSO 侧 switchboard
# ============================================================

class SB_PSO:
    """PSO 谱系 switchboard。旗舰 flag 组合 ≡ ADPSOv18（逐位一致）。

    flag:
      topology   "gbest"(StandardPSO) | "ring"(环形 K 邻域 lbest)
      channel    A 积累器 + 踢出 开/关
      k_adaptive K 自适应 开/关（关 = 固定 K）
      direction  "mantegna"(levy_batch 单位方向) | "gauss"(各向同性单位方向)
                 | "mantegna_raw"(kraw 对照：真重尾不归一化)
    评估协议恒为 v1.8 公平口径：内核移动后 clip → 全群评估 N 次 →
    积累/触发 → 踢出后 clip → 仅触发者重估（N + n_triggered/代）。
    """

    def __init__(self, D, N, bounds, func,
                 w=0.7, c1=1.4, c2=1.4,
                 topology="ring", channel=True, k_adaptive=True,
                 direction="mantegna",
                 T=100, eps=1e-6, A_thresh=1.0, drift_scale=1.0,
                 K=5, K_min=5, K_max=7, levy_beta=1.5):
        self.D, self.N, self._raw_func = D, N, func
        self.lb, self.ub = bounds
        self.w, self.c1, self.c2 = w, c1, c2
        self.topology, self.channel = topology, channel
        self.k_adaptive, self.direction = k_adaptive, direction
        self.alpha, self.beta = 1.0 / T, 0.1 / T
        self.eps, self.A_thresh, self.drift_scale = eps, A_thresh, drift_scale
        self.K_min, self.K_max = K_min, K_max
        if self.K_max % 2 == 0:
            self.K_max -= 1
        self.K = self.K_max if k_adaptive else K
        self.levy_beta = levy_beta
        # 轻量插桩（须先于首次评估初始化）
        self.fe_count = 0
        self.trig_count = 0
        self.kick_mags = []
        span = self.ub - self.lb
        self.pos = np.random.uniform(self.lb, self.ub, (N, D))
        self.vel = np.random.uniform(-span * 0.1, span * 0.1, (N, D))
        self.A = np.zeros(N)
        self.pb_pos = self.pos.copy()
        self.pb_fit = np.array([self._eval(x) for x in self.pos])
        idx = np.argmin(self.pb_fit)
        self.g_pos, self.g_fit = self.pb_pos[idx].copy(), self.pb_fit[idx]
        self._arN = np.arange(N)
        self.max_iter = 1000

    def _eval(self, x):
        self.fe_count += 1
        return float(self._raw_func(x))

    def _update_K(self):
        r = 1.0 - min(np.mean(self.A) / self.A_thresh, 1.0)
        self.K = max(self.K_min, min(int(self.K_min + (self.K_max - self.K_min) * r), self.K_max))
        if self.K % 2 == 0:
            self.K += 1

    def _lbest_vec(self):
        half = self.K // 2
        off = np.arange(-half, half + 1)
        idx = (self._arN[:, None] + off) % self.N
        nf = self.pb_fit[idx]
        bi = np.argmin(nf, axis=1)
        return self.pb_pos[idx[self._arN, bi]]

    def step(self):
        if self.k_adaptive:
            self._update_K()
        anchor = self._lbest_vec() if self.topology == "ring" else self.g_pos
        r1, r2 = np.random.random((self.N, self.D)), np.random.random((self.N, self.D))
        self.vel = self.w * self.vel + self.c1 * r1 * (self.pb_pos - self.pos) \
            + self.c2 * r2 * (anchor - self.pos)
        self.pos += self.vel
        np.clip(self.pos, self.lb, self.ub, out=self.pos)  # 边界：内核移动后 clip
        fit = np.array([self._eval(x) for x in self.pos])  # 唯一一次全群评估（N 次）
        if self.channel:
            imp = fit < self.pb_fit * (1.0 - self.eps)
            self.A = np.where(imp, self.A * (1.0 - self.beta), self.A + self.alpha)
            dm = self.A > self.A_thresh
            if np.any(dm):
                nd = np.sum(dm)
                if self.direction == "gauss":
                    d = gauss_unit_batch(nd, self.D)
                elif self.direction == "mantegna_raw":
                    d = levy_raw_batch(nd, self.D, self.levy_beta)
                else:
                    d = levy_batch(nd, self.D, self.levy_beta)
                self.kick_mags.extend((self.A[dm] * self.drift_scale).tolist())
                self.trig_count += int(nd)
                self.pos[dm] += self.A[dm, None] * self.drift_scale * d
                self.A[dm] = 0.0
                np.clip(self.pos, self.lb, self.ub, out=self.pos)  # 边界：踢出后 clip
                fit[dm] = np.array([self._eval(x) for x in self.pos[dm]])  # 仅触发者重估
        imp2 = fit < self.pb_fit
        self.pb_pos[imp2], self.pb_fit[imp2] = self.pos[imp2].copy(), fit[imp2]
        bi = np.argmin(fit)
        if fit[bi] < self.g_fit:
            self.g_pos, self.g_fit = self.pos[bi].copy(), fit[bi]
        return self.g_fit

    def run(self, max_iter):
        self.max_iter = max_iter
        conv = [self.g_fit]
        for _ in range(max_iter):
            conv.append(self.step())
        return self.g_pos.copy(), self.g_fit, conv


# ============================================================
# SCA 侧 switchboard
# ============================================================

class SB_SCA:
    """SCA 谱系 switchboard。旗舰 flag 组合 ≡ ADSCAv2.1（逐位一致）。

    flag:
      reference       "std": |r2·x − anchor|（StandardSCA 文献原味，非平移等变）
                      "ti":  r2·|x − anchor|（平移等变修复）
      topology        "gbest" | "ring"(环形 K 邻域 lbest)
      channel         A 积累器 + 踢出 开/关
      boost           振幅增益调制 a_mod = a + A·ds 开/关
                      （kick_timing="same_gen" 时结构性失活——备忘录 §2.3 M4：
                      A 越阈当代即复位，永远活不到下一次移动）
      kick_timing     "same_gen"(v2 族：同代触发，向量化) |
                      "next_gen"(SCA-AD 族：次代初踢出，逐粒子循环)
      clip_after_kick 踢出后补 clip（v2: False / v2.1: True）
      momentum        动量缓冲系数（0=off；阶消融格"SCA 加动量"用，重球法
                      m ← μ·m + Δx，pos += m；仅 vec 路径，不消耗 RNG）
      c               恢复项强度（StandardSCA/SCA-AD 真实值为 0——
                      注意 sca_family.SCA_AD 的构造参数 c 是死参数，见登记册 #13）
      direction       "mantegna"(levy_batch 单位方向，默认) | "gauss"(各向同性单位方向)
                      ——踢出方向分布开关（E2 方向对照格专用，2026-10-01 新增；
                      默认值不改变任何既有行为与 RNG 消耗，回归门不受影响）
    FE 口径：loop 路径 N/代（踢出在次代移动前，由当次全群评估覆盖）；
              vec 路径 N + n_triggered/代（触发者单独重估）。
    """

    def __init__(self, D, N, bounds, func,
                 a_max=2.0, c=0.0,
                 reference="ti", topology="ring", channel=True,
                 boost=True, kick_timing="same_gen", clip_after_kick=True,
                 k_adaptive=True, momentum=0.0, direction="mantegna",
                 T=100, eps=1e-6, A_thresh=1.0, drift_scale=1.0,
                 K=5, K_min=5, K_max=7, levy_beta=1.5):
        if kick_timing == "same_gen" and not channel:
            raise ValueError("same_gen 向量化路径要求 channel=True；"
                             "无通道节点请用 kick_timing='next_gen'（loop 路径）")
        self.D, self.N, self._raw_func = D, N, func
        self.lb, self.ub = bounds
        self.a_max, self.c = a_max, c
        self.reference, self.topology, self.channel = reference, topology, channel
        self.boost, self.kick_timing = boost, kick_timing
        self.clip_after_kick = clip_after_kick
        self.k_adaptive = k_adaptive
        self.momentum = momentum
        self.direction = direction
        self.alpha_drift, self.beta_drift = 1.0 / T, 0.1 / T
        self.eps, self.A_thresh, self.drift_scale = eps, A_thresh, drift_scale
        self.K_min, self.K_max = K_min, K_max
        if self.K_max % 2 == 0:
            self.K_max -= 1
        self.K = self.K_max if k_adaptive else K
        self.levy_beta = levy_beta
        # 轻量插桩（须先于首次评估初始化）
        self.fe_count = 0
        self.trig_count = 0
        self.boost_hits = 0
        self.kick_mags = []
        self.pos = np.random.uniform(self.lb, self.ub, (N, D))
        self.A_drift = np.zeros(N)
        self.mom = np.zeros((N, D))
        self.pb_pos = self.pos.copy()
        self.pb_fit = np.array([self._eval(x) for x in self.pos])
        idx = np.argmin(self.pb_fit)
        self.g_pos, self.g_fit = self.pb_pos[idx].copy(), self.pb_fit[idx]
        self.t = 0
        self.max_iter = 1000
        self._arN = np.arange(N)

    def _eval(self, x):
        self.fe_count += 1
        return float(self._raw_func(x))

    def _update_K(self):
        r = 1.0 - min(np.mean(self.A_drift) / self.A_thresh, 1.0)
        self.K = max(self.K_min, min(int(self.K_min + (self.K_max - self.K_min) * r), self.K_max))
        if self.K % 2 == 0:
            self.K += 1

    def _lbest_vec(self):
        half = self.K // 2
        off = np.arange(-half, half + 1)
        idx = (self._arN[:, None] + off) % self.N
        nf = self.pb_fit[idx]
        bi = np.argmin(nf, axis=1)
        return self.pb_pos[idx[self._arN, bi]]

    # ---------------- 逐粒子循环路径（StandardSCA / SCA-AD 族） ----------------
    def _step_loop(self):
        if self.channel and self.k_adaptive:
            self._update_K()
        lbest = self._lbest_vec() if self.topology == "ring" else None
        a = self.a_max * (1.0 - self.t / self.max_iter)
        for i in range(self.N):
            anchor_i = lbest[i] if self.topology == "ring" else self.g_pos
            boosted = (self.channel and self.boost
                       and self.A_drift[i] > self.A_thresh)
            a_mod = a + self.A_drift[i] * self.drift_scale if boosted else a
            if boosted:
                self.boost_hits += 1
            r1 = np.random.random() * 2.0 * np.pi
            r2 = np.random.random()
            if self.reference == "std":
                np.random.random()  # 冻结 StandardSCA 残留的 r3 抽签（抽到但未使用）——逐位一致所必需
                term = np.abs(r2 * self.pos[i] - anchor_i)      # 文献原味（非平移等变）
            else:
                term = r2 * np.abs(self.pos[i] - anchor_i)      # 平移等变形式
            if np.random.random() < 0.5:
                self.pos[i] += a_mod * np.sin(r1) * term
            else:
                self.pos[i] += a_mod * np.cos(r1) * term
            if self.c != 0.0:
                self.pos[i] += self.c * (anchor_i - self.pos[i])  # K2 恢复项（SCA-AD+c 格）
            if (self.channel and self.kick_timing == "next_gen"
                    and self.A_drift[i] > self.A_thresh):
                if self.direction == "gauss":
                    d = gauss_unit_batch(1, self.D)[0]
                else:
                    d = levy_batch(1, self.D, self.levy_beta)[0]
                self.kick_mags.append(float(self.A_drift[i] * self.drift_scale))
                self.trig_count += 1
                self.pos[i] += self.A_drift[i] * self.drift_scale * d
                self.A_drift[i] = 0.0
        self.pos = np.clip(self.pos, self.lb, self.ub)
        fit = np.array([self._eval(x) for x in self.pos])  # 唯一一次全群评估（N 次）
        if self.channel:
            imp = fit < self.pb_fit * (1.0 - self.eps)
            self.A_drift = np.where(imp, self.A_drift * (1.0 - self.beta_drift),
                                    self.A_drift + self.alpha_drift)
        imp2 = fit < self.pb_fit
        self.pb_pos[imp2] = self.pos[imp2].copy()
        self.pb_fit[imp2] = fit[imp2]
        bi = np.argmin(fit)
        if fit[bi] < self.g_fit:
            self.g_pos = self.pos[bi].copy()
            self.g_fit = fit[bi]
        self.t += 1
        return self.g_fit

    # ---------------- 向量化路径（ADSCAv2 / v2.1 族，同代时序） ----------------
    def _step_vec(self):
        if self.k_adaptive:
            self._update_K()
        lbest = self._lbest_vec()
        a = self.a_max * (1.0 - self.t / self.max_iter)
        n_hot = int(np.sum(self.A_drift > self.A_thresh))
        self.boost_hits += n_hot  # M4 证据计数：同代时序下结构性为 0
        if self.boost:
            a_mod = np.where(self.A_drift > self.A_thresh,
                             a + self.A_drift * self.drift_scale, a)
        else:
            a_mod = np.full(self.N, a)
        r1 = np.random.random(self.N) * 2.0 * np.pi
        r2 = np.random.random(self.N)
        use_sin = np.random.random(self.N) < 0.5
        term = r2[:, None] * np.abs(self.pos - lbest)  # v2 族恒为平移等变形式
        sca_sin = a_mod[:, None] * np.sin(r1[:, None]) * term
        sca_cos = a_mod[:, None] * np.cos(r1[:, None]) * term
        sca_move = np.where(use_sin[:, None], sca_sin, sca_cos)
        local_move = self.c * (lbest - self.pos)
        delta = sca_move + local_move
        if self.momentum:
            self.mom = self.momentum * self.mom + delta
            self.pos += self.mom      # 阶消融格：SCA 加动量缓冲（重球法）
        else:
            self.pos += delta
        np.clip(self.pos, self.lb, self.ub, out=self.pos)  # 边界：内核移动后 clip
        fit = np.array([self._eval(x) for x in self.pos])  # 全群评估（N 次）
        imp = fit < self.pb_fit * (1.0 - self.eps)
        self.A_drift = np.where(imp, self.A_drift * (1.0 - self.beta_drift),
                                self.A_drift + self.alpha_drift)
        dm = self.A_drift > self.A_thresh
        if np.any(dm):
            nd = np.sum(dm)
            if self.direction == "gauss":
                d = gauss_unit_batch(nd, self.D)
            else:
                d = levy_batch(nd, self.D, self.levy_beta)
            self.kick_mags.extend((self.A_drift[dm] * self.drift_scale).tolist())
            self.trig_count += int(nd)
            self.pos[dm] += self.A_drift[dm, None] * self.drift_scale * d
            self.A_drift[dm] = 0.0
            if self.clip_after_kick:
                np.clip(self.pos, self.lb, self.ub, out=self.pos)  # 边界：踢出后 clip
            fit[dm] = np.array([self._eval(x) for x in self.pos[dm]])  # 仅触发者重估
        imp2 = fit < self.pb_fit
        self.pb_pos[imp2] = self.pos[imp2].copy()
        self.pb_fit[imp2] = fit[imp2]
        bi = np.argmin(fit)
        if fit[bi] < self.g_fit:
            self.g_pos = self.pos[bi].copy()
            self.g_fit = fit[bi]
        self.t += 1
        return self.g_fit

    def step(self):
        if self.kick_timing == "same_gen":
            return self._step_vec()
        return self._step_loop()

    def run(self, max_iter):
        self.max_iter = max_iter
        conv = [self.g_fit]
        for _ in range(max_iter):
            conv.append(self.step())
        return self.g_pos.copy(), self.g_fit, conv


# ============================================================
# 预设：谱系节点 = flag 组合（Route B 节点表，备忘录 §2.5）
# ============================================================

PRESETS = {
    # ---- PSO 链 ----
    "PSO":         (SB_PSO, dict(topology="gbest", channel=False)),
    "PSO-ring":    (SB_PSO, dict(topology="ring", channel=False,
                                 k_adaptive=False, K=5)),                      # 缺失链接①：纯环拓扑
    "v1.3-like":   (SB_PSO, dict(topology="ring", channel=True,
                                 k_adaptive=False, K=5, direction="gauss")),   # v1.3 机制集 @ 新协议
    "v1.8":        (SB_PSO, dict(topology="ring", channel=True,
                                 k_adaptive=True, K_min=5, K_max=7,
                                 direction="mantegna")),                       # 旗舰（锚点：ADPSOv18）
    # ---- SCA 链 ----
    "SCA":         (SB_SCA, dict(reference="std", topology="gbest",
                                 channel=False, kick_timing="next_gen",
                                 c=0.0)),                                      # 锚点：StandardSCA
    "SCA-TI":      (SB_SCA, dict(reference="ti", topology="gbest",
                                 channel=False, kick_timing="next_gen",
                                 c=0.0)),                                      # 缺失链接②：纯平移等变修复
    "SCA-TI-ring": (SB_SCA, dict(reference="ti", topology="ring", channel=False,
                                 kick_timing="next_gen",
                                 k_adaptive=False, K=5, c=0.0)),               # 缺失链接③：TI + 环
    "SCA-AD":      (SB_SCA, dict(reference="ti", topology="ring", channel=True,
                                 boost=True, kick_timing="next_gen",
                                 k_adaptive=True, K_min=5, K_max=7, c=0.0)),   # 锚点：SCA_AD
    "SCA-AD-noboost": (SB_SCA, dict(reference="ti", topology="ring", channel=True,
                                    boost=False, kick_timing="next_gen",
                                    k_adaptive=True, K_min=5, K_max=7, c=0.0)),  # M4 量化格
    "SCA-AD+c":    (SB_SCA, dict(reference="ti", topology="ring", channel=True,
                                 boost=True, kick_timing="next_gen",
                                 k_adaptive=True, K_min=5, K_max=7, c=0.7)),   # 缺失链接④：隔离 K2
    "ADSCAv2":     (SB_SCA, dict(reference="ti", topology="ring", channel=True,
                                 boost=True, kick_timing="same_gen",
                                 clip_after_kick=False,
                                 k_adaptive=True, K_min=5, K_max=7, c=0.7)),   # 锚点：ADSCAv2
    "ADSCAv2.1":   (SB_SCA, dict(reference="ti", topology="ring", channel=True,
                                 boost=True, kick_timing="same_gen",
                                 clip_after_kick=True,
                                 k_adaptive=True, K_min=5, K_max=7, c=0.7)),   # 旗舰（锚点：ADSCAv21）
}


def make(name, D, N, bounds, func):
    """按预设名构造谱系节点。"""
    cls, kw = PRESETS[name]
    return cls(D=D, N=N, bounds=bounds, func=func, **kw)


# ============================================================
# 自测：回归门槛 + 插桩证据（python ad_switchboard.py）
# ============================================================

def _counted(func):
    def f(x):
        f.n += 1
        return func(x)
    f.n = 0
    return f


def _regression_gate():
    """旗舰/锚点预设 vs 冻结类：同种子逐位一致性（四重核对）。"""
    from pso_family import StandardPSO, ADPSOv18, rastrigin
    from sca_family import StandardSCA, SCA_AD, ADSCAv2, ADSCAv21

    D, N, ITERS, B = 10, 50, 200, (-5.12, 5.12)
    anchors = [
        ("PSO",       StandardPSO, {},                                 "PSO"),
        ("v1.8",      ADPSOv18,    {"K_min": 5, "K_max": 7},           "v1.8"),
        ("SCA",       StandardSCA, {},                                 "SCA"),
        ("SCA-AD",    SCA_AD,      {"c": 0.7, "K_min": 5, "K_max": 7}, "SCA-AD"),
        ("ADSCAv2",   ADSCAv2,     {"c": 0.7, "K_min": 5, "K_max": 7}, "ADSCAv2"),
        ("ADSCAv2.1", ADSCAv21,    {"c": 0.7, "K_min": 5, "K_max": 7}, "ADSCAv2.1"),
    ]
    seeds = [42, 7, 2024]
    print("=" * 78)
    print("  回归门槛：switchboard 预设 vs 冻结类（同种子逐位一致）")
    print("  D={} N={} iters={} func=Rastrigin".format(D, N, ITERS))
    print("=" * 78)
    all_pass = True
    for label, cls_frozen, kw_frozen, preset in anchors:
        for s in seeds:
            np.random.seed(s)
            fa = _counted(rastrigin)
            a = cls_frozen(D=D, N=N, bounds=B, func=fa, **kw_frozen)
            _, ga, ca = a.run(ITERS)
            np.random.seed(s)
            b = make(preset, D=D, N=N, bounds=B, func=rastrigin)
            _, gb, cb = b.run(ITERS)
            ok = (np.array_equal(np.asarray(ca, dtype=float), np.asarray(cb, dtype=float))
                  and np.array_equal(a.pos, b.pos)
                  and float(ga) == float(gb)
                  and fa.n == b.fe_count)
            all_pass &= ok
            print("  {:<10s} seed={:<5d} conv/pos/gfit/FE 逐位一致: {}".format(
                label, s, "PASS" if ok else "FAIL"))
    print("-" * 78)
    print("  回归门槛总判定: {}".format("PASS —— 谱系消融获准开跑" if all_pass
                                      else "FAIL —— 禁止开跑，先排查实现噪声"))
    return all_pass


def _cec2022_spot_check():
    """可选：CEC2022 F4 上的旗舰抽查（需 opfunu 与同级 cec_schwefel_fix）。"""
    try:
        import cec_schwefel_fix  # noqa: F401
        from opfunu.cec_based import cec2022
        from pso_family import ADPSOv18
        from sca_family import ADSCAv21
    except Exception as e:
        print("\n  [跳过] CEC2022 抽查：{}".format(e))
        return
    print("\n  CEC2022 F4 (D=10) 旗舰抽查（100 代, seed=42）:")
    f = cec2022.F42022(ndim=10)
    B = (float(f.lb[0]), float(f.ub[0]))
    for label, cls_frozen, preset in [("v1.8", ADPSOv18, "v1.8"),
                                      ("ADSCAv2.1", ADSCAv21, "ADSCAv2.1")]:
        np.random.seed(42)
        a = cls_frozen(D=10, N=50, bounds=B, func=lambda x: float(f.evaluate(x)),
                       **{"c": 0.7, "K_min": 5, "K_max": 7} if "SCA" in label else
                       {"K_min": 5, "K_max": 7})
        _, ga, _ = a.run(100)
        np.random.seed(42)
        b = make(preset, D=10, N=50, bounds=B, func=lambda x: float(f.evaluate(x)))
        _, gb, _ = b.run(100)
        print("    {:<10s} 冻结类 g={:.6e}  switchboard g={:.6e}  一致: {}".format(
            label, ga, gb, "PASS" if float(ga) == float(gb) else "FAIL"))


def _presets_smoke():
    """全谱系节点冒烟：FE 账本、触发统计、M4 证据、踢出幅度界。"""
    from pso_family import rastrigin
    D, N, ITERS, B = 10, 50, 300, (-5.12, 5.12)
    print("\n" + "=" * 78)
    print("  谱系节点冒烟（D={} N={} iters={} func=Rastrigin, seed=42）".format(D, N, ITERS))
    print("-" * 78)
    print("  {:<16s} {:>9s} {:>8s} {:>10s} {:>16s} {:>12s}".format(
        "节点", "FE/代×N", "踢出", "boost激活", "踢出幅度min~max", "终态 g"))
    print("-" * 78)
    for name in PRESETS:
        np.random.seed(42)
        algo = make(name, D=D, N=N, bounds=B, func=rastrigin)
        _, g, _ = algo.run(ITERS)
        fe_per_gen = (algo.fe_count - N) / ITERS
        km = algo.kick_mags if algo.kick_mags else [0.0]
        print("  {:<16s} {:>9.3f} {:>8d} {:>10d} {:>16s} {:>12.4e}".format(
            name, fe_per_gen / N, algo.trig_count,
            getattr(algo, "boost_hits", 0),
            "{:.4f}~{:.4f}".format(min(km), max(km)) if algo.kick_mags else "—",
            g))
    print("-" * 78)
    print("  检查点：旗舰踢出幅度应 ∈ (1.0, 1.01]（备忘录 §1.4 理论界）；")
    print("          ADSCAv2/v2.1 的 boost激活 应恒为 0（M4 时序推论）；")
    print("          SCA-AD 的 boost激活 应 > 0（次代时序下为活）。")


if __name__ == "__main__":
    ok = _regression_gate()
    _cec2022_spot_check()
    _presets_smoke()
    sys.exit(0 if ok else 1)
