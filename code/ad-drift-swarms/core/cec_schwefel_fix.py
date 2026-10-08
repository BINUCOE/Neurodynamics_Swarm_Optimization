# -*- coding: utf-8 -*-
"""
cec_schwefel_fix.py — 修复 opfunu 1.0.1 中 Schwefel 函数 |z|>500 延拓分支的缺陷

问题
----
opfunu 1.0.1 的 ``opfunu.utils.operator.modified_schwefel_func`` 在 z > 500 分支写作::

    (500 + |z| mod 500) * sin(sqrt(500 - |z| mod 500))

该延拓项在 z ≈ 997.5 处峰值约 997.5，是名义最优点 z = 420.97 处（418.98）的两倍多，
从而在实现版地貌中产生低于声明 f_global 的伪极小（例如 CEC2014 F10 的缩放为
10*(x - shift)，对应 x ≈ shift + 99.75，在搜索域 [-100, 100] 内部）。
后果：``fitness - f_global`` 误差指标在所有含 Schwefel 组件的函数上失真
（下潜越深的算法反而误差越大）。

修复
----
将 z > 500 分支改为 CEC 官方问题定义文档（technical report）的形式，与
opfunu 官方 1.0.2+ 的修复逐字一致::

    (500 - |z| mod 500) * sin(sqrt(500 - |z| mod 500))

修复后延拓分支峰值封顶于 418.98 且附罚项，真最小值恰好等于声明的 f_global。
本模块等价于"opfunu 1.0.1 + 反移植 1.0.4 的该处修复"，其余一切不动。

用法
----
在任何 opfunu cec_based 函数被调用之前 import 本模块一次即可（猴子补丁，
对 ``opfunu.utils.operator`` 模块属性生效；全部 CEC 模块均经该属性调用，
故一处补丁全局生效）::

    import cec_schwefel_fix   # noqa: F401

验证：apply_fix() 返回 True 表示补丁已生效。
"""

import numpy as np

__all__ = ["apply_fix", "modified_schwefel_func_fixed"]


def modified_schwefel_func_fixed(x):
    """与 opfunu 1.0.4 逐字一致的 modified_schwefel_func（符合 CEC 官方定义文档）。"""
    z = np.array(x).ravel() + 4.209687462275036e+2
    nx = len(z)

    mask1 = z > 500
    mask2 = z < -500
    mask3 = ~mask1 & ~mask2
    fx = np.zeros(nx)
    fx[mask1] -= ((500.0 - np.fmod(z[mask1], 500)) * np.sin(np.sqrt(500.0 - np.fmod(z[mask1], 500)))
                  - ((z[mask1] - 500.0) / 100.) ** 2 / nx)
    fx[mask2] -= (-500.0 + np.fmod(np.abs(z[mask2]), 500)) * np.sin(
        np.sqrt(500.0 - np.fmod(np.abs(z[mask2]), 500))) - (
                     (z[mask2] + 500.0) / 100.) ** 2 / nx
    fx[mask3] -= z[mask3] * np.sin(np.sqrt(np.abs(z[mask3])))

    return np.sum(fx) + 4.189828872724338e+002 * nx


def apply_fix():
    """将修复版函数注入 opfunu.utils.operator。返回 True 表示补丁已生效。"""
    from opfunu.utils import operator
    operator.modified_schwefel_func = modified_schwefel_func_fixed
    return operator.modified_schwefel_func is modified_schwefel_func_fixed


apply_fix()
