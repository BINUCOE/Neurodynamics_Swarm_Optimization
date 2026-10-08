"""figstyle.py — 论文图件统一样式（2026-10-02 台账 T0）。
所有 make_paper_figs.py 出的图 import 本文件。沙箱无正版 Times New Roman，
用 Liberation Serif 顶替（与 Times 逐字宽一致）；本机有正版时把 FAMILY 改一行即可。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "STIXGeneral",
                                    "DejaVu Serif"],  # 按可用性自动回退
                     "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8, "mathtext.fontset": "stix",  # 数学字体与 Times 风格一致
                     "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold",
                     "axes.linewidth": 0.8, "lines.linewidth": 1.2, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "figure.dpi": 110, })

# 色板：两身各一主色；基线三色可区分（Okabe-Ito 色盲安全）。全论文图统一。
C_ADPSO = "#2166ac"  # AD-PSO 蓝
C_ADSCA = "#b2182b"  # AD-SCA 红
C_PSO = "#009e73"  # PSO 青绿
C_SCA = "#e69f00"  # SCA 橙
C_WDEC = "#8c8c8c"  # wDecay 灰
C_BASE = C_WDEC  # 兼容旧引用
ALGO_STYLE = {  # 读者口径名 -> (颜色, 线宽)
    "AD-PSO": (C_ADPSO, 1.8), "AD-SCA": (C_ADSCA, 1.8), "PSO": (C_PSO, 1.0), "SCA": (C_SCA, 1.0),
    "wDecay": (C_WDEC, 1.0), }


def newfig(width_mm, height_mm):
    """按论文栏宽开图：单栏 88，双栏 180（单位 mm）。"""
    return plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4))
