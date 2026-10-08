# ad-drift-swarms

**One Nervous System, Two Bodies: A Controlled Comparison of Inertial and Oscillatory Motion Kernels in Neurodynamics-Inspired Adaptive-Drift Swarm Optimization** —— 论文配套代码。

同一个慢适应通道（"神经系统"，漏电积分 + 阈值复位）移植到两种运动内核（"身体"）：
PSO 二阶惯性内核 → **AD-PSO**；SCA 一阶振荡内核 → **AD-SCA**。
核心发现：慢通道可移植，但其价值与最优调度律由内核动力学阶数决定，且两侧最优方向相反。

## 仓库地图

```
ad-drift-swarms/
├── algorithms/                 纯净算法资产（仅依赖 numpy，可直接取用）
│   ├── ad_pso.py               AD-PSO 终版（class ADPSO，内置 demo）
│   ├── ad_sca.py               AD-SCA 终版（class ADSCA，内置 demo）
│   ├── pso.py                  原生 Standard PSO（基线）
│   ├── pso_wdecay.py           惯性权重线性衰减 PSO（基线变体）
│   └── sca.py                  原生 Standard SCA（基线）
│
├── core/                       共享内核（测评的单一实现源）
│   ├── ad_switchboard.py       开关板总线：全部中间/消融节点 = flag 组合（18 锚点回归门把守）
│   ├── pso_family.py           冻结依赖（勿动）
│   ├── sca_family.py           冻结依赖（勿动）
│   └── cec_schwefel_fix.py     opfunu 的 CEC2022 Schwefel 修复（勿动）
│
├── experiments/                测评脚本（一个子目录 = 论文的一组实验）
│   ├── benchmark_cec2022/      E1：CEC2022 正式基准（5 算法 × 12 函数 × 2 维 × 30 runs）
│   ├── ablation_chain/         E2：逐机制消融链（--with-extras 附加方向/逆序对照）
│   ├── scheduling_sweep/       E3：通道周期 T 剂量-响应（25/50/100/200 四档）
│   └── neuro_mapping/          E4：全基准插桩映射核验（逐粒子 A 轨迹 + 踢事件日志）
│
├── tests/
│   ├── test_bit_identity.py    三方 bit 一致性回归门（15 项应全部 PASS）
│   ├── make_paper_figs.py      论文全部正式图表一个脚本出（cd/conv/box/kernel/heatmap3/…）
│   └── figstyle.py             图表统一样式（Times 系字体回退链，两文件需放同一目录）
│
└── （仓库外）结果_E2E3E4/       统计产物与论文图表输出目录
```

**为什么这样分**：读者想"用算法"→ 只进 `algorithms/`，每个文件自包含、带 docstring 和 demo；
想"验证论文结论"→ 进 `experiments/`，按 E1–E4 编号对应论文实验链。全部中间/消融版本不散落各处——
它们以 flag 组合的形式收敛在 `core/ad_switchboard.py` 单一实现中，由测评脚本按需调用。

## 快速开始

```bash
# 用算法（以 AD-PSO 为例）
cd algorithms && python ad_pso.py              # 内置 demo：Rastrigin D=10

# 验证代码未被改动（任何时候）
python tests/test_bit_identity.py              # 15 项应全部 PASS

# 复现论文实验（需要 opfunu==1.0.1，Python 3.7）
pip install "opfunu==1.0.1"
python experiments/benchmark_cec2022/run_formal_cec2022.py --workers 8   # E1：3600 runs
python experiments/ablation_chain/run_e2_chain.py --workers 8            # E2：3600 runs
python experiments/ablation_chain/run_e2_chain.py --workers 8 --with-extras  # E2+：6480 runs
python experiments/scheduling_sweep/run_e3_tsweep.py --workers 8         # E3：4320 runs
python experiments/neuro_mapping/instrument_e4.py --workers 8            # E4：144 runs 插桩

# 统计出数（默认输出到 结果_E2E3E4/，不存在会自动创建）
python experiments/benchmark_cec2022/stats_formal.py --results phase4_results
python experiments/ablation_chain/stats_e2.py
python experiments/scheduling_sweep/stats_e3.py
python experiments/neuro_mapping/stats_e4.py
python experiments/neuro_mapping/stats_delay_corr.py

# 论文图表（一个脚本全出；缺数据的图会明确提示跳过，不会中断）
python tests/make_paper_figs.py                     # 全部
python tests/make_paper_figs.py cd conv box kernel heatmap3   # 只出指定图
```

## 实验协议（全部脚本统一）

- 基准：CEC2022（opfunu 1.0.1 + `core/cec_schwefel_fix.py` 修复），搜索域 [-100, 100]^D
- 种群 N=50；预算：D=10 → 200,000 FE，D=20 → 500,000 FE（按 FE 预算终止）
- 种子：`seed = 666 + run_id + func_id×100 + dim×1000`（跨算法、跨消融变体严格配对）
- 误差封底 10⁻¹⁰；统计：Wilcoxon 符号秩（成对）+ Friedman/Nemenyi
- 慢通道共享参数：T=100，A_thresh=1.0，α=1/T，β=0.1/T，ε=10⁻⁶，d_s=1
- 内核参数：AD-PSO w=0.7, c1=c2=1.4；AD-SCA a_max=2.0, c=0.7

## 一致性保证

三层实现（纯净类 / 冻结类 / switchboard 预设）同种子逐位一致——
收敛行为、终位置、终适应度、FE 计数多重相等，由 `tests/test_bit_identity.py`
与 `core/ad_switchboard.py` 内置回归门双重锚定。改动任何一处，门禁即 FAIL。

依赖：Python ≥ 3.7，numpy；统计另需 scipy、pandas、matplotlib；实验复现另需 opfunu==1.0.1。
