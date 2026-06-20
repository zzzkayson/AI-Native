# 🎯 AI-Native Strategy Analytics

> **基于 AI 工具链的策略分析与实验科学工具包**
>
> A toolkit for product strategy analytics, covering A/B testing, causal inference, anomaly detection, and competitor deconstruction — all powered by AI-native workflows.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

---

## 📖 目录

- [为什么做这个项目？](#为什么做这个项目)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [核心模块](#核心模块)
  - [A/B Testing](#ab-testing)
  - [Causal Inference](#causal-inference)
  - [Anomaly Detection](#anomaly-detection)
  - [Visualization](#visualization)
- [Demo 示例](#demo-示例)
- [AI-Native 工作流](#ai-native-工作流)
- [技能矩阵](#技能矩阵)
- [待办路线图](#待办路线图)

---

## 为什么做这个项目？

### 问题

策略产品/增长岗位的核心能力——因果推断、实验设计、指标监控——在传统课程中很少系统教授。大多数新人入职后才开始学习，学习曲线陡峭。

### 解决方案

这个项目把**策略分析的核心方法论**打包成一个可运行、可展示、可复现的工具包：

| 传统做法 | AI-Native 做法 (本项目) |
|---------|------------------------|
| 手动 Excel 拉数据 | Python 脚本自动分析 |
| 自己写统计检验代码 | 调用封装好的模块 + AI 解释结果 |
| 手动画图做 PPT | matplotlib/seaborn 自动生成 |
| 竞品分析靠直觉 | 结构化拆解框架 + AI 辅助 |
| 异常检测靠"看了一眼" | 3-Sigma + Bollinger + CUSUM 联合检测 |

### 适合谁？

- 🎓 正在找策略产品/数据分析/增长实习的在校生
- 📊 想从"拍脑袋"转向"科学实验"的产品经理
- 🤖 对 AI-Native 工作流感兴趣的数据分析师

---

## 项目结构

```
ai-native-strategy-analytics/
├── README.md                              ← 你在这里
├── requirements.txt                       ← Python 依赖
├── .gitignore
│
├── src/                                   ← 核心工具模块
│   ├── __init__.py
│   ├── ab_test.py                         ← A/B 测试工具包
│   ├── causal.py                          ← 因果推断工具包
│   ├── anomaly.py                         ← 异常检测工具包
│   └── viz.py                             ← 可视化工具包
│
├── examples/                              ← 完整 Demo 脚本
│   ├── 01_ab_test_demo.py                 ← A/B 测试完整流水线
│   ├── 02_causal_inference_demo.py        ← DID + PSM + IV 实战
│   └── 03_anomaly_monitor_demo.py         ← 指标监控与预警
│
├── reports/                               ← 分析报告与输出图表
│   └── competitor_deconstruction.md       ← 抖音推荐系统拆解
│
└── data/                                  ← 数据目录
```

---

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/YOUR_USERNAME/ai-native-strategy-analytics.git
cd ai-native-strategy-analytics
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行第一个 Demo

```bash
# A/B 测试完整流水线
python examples/01_ab_test_demo.py

# 因果推断实战
python examples/02_causal_inference_demo.py

# 异常监控系统
python examples/03_anomaly_monitor_demo.py
```

---

## 核心模块

### A/B Testing

`src/ab_test.py` — 频率学派 + 贝叶斯方法全覆盖

```python
from src.ab_test import estimate_sample_size, proportions_ztest, bayesian_ab_test_beta_binomial

# 1. 实验前：计算需要多少样本
n = estimate_sample_size(baseline_rate=0.10, mde=0.02, alpha=0.05, power=0.80)
print(f"每组需要 {n:,} 个样本")

# 2. 实验后：频率学派检验
result = proportions_ztest(
    successes_c=450, n_c=5000,   # 对照组: 9% 转化
    successes_t=520, n_t=5000,   # 实验组: 10.4% 转化
)
print(result.summary())

# 3. 贝叶斯验证
bayes = bayesian_ab_test_beta_binomial(450, 5000, 520, 5000)
print(f"P(Treatment > Control) = {bayes.prob_b_better_than_a:.2%}")
```

**功能清单**:
- ✅ 最小样本量估算（比率/连续指标）
- ✅ Welch's t-test + 比率 z-test
- ✅ 贝叶斯 Beta-Binomial A/B 测试
- ✅ Bonferroni & Benjamini-Hochberg 多重检验校正
- ✅ Pocock 序贯检验边界
- ✅ Cohen's d 效应量 + 统计功效计算

### Causal Inference

`src/causal.py` — 观测数据的因果效应估计

```python
from src.causal import difference_in_differences, propensity_score_matching, iv_2sls

# Difference-in-Differences
did = difference_in_differences(df, time_col="period", group_col="treated",
                                 outcome_col="revenue", treatment_time=1)

# Propensity Score Matching
psm = propensity_score_matching(df, treatment_col="got_coupon",
                                 covariates=["age", "past_orders", "is_vip"],
                                 outcome_col="purchased")

# Instrumental Variables (2SLS)
iv = iv_2sls(df, outcome_col="retention", treatment_col="used_feature",
             instrument_col="received_push")
```

**功能清单**:
- ✅ Difference-in-Differences (DID) with bootstrap SE
- ✅ Propensity Score Matching (1:k NN, caliper)
- ✅ Balance check (SMD before/after matching)
- ✅ Instrumental Variables (2SLS with F-stat)
- ✅ Sharp RDD (Local Linear Estimation)

### Anomaly Detection

`src/anomaly.py` — 多方法联合检测，降低误报

```python
from src.anomaly import generate_anomaly_digest, detect_anomalies_3sigma

# 联合检测（3-Sigma + Bollinger + CUSUM 投票）
digest = generate_anomaly_digest(metric_data, metric_name="cart_conversion_rate")
print(f"高置信异常: {digest['high_confidence_anomalies']}")

# 单一方法
result = detect_anomalies_3sigma(data, window=30, n_sigmas=3.0)
print(result.summary())
```

**功能清单**:
- ✅ 滚动 3-Sigma 检测
- ✅ Bollinger Band 风格检测
- ✅ CUSUM 漂移检测
- ✅ Isolation Forest 多维检测
- ✅ 多方法投票 + 高置信异常输出
- ✅ 自动严重程度分级 (Normal/Warning/Critical)

### Visualization

`src/viz.py` — 策略分析专用可视化

- ✅ A/B 测试分布对比图 + 置信区间图
- ✅ 漏斗转化率可视化
- ✅ 时序异常标注图（正常区间 + 异常高亮）
- ✅ DID 因果效应示意图（含反事实虚线）

---

## Demo 示例

| Demo | 场景 | 技能展示 |
|------|------|---------|
| `01_ab_test_demo.py` | 付费转化率 A/B 测试 | 样本量计算→假设检验→贝叶斯验证→多重校正→可视化 |
| `02_causal_inference_demo.py` | 策略评估的因果效应估计 | DID + PSM + IV 三种方法对比 |
| `03_anomaly_monitor_demo.py` | 90天漏斗指标监控 | 多方法联合检测→分级预警→自动报告生成 |

---

## AI-Native 工作流

本项目的核心哲学：**AI 不是替代分析思维，而是加速和深化分析过程**。

### 实际使用方式

```
┌──────────────────────────────────────────────┐
│            AI-Native 策略分析工作流             │
├──────────────────────────────────────────────┤
│                                              │
│  1. 用 Cursor/Claude Code 写分析框架          │
│     ↓                                        │
│  2. 运行本项目模块 → 得到统计结果              │
│     ↓                                        │
│  3. 结果喂给 Claude → "帮我解释这个 DID 结果"  │
│     ↓                                        │
│  4. Claude 生成报告草稿 → 人工审核修正          │
│     ↓                                        │
│  5. 可视化图表拖入 Claude → "看看有没有异常"    │
│     ↓                                        │
│  6. 最终报告 + 图表 → 分享给团队               │
│                                              │
└──────────────────────────────────────────────┘
```

### 为什么这比传统方法快 5-10x？

1. **代码生成**: "帮我写一个 Welch's t-test 函数" → 10 秒 vs 手动查文档 10 分钟
2. **结果解释**: 粘贴统计输出 → AI 用通俗语言解释 → 即时
3. **报告撰写**: "基于以上结果写一段实验结论" → 1 分钟 vs 手写 30 分钟
4. **Debug**: 报错信息丢给 AI → 80% 的情况直接定位问题

---

## 技能矩阵

本项目在策略产品实习生 JD 中的映射：

| JD 要求 | 本项目覆盖 | 对应模块 |
|---------|-----------|---------|
| 指标监控与预警 | ✅ 多方法联合检测 + 自动分级 | `anomaly.py` + Demo 03 |
| 系统机制拆解 | ✅ 抖音推荐系统结构化拆解 | `competitor_deconstruction.md` |
| A/B 测试配置与归因 | ✅ 频率学派+贝叶斯全覆盖 | `ab_test.py` + Demo 01 |
| 因果推断 | ✅ DID + PSM + IV + RDD | `causal.py` + Demo 02 |
| 熟练运用 Cursor/Claude Code | ✅ 整个项目用 AI 工具辅助开发 | AI-Native 工作流 |
| 证伪原则 | ✅ 多重检验校正、混淆变量分析 | Demos 中 explicit 讨论 |
| 数理逻辑 | ✅ 统计检验 + Bootstrap + 效应量 | 所有核心模块 |
| 数据清洗与逻辑一致 | ✅ 异常检测 + Balance Check | `anomaly.py` + PSM |

---

## 待办路线图

- [ ] 接入真实数据源（MySQL/ClickHouse connector）
- [ ] 飞书/钉钉 Webhook 自动推送异常告警
- [ ] Streamlit Web UI 交互式看板
- [ ] 更多因果推断方法（Synthetic Control, Causal Forest）
- [ ] 支持序贯检验的完整决策框架
- [ ] 单元测试覆盖
- [ ] CI/CD 自动运行 Demo 脚本

---

## 技术栈

- **Core**: Python 3.9+, NumPy, Pandas, SciPy
- **Statistics**: statsmodels
- **ML**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **AI Tools**: Claude Code, Cursor (开发辅助)

---

## 许可证

MIT License — 随意使用、修改和分享。

---

## 联系

如果你对这个项目有任何建议，或者想讨论策略产品/因果推断相关的话题，欢迎提 Issue 或 PR！

---

*Built with ❤️ and 🤖 AI-Native workflow | 2026*
