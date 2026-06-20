"""
Demo 01: 完整的 A/B 测试分析流水线
===================================
场景：某产品的付费转化率优化实验

对照组 (Control): 原有支付流程
实验组 (Treatment): 新的一键支付流程

目标：判断新流程是否显著提升付费转化率

本脚本展示了一个 AI-Native 策略分析师的标准工作流：
1. 实验前：计算所需样本量
2. 实验后：假设检验 + 效应量 + 置信区间
3. 贝叶斯验证：后验概率 & 期望损失
4. 多重检验：如果同时看多个指标
5. 可视化输出

AI 工具使用提示:
- 用 Claude Code / Cursor 解释统计输出
- 用 AI 自动生成实验报告草稿
- 用 AI 辅助检查分析逻辑漏洞
"""

import sys
import os

# Windows: force UTF-8 output to avoid encoding errors with CJK/emoji
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.ab_test import (
    estimate_sample_size,
    proportions_ztest,
    bayesian_ab_test_beta_binomial,
    bh_fdr_correction,
    simulate_ab_data,
    ABTestResult,
)
from src.viz import plot_ab_comparison, plot_ab_confidence_interval


# ============================================================================
# Step 1: 实验前 — 计算所需样本量
# ============================================================================

print("=" * 65)
print("  Step 1: 实验前样本量计算")
print("=" * 65)

baseline_rate = 0.08        # 当前付费转化率 8%
mde = 0.015                 # 希望检测到 1.5% 的绝对提升
alpha = 0.05                # 显著性水平
power = 0.80                # 统计功效

n_per_group = estimate_sample_size(baseline_rate, mde, alpha, power)

print(f"""
  基线转化率:     {baseline_rate:.1%}
  最小可检测效应:  {mde:.1%}
  显著性水平 α:   {alpha}
  统计功效 1-β:   {power}
  ─────────────────────────────
  → 每组最少需要: {n_per_group:,} 个样本
  → 总计最少需要: {n_per_group * 2:,} 个样本
""")

# ============================================================================
# Step 2: 实验数据模拟 & 假设检验
# ============================================================================

print("=" * 65)
print("  Step 2: 实验数据分析 — 频率学派方法")
print("=" * 65)

# 模拟实验数据：假设新支付流程真实提升了 1.5%
control, treatment = simulate_ab_data(
    n_control=n_per_group,
    n_treatment=n_per_group,
    baseline_rate=baseline_rate,
    lift=0.015,        # 真实效应 = 1.5%
    seed=42,
)

successes_c = int(np.sum(control))
successes_t = int(np.sum(treatment))

print(f"  对照组转化: {successes_c}/{n_per_group} = {successes_c/n_per_group:.3%}")
print(f"  实验组转化: {successes_t}/{n_per_group} = {successes_t/n_per_group:.3%}")

# z-test for proportions
result = proportions_ztest(
    successes_c, n_per_group,
    successes_t, n_per_group,
    metric_name="付费转化率",
)
print(result.summary())

# ============================================================================
# Step 3: 贝叶斯验证
# ============================================================================

print("=" * 65)
print("  Step 3: 贝叶斯验证")
print("=" * 65)

bayes_result = bayesian_ab_test_beta_binomial(
    successes_c, n_per_group,
    successes_t, n_per_group,
    prior_c=(1, 1),  # Uniform prior
    prior_t=(1, 1),
    n_samples=200_000,
)
print(bayes_result.summary())

# 决策建议
if bayes_result.prob_b_better_than_a > 0.95 and bayes_result.expected_loss < 0.001:
    decision = "✅ 强烈建议全量上线"
elif bayes_result.prob_b_better_than_a > 0.80:
    decision = "🟡 建议继续观察或加大样本"
else:
    decision = "❌ 不建议上线"

print(f"\n  → 决策建议: {decision}")

# ============================================================================
# Step 4: 多重检验校正 (模拟看多个指标的场景)
# ============================================================================

print("\n" + "=" * 65)
print("  Step 4: 多重检验校正")
print("=" * 65)

# 假设同时观察 5 个指标
p_values_multi = [0.003, 0.042, 0.18, 0.61, 0.009]
metric_names = ["付费率", "ARPU", "次日留存", "7日留存", "客单价"]

fdr = bh_fdr_correction(p_values_multi, alpha=0.05)
print("\n  Benjamini-Hochberg FDR Correction:\n")
print(f"  {'指标':<12} {'原始p':<10} {'校正p':<10} {'拒绝?':<8}")
print(f"  {'-'*40}")
for name, p, adj, rej in zip(metric_names, fdr["original_p_values"],
                                fdr["adjusted_p_values"], fdr["rejected"]):
    status = "✅ Yes" if rej else "— No"
    print(f"  {name:<12} {p:<10.4f} {adj:<10.4f} {status:<8}")

print(f"\n  多重检验校正后，{sum(fdr['rejected'])}/{len(p_values_multi)} 个指标显著")

# ============================================================================
# Step 5: 可视化
# ============================================================================

print("\n" + "=" * 65)
print("  Step 5: 生成可视化报告")
print("=" * 65)

# 取连续型指标模拟数据做分布图
from src.ab_test import simulate_ab_data_continuous

rev_control, rev_treatment = simulate_ab_data_continuous(
    n_control=2000, n_treatment=2000,
    baseline_mean=25.0, baseline_std=8.0, lift=2.0,
)

# Get the continuous AB test result for CI plot
from src.ab_test import two_sample_ttest
cont_result = two_sample_ttest(rev_control, rev_treatment, metric_name="ARPU ($)")

fig1 = plot_ab_comparison(rev_control, rev_treatment, metric_name="ARPU ($)")
fig2 = plot_ab_confidence_interval(cont_result)

output_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(output_dir, exist_ok=True)
fig1.savefig(os.path.join(output_dir, "ab_test_distribution.png"), dpi=150, bbox_inches="tight")
fig2.savefig(os.path.join(output_dir, "ab_test_confidence_interval.png"), dpi=150, bbox_inches="tight")

print(f"  图表已保存至: {output_dir}/")
print("  - ab_test_distribution.png")
print("  - ab_test_confidence_interval.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 65)
print("  🎯 AI-Native 工作流要点")
print("=" * 65)
print("""
  1. 用 AI 工具 (Claude Code/Cursor) 快速生成分析框架
  2. 用统计检验保证结论的科学性
  3. 用贝叶斯方法提供决策直觉
  4. 用多重检验校正避免 p-hacking
  5. 用可视化让非技术 Stakeholder 也能理解
  6. 全程代码可复现、可审计

  🤖 AI 协作提示:
  - 把上面的统计输出直接发给 Claude，让它帮你写实验报告
  - 用 Cursor 的 inline chat 解释你不理解的统计概念
  - 把图表拖入 Claude Code 让它做视觉解读
""")
