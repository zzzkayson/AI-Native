"""
Demo 02: 因果推断实战 — DID + PSM + IV
========================================
场景：评估某产品策略变更的因果效应

三个子场景：
1. DID:  某城市上线了新推荐算法 → 评估对用户时长的因果影响
2. PSM:  用户是否收到优惠券 → 评估对购买概率的影响（观察性数据）
3. IV:   推送是否触达 → 评估实际使用新功能对留存的影响

为什么需要因果推断？
"相关性 ≠ 因果性" 是策略分析中最核心的原则。
简单地比较"用了功能的用户 vs 没用的用户"会受选择偏差影响。
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
import pandas as pd
from src.causal import (
    difference_in_differences,
    propensity_score_matching,
    iv_2sls,
    DIDResult,
    IVResult,
)
from src.viz import plot_did


# ============================================================================
# Part A: Difference-in-Differences (DID)
# ============================================================================

print("=" * 65)
print("  Part A: Difference-in-Differences")
print("=" * 65)
print("""
  场景：某城市在第 30 天对 50% 用户上线了新推荐算法。
  对照组：未上线城市同期的用户
  处理组：上线城市的用户
  指标：用户日均使用时长（分钟）

  核心识别假设：Parallel Trends（平行趋势）
  → 如果没有干预，处理组和对照组的变化趋势应该相同
""")

# Generate DID data
rng = np.random.default_rng(42)
n_units = 200
records = []

for i in range(n_units):
    is_treated = 1 if i < n_units // 2 else 0
    # City-level fixed effect
    city_fixed = 2.0 if is_treated else 0.0
    unit_noise = rng.normal(0, 3)

    for t in range(2):
        # Baseline + time trend + city effect + unit noise
        y = 30 + 1.0 * t + city_fixed + unit_noise + rng.normal(0, 2)
        if is_treated and t == 1:
            y += 5.0  # Treatment effect: +5 min
        records.append({
            "unit": i,
            "time": t,
            "treated": is_treated,
            "usage_minutes": y,
        })

df_did = pd.DataFrame(records)

# Run DID
did_result = difference_in_differences(
    df_did,
    time_col="time",
    group_col="treated",
    outcome_col="usage_minutes",
    treatment_time=1,
)
print(did_result.summary())

# Visualize
pre_c = df_did[(df_did["time"] == 0) & (df_did["treated"] == 0)]["usage_minutes"].mean()
pre_t = df_did[(df_did["time"] == 0) & (df_did["treated"] == 1)]["usage_minutes"].mean()
post_c = df_did[(df_did["time"] == 1) & (df_did["treated"] == 0)]["usage_minutes"].mean()
post_t = df_did[(df_did["time"] == 1) & (df_did["treated"] == 1)]["usage_minutes"].mean()

fig_did = plot_did(pre_c, pre_t, post_c, post_t, metric_name="Usage (min/day)")

output_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(output_dir, exist_ok=True)
fig_did.savefig(os.path.join(output_dir, "did_estimate.png"), dpi=150, bbox_inches="tight")
print("  图表已保存: reports/did_estimate.png")


# ============================================================================
# Part B: Propensity Score Matching (PSM)
# ============================================================================

print("\n" + "=" * 65)
print("  Part B: Propensity Score Matching")
print("=" * 65)
print("""
  场景：运营给部分用户发了优惠券，想评估优惠券对购买概率的因果效应。
  但发券并不是随机的：运营倾向于给"可能流失的高价值用户"发券。
  → 存在选择偏差 (Selection Bias)
  → PSM 试图在倾向性得分上匹配"相似的"发券和未发券用户
""")

# Generate PSM demo data
n = 2000
rng = np.random.default_rng(42)

# Confounders
age = rng.normal(30, 8, n)                           # 年龄
past_purchase_count = rng.poisson(3, n)               # 历史购买次数
is_vip = (rng.random(n) < 0.2).astype(int)            # 是否VIP

# Treatment assignment depends on confounders (selection bias!)
# Weakened bias: the confounders affect treatment, but not overwhelmingly
ps_true = 1 / (1 + np.exp(-(-0.5 + 0.015 * age + 0.25 * past_purchase_count + 0.8 * is_vip + rng.normal(0, 0.8, n))))
received_coupon = (rng.random(n) < ps_true).astype(int)

# Outcome: purchase (also depends on confounders + treatment)
# True treatment effect = 0.35
purchase_prob = 1 / (1 + np.exp(-(-1.5 + 0.02 * age + 0.15 * past_purchase_count + 0.5 * is_vip + 0.35 * received_coupon + rng.normal(0, 0.5, n))))
purchased = (rng.random(n) < purchase_prob).astype(int)

df_psm = pd.DataFrame({
    "age": age,
    "past_purchase_count": past_purchase_count,
    "is_vip": is_vip,
    "received_coupon": received_coupon,
    "purchased": purchased,
})

# Naive comparison (biased!)
naive_treat_rate = df_psm[df_psm["received_coupon"] == 1]["purchased"].mean()
naive_control_rate = df_psm[df_psm["received_coupon"] == 0]["purchased"].mean()
naive_diff = naive_treat_rate - naive_control_rate

print(f"\n  ⚠️  Naive comparison (有偏):")
print(f"    发券组购买率: {naive_treat_rate:.3%}")
print(f"    未发券组购买率: {naive_control_rate:.3%}")
print(f"    Naive ATE: {naive_diff:.3%}  ← 混杂了选择偏差")

# PSM
psm_result = propensity_score_matching(
    df_psm,
    treatment_col="received_coupon",
    covariates=["age", "past_purchase_count", "is_vip"],
    outcome_col="purchased",
    caliper=0.5,
    k=1,
)

if psm_result.get("att") is not None:
    print(f"\n  ✅ PSM-adjusted ATT: {psm_result['att']:.3%}")
    print(f"     p-value: {psm_result['p_value']:.4f}")
    print(f"     Matched pairs: {psm_result['n_matched']}")

    print(f"\n  📊 Balance Check (SMD < 0.1 表示匹配良好):")
    print(f"    Before matching: {psm_result['balance_before']}")
    print(f"    After matching:  {psm_result['balance_after']}")
else:
    print(f"\n  ⚠️  {psm_result.get('error', 'PSM failed')}")


# ============================================================================
# Part C: Instrumental Variables (2SLS)
# ============================================================================

print("\n" + "=" * 65)
print("  Part C: Instrumental Variables (2SLS)")
print("=" * 65)
print("""
  场景：平台推送了"新功能引导"，但不是所有收到推送的用户都实际使用了。
  直接比较"用了功能的 vs 没用的"会高估效果（因为主动使用的人本身就活跃）。

  工具变量 Z = 是否收到推送提醒（随机分配 → 满足外生性）
  内生变量 D = 是否实际使用新功能
  结果变量 Y = 7日留存

  IV 估计的是 LATE (Local Average Treatment Effect)：
  对于那些"因为推送才使用"的用户（Compliers），功能对留存的影响。
""")

# Generate IV data
n_iv = 3000
rng = np.random.default_rng(42)

# Instrument Z: randomly assigned push notification
Z = (rng.random(n_iv) < 0.5).astype(int)

# Unobserved confounder (e.g., user tech-savviness)
U = rng.normal(0, 1, n_iv)

# Treatment D: actual feature usage (affected by Z and U)
D_star = -0.5 + 1.2 * Z + 0.8 * U + rng.normal(0, 0.5, n_iv)
D = (D_star > 0).astype(int)

# Outcome Y: 7-day retention (affected by D and U, NOT directly by Z)
Y = 0.3 + 0.15 * D + 0.4 * U + rng.normal(0, 0.1, n_iv)
# Clip to [0, 1]
Y = np.clip(Y, 0, 1)

df_iv = pd.DataFrame({
    "push_notification": Z,
    "used_feature": D,
    "retention_7d": Y,
    "age": rng.normal(30, 8, n_iv),  # control variable
})

# Run 2SLS
iv_result = iv_2sls(
    df_iv,
    outcome_col="retention_7d",
    treatment_col="used_feature",
    instrument_col="push_notification",
    controls=["age"],
)
print(iv_result.summary())

# Compliance check
compliance = pd.crosstab(df_iv["push_notification"], df_iv["used_feature"],
                          normalize="index")
print(f"\n  📊 依从性分析:")
print(f"    收到推送且使用: {compliance.loc[1, 1]:.1%}")
print(f"    未收到推送但使用: {compliance.loc[0, 1]:.1%} (Always-takers)")
print(f"    收到推送未使用: {compliance.loc[1, 0]:.1%} (Never-takers)")

# ============================================================================
# Key Takeaways
# ============================================================================

print("\n" + "=" * 65)
print("  🎯 因果推断核心 Checklist")
print("=" * 65)
print("""
  1. 相关性 ≠ 因果性 — 永远先怀疑混淆变量
  2. DID 依赖平行趋势假设 — 用 pre-trend 可视化验证
  3. PSM 只能控制可观测混淆 — 不能替代随机实验
  4. IV 需要强工具变量 (F > 10) + 外生性论证
  5. 所有因果推断结论都应做敏感性分析

  🤖 AI 协作提示:
  - 用 Claude Code 帮你检查因果图中的后门路径
  - 用 AI 生成敏感性分析 (E-value, Rosenbaum bounds)
  - 把 DID 图给 AI，让它自动撰写策略评估报告
""")
