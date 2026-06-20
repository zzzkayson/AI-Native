"""
Demo 03: 指标异常监控与预警系统
================================
场景：产品核心漏斗的日常监控

监控指标：
1. 首页曝光 → 商品详情页 (浏览转化)
2. 详情页 → 加购 (加购转化)
3. 加购 → 下单 (下单转化)
4. 下单 → 支付 (支付转化)

本脚本展示：
1. 多方法联合异常检测
2. 异常严重程度分级
3. 自动化预警报告生成
4. 漏斗健康度看板

AI-Native 亮点：
- 传统做法：每天手动拉数据看板
- AI-Native：自动检测 → AI 生成解读 → 推送预警到飞书/钉钉
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
from src.anomaly import (
    detect_anomalies_3sigma,
    detect_anomalies_bollinger,
    detect_anomalies_cusum,
    generate_anomaly_digest,
    generate_metric_data,
)
from src.viz import plot_anomaly_timeseries, plot_funnel


# ============================================================================
# Step 1: 生成模拟的 90 天漏斗数据
# ============================================================================

print("=" * 65)
print("  Step 1: 模拟 90 天漏斗数据")
print("=" * 65)

n_days = 90
rng = np.random.default_rng(42)

# 定义异常日（模拟系统故障 / 竞品冲击）
anomaly_days = [25, 26, 55, 78]

# 各阶段漏斗
impressions = generate_metric_data(n_days, baseline=50000, noise_std=3000,
                                   anomaly_days=None, trend=50,
                                   seasonality_amplitude=5000, seed=42)

detail_rate = generate_metric_data(n_days, baseline=0.35, noise_std=0.02,
                                   anomaly_days=anomaly_days, anomaly_magnitude=0.08,
                                   seed=43)
detail_rate = np.clip(detail_rate, 0.05, 0.95)

cart_rate = generate_metric_data(n_days, baseline=0.20, noise_std=0.015,
                                  anomaly_days=anomaly_days, anomaly_magnitude=0.05,
                                  seed=44)
cart_rate = np.clip(cart_rate, 0.02, 0.80)

order_rate = generate_metric_data(n_days, baseline=0.50, noise_std=0.03,
                                   anomaly_days=anomaly_days, anomaly_magnitude=0.12,
                                   seed=45)
order_rate = np.clip(order_rate, 0.05, 0.95)

pay_rate = generate_metric_data(n_days, baseline=0.85, noise_std=0.03,
                                 anomaly_days=anomaly_days, anomaly_magnitude=0.15,
                                 seed=46)
pay_rate = np.clip(pay_rate, 0.10, 0.99)

print(f"  已生成 {n_days} 天数据，注入异常日: {anomaly_days}")


# ============================================================================
# Step 2: 多方法联合异常检测
# ============================================================================

print("\n" + "=" * 65)
print("  Step 2: 联合异常检测 (3-Sigma + Bollinger + CUSUM)")
print("=" * 65)

# 对加购转化率做联合检测
digest = generate_anomaly_digest(cart_rate, metric_name="加购转化率")

print(f"""
  📊 加购转化率异常检测综合报告
  ─────────────────────────────────
  总数据点:    {digest['total_points']}
  3-Sigma:     {digest['individual_results']['3sigma'].anomalies_count} 个异常点
  Bollinger:   {digest['individual_results']['bollinger'].anomalies_count} 个异常点
  CUSUM:       {digest['individual_results']['cusum'].anomalies_count} 个异常点
  ─────────────────────────────────
  🔴 高置信异常 (≥2 方法一致): {digest['high_confidence_count']} 个
  异常日索引:  {digest['high_confidence_anomalies']}
""")

# 判定严重程度
severity = "🟢 Normal"
if digest["high_confidence_count"] >= 3:
    severity = "🔴 Critical — 需要立即排查！"
elif digest["high_confidence_count"] >= 1:
    severity = "🟡 Warning — 建议重点关注"

print(f"  → 当前状态: {severity}")


# ============================================================================
# Step 3: 全漏斗多指标扫描
# ============================================================================

print("\n" + "=" * 65)
print("  Step 3: 全漏斗多指标扫描")
print("=" * 65)

funnel_metrics = {
    "浏览转化率 (曝光→详情)": detail_rate,
    "加购转化率 (详情→加购)": cart_rate,
    "下单转化率 (加购→下单)": order_rate,
    "支付转化率 (下单→支付)": pay_rate,
}

all_alerts = []
for name, data in funnel_metrics.items():
    r = detect_anomalies_3sigma(data, metric_name=name, window=14, n_sigmas=2.5)
    if r.anomalies_count > 0:
        all_alerts.append(r)
        print(f"  {r.severity.upper():<10} | {name:<30} | {r.anomalies_count:>3} 异常点 | "
              f"异常率 {r.anomaly_rate:.1%}")

if not all_alerts:
    print("  ✅ 所有指标正常")

# ============================================================================
# Step 4: 漏斗健康度快照
# ============================================================================

print("\n" + "=" * 65)
print("  Step 4: 漏斗健康度快照")
print("=" * 65)

# 最近 7 天的均值
recent_days = slice(-7, None)
funnel_stages = ["曝光", "详情页浏览", "加购", "下单", "支付"]

# 从曝光开始估算各阶段绝对值（用于可视化）
avg_impressions = int(np.mean(impressions[recent_days]))
avg_detail = int(avg_impressions * np.mean(detail_rate[recent_days]))
avg_cart = int(avg_detail * np.mean(cart_rate[recent_days]))
avg_order = int(avg_cart * np.mean(order_rate[recent_days]))
avg_pay = int(avg_order * np.mean(pay_rate[recent_days]))

funnel_values = [avg_impressions, avg_detail, avg_cart, avg_order, avg_pay]

print(f"\n  近7日平均漏斗 (n={n_days}天数据):")
for stage, val in zip(funnel_stages, funnel_values):
    rate = val / funnel_values[0]
    print(f"    {stage:<12}: {val:>8,}  ({rate:.2%})")

fig_funnel = plot_funnel(funnel_stages, funnel_values,
                          title="Product Funnel Health (Last 7 Days)")
output_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(output_dir, exist_ok=True)
fig_funnel.savefig(os.path.join(output_dir, "funnel_health.png"), dpi=150, bbox_inches="tight")

# ============================================================================
# Step 5: 异常可视化
# ============================================================================

print("\n" + "=" * 65)
print("  Step 5: 生成异常监控图表")
print("=" * 65)

# 对加购转化率做 Bollinger 检测并画图
r, upper, lower, ma = detect_anomalies_bollinger(cart_rate, metric_name="加购转化率",
                                                  window=14, n_std=2.0)

fig_anomaly = plot_anomaly_timeseries(
    cart_rate,
    anomaly_indices=r.anomaly_indices,
    upper_band=upper,
    lower_band=lower,
    moving_avg=ma,
    metric_name="加购转化率 (Cart Rate)",
)
fig_anomaly.savefig(os.path.join(output_dir, "anomaly_monitor.png"), dpi=150, bbox_inches="tight")

print(f"  图表已保存至: {output_dir}/")
print("  - funnel_health.png")
print("  - anomaly_monitor.png")


# ============================================================================
# Step 6: 自动预警报告 (模拟 AI 生成)
# ============================================================================

print("\n" + "=" * 65)
print("  Step 6: AI-Native 自动预警报告")
print("=" * 65)

critical_metrics = [a for a in all_alerts if a.severity == "critical"]
warning_metrics = [a for a in all_alerts if a.severity == "warning"]

report = f"""
╔══════════════════════════════════════════════════════════╗
║           📊 每日指标监控自动报告 (AI-Generated)           ║
╠══════════════════════════════════════════════════════════╣
║  监控时间: Day 1 ~ Day {n_days}                              ║
║  监控指标: {len(funnel_metrics)} 个漏斗指标                      ║
║  检测方法: 3-Sigma + Bollinger + CUSUM                    ║
╠══════════════════════════════════════════════════════════╣
║  🔴 Critical:  {len(critical_metrics)} 个指标异常                     ║
║  🟡 Warning:   {len(warning_metrics)} 个指标异常                     ║
║  🟢 Normal:    {len(funnel_metrics) - len(critical_metrics) - len(warning_metrics)} 个指标正常                      ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  🤖 AI 分析摘要:                                          ║
║  建议排查方向：                                             ║
║  1. 检查异常日 ({anomaly_days}) 是否有系统发布或运营活动         ║
║  2. 重点看加购→下单环节，该环节对整体漏斗影响最大                  ║
║  3. 如为外部竞品冲击，建议对比竞品同期数据                          ║
║                                                          ║
║  ⚡ 本报告由 AI-Native Pipeline 自动生成                    ║
║  Next: 接入飞书/钉钉 Webhook 实现自动推送                   ║
╚══════════════════════════════════════════════════════════╝
"""

print(report)

print("\n" + "=" * 65)
print("  🎯 AI-Native 异常监控 Checklist")
print("=" * 65)
print("""
  1. 多方法联合检测 → 降低误报率
  2. 严重程度分级 → 避免告警疲劳
  3. 全漏斗扫描 → 快速定位问题环节
  4. AI 自动生成解读 → 减少人工分析时间
  5. 图表自动输出 → Stakeholder 自助查看

  🚀 进阶方向:
  - 接入真实数据源 (MySQL/ClickHouse)
  - 用 Claude API 自动生成异常解读文本
  - 飞书/钉钉 Webhook 自动推送
  - 基于历史告警训练个性化阈值
""")
