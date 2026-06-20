"""
Visualization Utilities
=======================
策略分析可视化工具模块。

包含:
- A/B 测试结果可视化
- 漏斗对比图
- 时序异常标注图
- 因果推断效应图
- 风格统一的 matplotlib/seaborn 主题

Author: AI-Native Strategy Analytics
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，适合脚本/CI/CD 环境
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
import seaborn as sns
from typing import Optional, List, Tuple
import platform

# ---------------------------------------------------------------------------
# CJK Font Setup (Windows: Microsoft YaHei, macOS: PingFang, Linux: Noto Sans CJK)
# ---------------------------------------------------------------------------

def _find_cjk_font() -> str:
    """Find an available CJK font on the system."""
    candidates = [
        "Microsoft YaHei", "SimHei", "PingFang SC", "PingFang TC",
        "Noto Sans CJK SC", "Noto Sans CJK TC", "WenQuanYi Micro Hei",
        "Source Han Sans SC", "Hiragino Sans GB", "STHeiti",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            return font
    return "sans-serif"

_CJK_FONT = _find_cjk_font()

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.figsize": (10, 5),
    "font.family": "sans-serif",
    "font.sans-serif": [_CJK_FONT, "Arial", "DejaVu Sans"],
    "axes.unicode_minus": False,  # 防止负号显示为方块
})
sns.set_style("whitegrid")

print(f"[viz] Using CJK font: {_CJK_FONT}")


# ---------------------------------------------------------------------------
# A/B Test Visualizations
# ---------------------------------------------------------------------------

def plot_ab_comparison(
    control: np.ndarray,
    treatment: np.ndarray,
    metric_name: str = "Metric",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制 A/B 测试对照组 vs 实验组分布对比图。
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: KDE + histogram
    ax = axes[0]
    sns.histplot(control, kde=True, color="#4C72B0", alpha=0.4, label="Control", ax=ax, stat="density")
    sns.histplot(treatment, kde=True, color="#DD8452", alpha=0.4, label="Treatment", ax=ax, stat="density")
    ax.axvline(np.mean(control), color="#4C72B0", linestyle="--", linewidth=1.5)
    ax.axvline(np.mean(treatment), color="#DD8452", linestyle="--", linewidth=1.5)
    ax.set_title(f"{metric_name} Distribution")
    ax.set_xlabel(metric_name)
    ax.legend()

    # Right: Box plot
    ax2 = axes[1]
    box_data = [control, treatment]
    bp = ax2.boxplot(box_data, labels=["Control", "Treatment"], patch_artist=True,
                     widths=0.4)
    bp["boxes"][0].set_facecolor("#4C72B0")
    bp["boxes"][1].set_facecolor("#DD8452")
    ax2.set_title(f"{metric_name} Box Plot")
    ax2.set_ylabel(metric_name)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_ab_confidence_interval(
    result,  # ABTestResult
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制 A/B 测试效应量的置信区间。
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    # Plot CI
    ax.errorbar(
        x=result.abs_lift,
        y=0,
        xerr=[[result.abs_lift - result.ci_lower], [result.ci_upper - result.abs_lift]],
        fmt="o",
        color="#DD8452",
        capsize=8,
        markersize=10,
        linewidth=2,
    )

    ax.set_xlabel(f"Absolute Lift ({result.metric_name})")
    ax.set_title(f"A/B Test: {result.metric_name}\n"
                 f"Lift = {result.abs_lift:.4f}, "
                 f"95% CI = [{result.ci_lower:.4f}, {result.ci_upper:.4f}], "
                 f"p = {result.p_value:.4f}")
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Funnel Visualization
# ---------------------------------------------------------------------------

def plot_funnel(
    stages: List[str],
    values: List[float],
    title: str = "Conversion Funnel",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制转化漏斗图。
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    n = len(stages)
    colors = sns.color_palette("viridis", n)

    y_pos = range(n)
    max_val = values[0]

    for i, (stage, val) in enumerate(zip(stages, values)):
        width = val / max_val
        ax.barh(i, width, color=colors[i], alpha=0.85, height=0.6)
        ax.text(width + 0.02, i, f"{stage}: {val:,.0f} ({val/max_val:.1%})",
                va="center", fontsize=10)

        if i < n - 1:
            drop = values[i] - values[i + 1]
            drop_rate = drop / values[i]
            ax.text(0.5, i + 0.5, f"↓ {drop_rate:.1%} drop",
                    va="center", ha="center", fontsize=8, color="red", alpha=0.7)

    ax.set_xlim(0, 1.2)
    ax.set_yticks([])
    ax.set_xlabel("Proportion Remaining")
    ax.set_title(title)
    ax.invert_yaxis()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Time Series with Anomaly Highlighting
# ---------------------------------------------------------------------------

def plot_anomaly_timeseries(
    data: np.ndarray,
    anomaly_indices: List[int],
    upper_band: Optional[np.ndarray] = None,
    lower_band: Optional[np.ndarray] = None,
    moving_avg: Optional[np.ndarray] = None,
    metric_name: str = "Metric",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制时序数据并标注异常点。
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    days = np.arange(len(data))
    ax.plot(days, data, color="#4C72B0", linewidth=1.2, alpha=0.8, label=metric_name)

    if moving_avg is not None:
        ax.plot(days, moving_avg, color="#55A868", linewidth=1.5, label="Moving Avg",
                alpha=0.7)

    if upper_band is not None and lower_band is not None:
        ax.fill_between(days, lower_band, upper_band, alpha=0.15, color="gray",
                        label="Normal Range")

    if anomaly_indices:
        anomaly_values = data[anomaly_indices]
        ax.scatter(anomaly_indices, anomaly_values, color="#C44E52", s=60, zorder=5,
                   edgecolors="darkred", linewidths=0.8, label=f"Anomalies ({len(anomaly_indices)})")

    ax.set_xlabel("Time (days)")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} — Anomaly Detection Report")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Causal Effect Visualization
# ---------------------------------------------------------------------------

def plot_did(
    pre_control: float,
    pre_treated: float,
    post_control: float,
    post_treated: float,
    metric_name: str = "Outcome",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制 Difference-in-Differences 示意图。
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Control line
    ax.plot([0, 1], [pre_control, post_control], "o-", color="#4C72B0",
            linewidth=2.5, markersize=10, label="Control")
    # Treated line
    ax.plot([0, 1], [pre_treated, post_treated], "o-", color="#DD8452",
            linewidth=2.5, markersize=10, label="Treatment")

    # Counterfactual (dashed)
    counterfactual_post = pre_treated + (post_control - pre_control)
    ax.plot([0, 1], [pre_treated, counterfactual_post], "--", color="#DD8452",
            linewidth=1.5, alpha=0.5, label="Counterfactual")

    # DiD arrow
    did = (post_treated - pre_treated) - (post_control - pre_control)
    ax.annotate(
        f"DiD = {did:.2f}",
        xy=(1, (post_treated + counterfactual_post) / 2),
        xytext=(1.3, (post_treated + counterfactual_post) / 2),
        arrowprops=dict(arrowstyle="->", color="green", lw=2),
        fontsize=12,
        color="green",
        fontweight="bold",
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre-Treatment", "Post-Treatment"])
    ax.set_ylabel(metric_name)
    ax.set_title("Difference-in-Differences (DiD)")
    ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
