"""
Metric Anomaly Detection & Monitoring
=====================================
指标异常检测与监控模块。

包含:
- 3-Sigma 异常检测
- 移动平均 + Bollinger Band 风格检测
- CUSUM (Cumulative Sum) 漂移检测
- 简单 Isolation Forest 集成
- 异常报告自动生成

Author: AI-Native Strategy Analytics
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from scipy import stats as sp_stats


@dataclass
class AnomalyReport:
    """异常检测报告"""
    metric_name: str
    total_points: int
    anomalies_count: int
    anomaly_rate: float
    anomaly_indices: List[int]
    anomaly_values: List[float]
    threshold_upper: float
    threshold_lower: float
    method: str
    severity: str = "normal"  # normal, warning, critical

    def summary(self) -> str:
        sev_emoji = {"normal": "🟢", "warning": "🟡", "critical": "🔴"}
        return (
            f"\n{'='*55}\n"
            f"  {sev_emoji.get(self.severity, '⚪')} Anomaly Report: {self.metric_name}\n"
            f"{'='*55}\n"
            f"  Method:         {self.method}\n"
            f"  Total Points:   {self.total_points}\n"
            f"  Anomalies:      {self.anomalies_count} ({self.anomaly_rate:.1%})\n"
            f"  Threshold:      [{self.threshold_lower:.2f}, {self.threshold_upper:.2f}]\n"
            f"  Severity:       {self.severity.upper()}\n"
            f"{'='*55}\n"
        )


# ---------------------------------------------------------------------------
# 3-Sigma Rule
# ---------------------------------------------------------------------------

def detect_anomalies_3sigma(
    data: np.ndarray,
    metric_name: str = "metric",
    window: int = 30,
    n_sigmas: float = 3.0,
) -> AnomalyReport:
    """
    基于滚动 3-Sigma 的异常检测。

    对每个点，用前 window 个点计算均值和标准差。
    如果当前点偏离均值超过 n_sigmas 个标准差，标记为异常。
    """
    anomalies_idx = []
    anomalies_val = []
    n = len(data)

    for i in range(window, n):
        window_data = data[i - window : i]
        mu, std = np.mean(window_data), np.std(window_data, ddof=1)
        if std == 0:
            continue
        z_score = abs(data[i] - mu) / std
        if z_score > n_sigmas:
            anomalies_idx.append(i)
            anomalies_val.append(data[i])

    # Global thresholds for reporting
    global_mu = np.mean(data)
    global_std = np.std(data, ddof=1)

    severity = "normal"
    anomaly_rate = len(anomalies_idx) / n
    if anomaly_rate > 0.10:
        severity = "critical"
    elif anomaly_rate > 0.05:
        severity = "warning"

    return AnomalyReport(
        metric_name=metric_name,
        total_points=n,
        anomalies_count=len(anomalies_idx),
        anomaly_rate=anomaly_rate,
        anomaly_indices=anomalies_idx,
        anomaly_values=anomalies_val,
        threshold_upper=global_mu + n_sigmas * global_std,
        threshold_lower=global_mu - n_sigmas * global_std,
        method=f"Rolling {n_sigmas}-Sigma (window={window})",
        severity=severity,
    )


# ---------------------------------------------------------------------------
# Moving Average + Bollinger-style Bands
# ---------------------------------------------------------------------------

def detect_anomalies_bollinger(
    data: np.ndarray,
    metric_name: str = "metric",
    window: int = 20,
    n_std: float = 2.0,
) -> Tuple[AnomalyReport, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bollinger Band 风格异常检测。

    Returns
    -------
    report : AnomalyReport
    upper_band, lower_band, moving_avg : np.ndarray
        用于可视化的上下轨和均线
    """
    n = len(data)
    moving_avg = np.full(n, np.nan)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    anomalies_idx = []
    anomalies_val = []

    for i in range(window - 1, n):
        window_data = data[i - window + 1 : i + 1]
        ma = np.mean(window_data)
        std = np.std(window_data, ddof=1)
        moving_avg[i] = ma
        upper_band[i] = ma + n_std * std
        lower_band[i] = ma - n_std * std

        if data[i] > upper_band[i] or data[i] < lower_band[i]:
            anomalies_idx.append(i)
            anomalies_val.append(data[i])

    report = AnomalyReport(
        metric_name=metric_name,
        total_points=n,
        anomalies_count=len(anomalies_idx),
        anomaly_rate=len(anomalies_idx) / n,
        anomaly_indices=anomalies_idx,
        anomaly_values=anomalies_val,
        threshold_upper=np.nanmean(upper_band),
        threshold_lower=np.nanmean(lower_band),
        method=f"Bollinger Bands (window={window}, {n_std}σ)",
    )
    return report, upper_band, lower_band, moving_avg


# ---------------------------------------------------------------------------
# CUSUM (Cumulative Sum) — 检测均值漂移
# ---------------------------------------------------------------------------

def detect_anomalies_cusum(
    data: np.ndarray,
    metric_name: str = "metric",
    threshold: float = 5.0,
    drift: float = 0.5,
) -> AnomalyReport:
    """
    CUSUM 漂移检测。
    检测过程均值的持续性变化（向上或向下漂移）。

    Parameters
    ----------
    threshold : float
        报警阈值。CUSUM 超过此值触发检测。
    drift : float
        允许的漂移量（越小越敏感）。
    """
    n = len(data)
    mu = np.mean(data)

    cusum_pos = np.zeros(n)
    cusum_neg = np.zeros(n)
    anomalies_idx = []
    anomalies_val = []

    for i in range(1, n):
        cusum_pos[i] = max(0, cusum_pos[i - 1] + (data[i] - mu) - drift)
        cusum_neg[i] = min(0, cusum_neg[i - 1] + (data[i] - mu) + drift)

        if cusum_pos[i] > threshold or cusum_neg[i] < -threshold:
            anomalies_idx.append(i)
            anomalies_val.append(data[i])

    return AnomalyReport(
        metric_name=metric_name,
        total_points=n,
        anomalies_count=len(anomalies_idx),
        anomaly_rate=len(anomalies_idx) / n if n > 0 else 0,
        anomaly_indices=anomalies_idx,
        anomaly_values=anomalies_val,
        threshold_upper=threshold,
        threshold_lower=-threshold,
        method=f"CUSUM (threshold={threshold}, drift={drift})",
    )


# ---------------------------------------------------------------------------
# Isolation Forest — 多维异常检测
# ---------------------------------------------------------------------------

def detect_anomalies_iforest(
    df: pd.DataFrame,
    feature_cols: List[str],
    contamination: float = 0.05,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict]:
    """
    基于 Isolation Forest 的多维异常检测。

    适用于多指标联合监控场景。
    """
    from sklearn.ensemble import IsolationForest

    X = df[feature_cols].values
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
    )
    predictions = model.fit_predict(X)
    # predictions: 1 = normal, -1 = anomaly
    anomaly_mask = predictions == -1

    return anomaly_mask, {
        "n_total": len(df),
        "n_anomalies": int(np.sum(anomaly_mask)),
        "anomaly_rate": float(np.mean(anomaly_mask)),
        "feature_cols": feature_cols,
        "method": "Isolation Forest",
    }


# ---------------------------------------------------------------------------
# Automated Anomaly Report Generation
# ---------------------------------------------------------------------------

def generate_anomaly_digest(
    data: np.ndarray,
    metric_name: str = "metric",
    methods: Optional[List[str]] = None,
) -> Dict:
    """
    多方法联合异常检测，生成综合报告。

    使用方法: 3-Sigma + Bollinger + CUSUM 三管齐下。
    至少被两种方法检测到的点视为高置信异常。
    """
    if methods is None:
        methods = ["3sigma", "bollinger", "cusum"]

    results = {}
    vote_counts = np.zeros(len(data), dtype=int)

    if "3sigma" in methods:
        r = detect_anomalies_3sigma(data, metric_name)
        results["3sigma"] = r
        for idx in r.anomaly_indices:
            vote_counts[idx] += 1

    if "bollinger" in methods:
        r, _, _, _ = detect_anomalies_bollinger(data, metric_name)
        results["bollinger"] = r
        for idx in r.anomaly_indices:
            vote_counts[idx] += 1

    if "cusum" in methods:
        r = detect_anomalies_cusum(data, metric_name)
        results["cusum"] = r
        for idx in r.anomaly_indices:
            vote_counts[idx] += 1

    high_confidence = np.where(vote_counts >= 2)[0]

    return {
        "metric_name": metric_name,
        "total_points": len(data),
        "individual_results": results,
        "high_confidence_anomalies": list(high_confidence),
        "high_confidence_count": len(high_confidence),
        "vote_counts": vote_counts.tolist(),
    }


# ---------------------------------------------------------------------------
# Utility: generate demo time-series data
# ---------------------------------------------------------------------------

def generate_metric_data(
    n_days: int = 90,
    baseline: float = 100.0,
    noise_std: float = 5.0,
    anomaly_days: Optional[List[int]] = None,
    anomaly_magnitude: float = 20.0,
    trend: float = 0.0,
    seasonality_amplitude: float = 3.0,
    seasonality_period: int = 7,
    seed: int = 42,
) -> np.ndarray:
    """
    生成带异常的模拟时序指标数据。

    Parameters
    ----------
    n_days : int — 天数
    baseline : float — 基线值
    noise_std : float — 白噪声标准差
    anomaly_days : List[int] — 哪些天注入异常
    anomaly_magnitude : float — 异常幅度
    trend : float — 每日趋势
    seasonality_amplitude : float — 周期性幅度
    seasonality_period : int — 周期（默认7天）
    seed : int
    """
    rng = np.random.default_rng(seed)
    days = np.arange(n_days)

    # Components
    trend_component = trend * days
    seasonality = seasonality_amplitude * np.sin(2 * np.pi * days / seasonality_period)
    noise = rng.normal(0, noise_std, n_days)

    data = baseline + trend_component + seasonality + noise

    # Inject anomalies
    if anomaly_days:
        for day in anomaly_days:
            if 0 <= day < n_days:
                direction = rng.choice([-1, 1])
                data[day] += direction * anomaly_magnitude

    return data
