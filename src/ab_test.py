"""
A/B Testing & Statistical Inference Toolkit
============================================
频率学派与贝叶斯方法的 A/B 测试工具模块。

包含:
- 最小样本量估算
- 双样本 t 检验 & z 检验
- 贝叶斯 A/B 测试
- 多重检验校正 (Bonferroni, FDR)
- 序贯检验 (Sequential Testing)
- 置信区间 & 效应量

Author: AI-Native Strategy Analytics
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import numpy as np
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ABTestResult:
    """A/B 测试结果的标准输出"""
    metric_name: str
    control_mean: float
    treatment_mean: float
    relative_lift: float          # (treat - control) / control
    abs_lift: float               # treat - control
    p_value: float
    ci_lower: float               # 95% CI lower bound
    ci_upper: float               # 95% CI upper bound
    sample_size_control: int
    sample_size_treatment: int
    significant: bool             # p < 0.05
    practical_significant: bool   # 是否超过实际显著性阈值
    test_method: str = "two-sided t-test"
    effect_size_cohens_d: float = 0.0
    power: float = 0.0

    def summary(self) -> str:
        verdict = (
            "✅ 显著 (Statistically Significant)"
            if self.significant
            else "❌ 不显著 (Not Significant)"
        )
        return (
            f"\n{'='*60}\n"
            f"  A/B Test Report: {self.metric_name}\n"
            f"{'='*60}\n"
            f"  Control   (n={self.sample_size_control}): {self.control_mean:.4f}\n"
            f"  Treatment (n={self.sample_size_treatment}): {self.treatment_mean:.4f}\n"
            f"  Absolute Lift: {self.abs_lift:.4f}\n"
            f"  Relative Lift: {self.relative_lift:.2%}\n"
            f"  95% CI:       [{self.ci_lower:.4f}, {self.ci_upper:.4f}]\n"
            f"  Cohen's d:    {self.effect_size_cohens_d:.3f}\n"
            f"  p-value:      {self.p_value:.4f}\n"
            f"  Power:        {self.power:.2%}\n"
            f"  Verdict:      {verdict}\n"
            f"{'='*60}\n"
        )


# ---------------------------------------------------------------------------
# Sample size estimation
# ---------------------------------------------------------------------------

def estimate_sample_size(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    tail: str = "two-sided",
) -> int:
    """
    估算 A/B 测试所需的最小每组样本量。

    Parameters
    ----------
    baseline_rate : float
        对照组基准转化率 (0 ~ 1)
    mde : float
        最小可检测效应 (Minimum Detectable Effect)，绝对差值
    alpha : float
        第一类错误率 (默认 0.05)
    power : float
        统计功效 (默认 0.80)
    tail : str
        "two-sided" 或 "one-sided"

    Returns
    -------
    int
        每组所需最小样本量

    Example
    -------
    >>> estimate_sample_size(baseline_rate=0.10, mde=0.02)
    3522  # 每组需要约 3522 个样本
    """
    if tail == "two-sided":
        z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = sp_stats.norm.ppf(1 - alpha)

    z_beta = sp_stats.norm.ppf(power)

    p1 = baseline_rate
    p2 = baseline_rate + mde

    # Pooled variance approximation for proportions
    p_bar = (p1 + p2) / 2
    variance = 2 * p_bar * (1 - p_bar)

    n = ((z_alpha + z_beta) ** 2 * variance) / (mde**2)
    return int(np.ceil(n))


def estimate_sample_size_continuous(
    baseline_mean: float,
    baseline_std: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    连续指标的最小样本量估算。
    用于收入、时长等连续型指标。
    """
    z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
    z_beta = sp_stats.norm.ppf(power)
    n = 2 * (baseline_std**2) * ((z_alpha + z_beta) ** 2) / (mde**2)
    return int(np.ceil(n))


# ---------------------------------------------------------------------------
# Frequentist A/B Test
# ---------------------------------------------------------------------------

def two_sample_ttest(
    control: np.ndarray,
    treatment: np.ndarray,
    metric_name: str = "metric",
    alpha: float = 0.05,
    practical_threshold: float = 0.0,
) -> ABTestResult:
    """
    双样本独立 t 检验 (Welch's t-test)。

    适用于连续型指标 (如 ARPU、使用时长)。
    """
    n_c, n_t = len(control), len(treatment)
    mean_c, mean_t = np.mean(control), np.mean(treatment)
    std_c, std_t = np.std(control, ddof=1), np.std(treatment, ddof=1)

    # Welch's t-test (不假设方差齐性)
    t_stat, p_value = sp_stats.ttest_ind(control, treatment, equal_var=False)

    # Confidence interval for the difference
    se = np.sqrt(std_c**2 / n_c + std_t**2 / n_t)
    margin = sp_stats.t.ppf(1 - alpha / 2, df=min(n_c, n_t) - 1) * se
    diff = mean_t - mean_c
    ci_lower, ci_upper = diff - margin, diff + margin

    # Cohen's d
    pooled_std = np.sqrt((std_c**2 + std_t**2) / 2)
    cohens_d = diff / pooled_std if pooled_std > 0 else 0.0

    # Power (post-hoc)
    ncp = diff / se if se > 0 else 0
    df = min(n_c, n_t) - 1
    power = 1 - sp_stats.nct.cdf(sp_stats.t.ppf(1 - alpha / 2, df), df, ncp)

    rel_lift = diff / mean_c if mean_c != 0 else 0.0

    return ABTestResult(
        metric_name=metric_name,
        control_mean=mean_c,
        treatment_mean=mean_t,
        relative_lift=rel_lift,
        abs_lift=diff,
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        sample_size_control=n_c,
        sample_size_treatment=n_t,
        significant=p_value < alpha,
        practical_significant=abs(diff) > practical_threshold,
        test_method="Welch's t-test",
        effect_size_cohens_d=cohens_d,
        power=power,
    )


def proportions_ztest(
    successes_c: int, n_c: int,
    successes_t: int, n_t: int,
    metric_name: str = "conversion_rate",
    alpha: float = 0.05,
) -> ABTestResult:
    """
    双样本比率 z 检验。

    适用于二值指标 (如转化率、留存率)。
    """
    p_c = successes_c / n_c
    p_t = successes_t / n_t
    diff = p_t - p_c

    # Pooled proportion
    p_pool = (successes_c + successes_t) / (n_c + n_t)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))

    z_stat = diff / se if se > 0 else 0
    p_value = 2 * (1 - sp_stats.norm.cdf(abs(z_stat)))

    # CI for difference of proportions
    se_unpooled = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    ci_lower = diff - z_crit * se_unpooled
    ci_upper = diff + z_crit * se_unpooled

    # Cohen's h
    cohens_h = 2 * (np.arcsin(np.sqrt(p_t)) - np.arcsin(np.sqrt(p_c)))

    rel_lift = diff / p_c if p_c > 0 else 0.0

    return ABTestResult(
        metric_name=metric_name,
        control_mean=p_c,
        treatment_mean=p_t,
        relative_lift=rel_lift,
        abs_lift=diff,
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        sample_size_control=n_c,
        sample_size_treatment=n_t,
        significant=p_value < alpha,
        practical_significant=abs(rel_lift) > 0.01,
        test_method="two-proportion z-test",
        effect_size_cohens_d=cohens_h,
        power=0.0,  # analytic power not computed here
    )


# ---------------------------------------------------------------------------
# Bayesian A/B Test (simple Beta-Binomial model)
# ---------------------------------------------------------------------------

@dataclass
class BayesianABResult:
    """贝叶斯 A/B 测试结果"""
    prob_b_better_than_a: float  # P(θ_treat > θ_control)
    expected_loss: float          # 如果选错，期望损失
    control_posterior_mean: float
    treatment_posterior_mean: float
    cred_interval: Tuple[float, float]

    def summary(self) -> str:
        return (
            f"Bayesian A/B Test:\n"
            f"  P(Treatment > Control): {self.prob_b_better_than_a:.2%}\n"
            f"  Expected Loss:          {self.expected_loss:.4f}\n"
            f"  Control  Posterior Mean: {self.control_posterior_mean:.4f}\n"
            f"  Treatment Posterior Mean: {self.treatment_posterior_mean:.4f}\n"
        )


def bayesian_ab_test_beta_binomial(
    successes_c: int, n_c: int,
    successes_t: int, n_t: int,
    prior_c: Tuple[float, float] = (1, 1),
    prior_t: Tuple[float, float] = (1, 1),
    n_samples: int = 100_000,
) -> BayesianABResult:
    """
    基于 Beta-Binomial 共轭模型的贝叶斯 A/B 测试。

    使用 Monte Carlo 采样估计 P(θ_treat > θ_control)。
    默认使用 Uniform(0,1) 先验 (Beta(1,1))。
    """
    alpha_c, beta_c = prior_c[0] + successes_c, prior_c[1] + (n_c - successes_c)
    alpha_t, beta_t = prior_t[0] + successes_t, prior_t[1] + (n_t - successes_t)

    samples_c = np.random.beta(alpha_c, beta_c, n_samples)
    samples_t = np.random.beta(alpha_t, beta_t, n_samples)

    prob_b_better = np.mean(samples_t > samples_c)
    diff_samples = samples_t - samples_c
    expected_loss = np.mean(np.maximum(-diff_samples, 0))

    lower, upper = np.percentile(diff_samples, [2.5, 97.5])

    return BayesianABResult(
        prob_b_better_than_a=prob_b_better,
        expected_loss=expected_loss,
        control_posterior_mean=alpha_c / (alpha_c + beta_c),
        treatment_posterior_mean=alpha_t / (alpha_t + beta_t),
        cred_interval=(lower, upper),
    )


# ---------------------------------------------------------------------------
# Multiple Testing Correction
# ---------------------------------------------------------------------------

def bonferroni_correction(p_values: List[float]) -> List[float]:
    """Bonferroni 校正 — 最保守"""
    m = len(p_values)
    return [min(p * m, 1.0) for p in p_values]


def bh_fdr_correction(p_values: List[float], alpha: float = 0.05) -> Dict:
    """
    Benjamini-Hochberg FDR 校正。
    返回校正后的 p 值及哪些假设被拒绝。
    """
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    # BH critical values
    bh_critical = np.array([(i + 1) / m * alpha for i in range(m)])

    # Find the largest k where p_(k) <= critical_(k)
    significant = sorted_p <= bh_critical
    if np.any(significant):
        max_k = np.max(np.where(significant)[0])
        rejected_indices = sorted_indices[: max_k + 1]
    else:
        rejected_indices = np.array([], dtype=int)

    # FDR-adjusted p-values
    adjusted = np.ones(m)
    for i in range(m - 1, -1, -1):
        if i == m - 1:
            adjusted[sorted_indices[i]] = sorted_p[i]
        else:
            adjusted[sorted_indices[i]] = min(
                adjusted[sorted_indices[i + 1]],
                sorted_p[i] * m / (i + 1),
            )

    return {
        "original_p_values": p_values,
        "adjusted_p_values": list(adjusted),
        "rejected": [i in rejected_indices for i in range(m)],
        "method": "Benjamini-Hochberg FDR",
    }


# ---------------------------------------------------------------------------
# Sequential Testing (alpha spending — simple Pocock boundary)
# ---------------------------------------------------------------------------

def sequential_pocock_boundary(n_looks: int, alpha: float = 0.05) -> List[float]:
    """
    Pocock 序贯检验边界：每次中期分析使用相同的 alpha。
    适用于有 peek 需求的 A/B 测试场景。
    """
    # Pocock 近似：每次 look 的 nominal alpha
    nominal = 1 - (1 - alpha) ** (1 / n_looks)
    # 更精确的 Pocock boundary z-value (近似)
    z_crit = sp_stats.norm.ppf(1 - nominal / 2)
    return [z_crit] * n_looks


def sequential_check(
    looks: List[Tuple[int, int, float, float]],  # [(n_c, n_t, mean_c, mean_t), ...]
    n_looks_planned: int = 5,
    alpha: float = 0.05,
) -> List[Dict]:
    """
    序贯检验的多次中期分析。
    每一步返回是否可提前停止。

    looks: [(对照组样本量, 实验组样本量, 对照组均值, 实验组均值), ...]
    每次 peek 传入截止目前的累计数据。
    """
    boundaries = sequential_pocock_boundary(n_looks_planned, alpha)
    results = []

    for i, (n_c, n_t, mean_c, mean_t) in enumerate(looks):
        if i >= len(boundaries):
            break
        se = np.sqrt(2 / min(n_c, n_t))  # rough SE approximation
        z_stat = (mean_t - mean_c) / se if se > 0 else 0
        z_crit = boundaries[i]
        stop_early = abs(z_stat) > z_crit

        results.append({
            "look": i + 1,
            "n_control": n_c,
            "n_treatment": n_t,
            "z_statistic": round(z_stat, 3),
            "z_critical": round(z_crit, 3),
            "stop_early": stop_early,
            "direction": "treatment" if z_stat > 0 else "control",
        })

    return results


# ---------------------------------------------------------------------------
# Utility: simulate A/B test data
# ---------------------------------------------------------------------------

def simulate_ab_data(
    n_control: int = 5000,
    n_treatment: int = 5000,
    baseline_rate: float = 0.10,
    lift: float = 0.02,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成模拟 A/B 测试数据。

    Returns
    -------
    control : np.ndarray  (0/1 binary outcomes)
    treatment : np.ndarray (0/1 binary outcomes)
    """
    rng = np.random.default_rng(seed)
    control = (rng.random(n_control) < baseline_rate).astype(int)
    treatment = (rng.random(n_treatment) < baseline_rate + lift).astype(int)
    return control, treatment


def simulate_ab_data_continuous(
    n_control: int = 5000,
    n_treatment: int = 5000,
    baseline_mean: float = 25.0,
    baseline_std: float = 10.0,
    lift: float = 1.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """生成连续型 A/B 测试数据（如收入、时长）"""
    rng = np.random.default_rng(seed)
    control = rng.normal(baseline_mean, baseline_std, n_control)
    treatment = rng.normal(baseline_mean + lift, baseline_std, n_treatment)
    return control, treatment
