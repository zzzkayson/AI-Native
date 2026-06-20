"""
Causal Inference Toolkit
========================
因果推断核心方法模块。

包含:
- Difference-in-Differences (DID)
- Propensity Score Matching (PSM)
- 简单工具变量 (IV/2SLS)
- 断点回归 (RDD) — 概念演示
- 因果图 / 混杂因子检测辅助

Author: AI-Native Strategy Analytics
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression, LinearRegression


# ---------------------------------------------------------------------------
# Difference-in-Differences
# ---------------------------------------------------------------------------

@dataclass
class DIDResult:
    """DID 估计结果"""
    att: float                     # Average Treatment effect on the Treated
    pre_treat_diff: float          # 处理前的组间差异
    post_treat_diff: float         # 处理后的组间差异
    did_estimate: float            # DID 估计量
    std_error: float
    p_value: float
    ci_lower: float
    ci_upper: float

    def summary(self) -> str:
        return (
            f"\n{'='*50}\n"
            f"  Difference-in-Differences Estimate\n"
            f"{'='*50}\n"
            f"  Pre-treatment  diff: {self.pre_treat_diff:.4f}\n"
            f"  Post-treatment diff: {self.post_treat_diff:.4f}\n"
            f"  DiD Estimate (ATT):  {self.did_estimate:.4f}\n"
            f"  Std Error:           {self.std_error:.4f}\n"
            f"  95% CI:              [{self.ci_lower:.4f}, {self.ci_upper:.4f}]\n"
            f"  p-value:             {self.p_value:.4f}\n"
            f"{'='*50}\n"
        )


def difference_in_differences(
    df: pd.DataFrame,
    time_col: str,
    group_col: str,
    outcome_col: str,
    treatment_time,
) -> DIDResult:
    """
    经典 2x2 Difference-in-Differences。

    Parameters
    ----------
    df : pd.DataFrame
        面板数据，需包含时间列、分组列、结果列
    time_col : str
        时间列名 (0=处理前, 1=处理后)
    group_col : str
        分组列名 (0=对照组, 1=处理组)
    outcome_col : str
        结果列名 (连续型)
    treatment_time :
        处理发生的时点

    Returns
    -------
    DIDResult
    """
    pre = df[df[time_col] < treatment_time] if not isinstance(treatment_time, (int, float)) else df[df[time_col] == 0]
    post = df[df[time_col] >= treatment_time] if not isinstance(treatment_time, (int, float)) else df[df[time_col] == 1]

    # Pre-period
    pre_control = pre[pre[group_col] == 0][outcome_col].mean()
    pre_treated = pre[pre[group_col] == 1][outcome_col].mean()

    # Post-period
    post_control = post[post[group_col] == 0][outcome_col].mean()
    post_treated = post[post[group_col] == 1][outcome_col].mean()

    # DiD = (T_post - T_pre) - (C_post - C_pre)
    did = (post_treated - pre_treated) - (post_control - pre_control)

    # Standard error via clustered bootstrap (simplified)
    n_bootstrap = 500
    estimates = []
    rng = np.random.default_rng(42)
    unique_ids = df.index.unique() if hasattr(df.index, 'unique') else np.arange(len(df))
    for _ in range(n_bootstrap):
        bs_idx = rng.choice(unique_ids, size=len(unique_ids), replace=True)
        bs_df = df.iloc[bs_idx] if isinstance(df.index, pd.RangeIndex) else df.loc[bs_idx]
        try:
            bs_did = _compute_did(bs_df, time_col, group_col, outcome_col, treatment_time)
            estimates.append(bs_did)
        except Exception:
            continue

    se = np.std(estimates, ddof=1) if len(estimates) > 1 else 0.1
    z_stat = did / se if se > 0 else 0
    p_value = 2 * (1 - sp_stats.norm.cdf(abs(z_stat)))
    ci_lower = did - 1.96 * se
    ci_upper = did + 1.96 * se

    return DIDResult(
        att=did,
        pre_treat_diff=pre_treated - pre_control,
        post_treat_diff=post_treated - post_control,
        did_estimate=did,
        std_error=se,
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


def _compute_did(df, time_col, group_col, outcome_col, treatment_time):
    """Helper: compute DID on a single bootstrap sample."""
    pre = df[df[time_col] == 0] if isinstance(treatment_time, (int, float)) and treatment_time == 1 else df[df[time_col] < treatment_time]
    post = df[df[time_col] == 1] if isinstance(treatment_time, (int, float)) and treatment_time == 1 else df[df[time_col] >= treatment_time]

    pre_c = pre[pre[group_col] == 0][outcome_col].mean()
    pre_t = pre[pre[group_col] == 1][outcome_col].mean()
    post_c = post[post[group_col] == 0][outcome_col].mean()
    post_t = post[post[group_col] == 1][outcome_col].mean()

    return (post_t - pre_t) - (post_c - pre_c)


# ---------------------------------------------------------------------------
# Propensity Score Matching (PSM)
# ---------------------------------------------------------------------------

def propensity_score_matching(
    df: pd.DataFrame,
    treatment_col: str,
    covariates: list,
    outcome_col: str,
    caliper: float = 0.2,
    k: int = 1,
) -> Dict:
    """
    基于倾向性得分的最邻近匹配 (1:k Nearest Neighbor Matching)。

    Parameters
    ----------
    df : pd.DataFrame
    treatment_col : str
        处理指示变量 (0/1)
    covariates : list
        协变量列名列表
    outcome_col : str
        结果变量
    caliper : float
        卡钳宽度（倾向性得分标准差的倍数）
    k : int
        每个处理单元的匹配数

    Returns
    -------
    dict with ATT, ATE, balance stats
    """
    # Step 1: Estimate propensity scores
    X = df[covariates].values
    y_treat = df[treatment_col].values

    # Standardize
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)

    ps_model = LogisticRegression(max_iter=1000, random_state=42)
    ps_model.fit(X_scaled, y_treat)
    ps_scores = ps_model.predict_proba(X_scaled)[:, 1]

    df_ps = df.copy()
    df_ps["ps_score"] = ps_scores

    treated = df_ps[df_ps[treatment_col] == 1].copy()
    control = df_ps[df_ps[treatment_col] == 0].copy()

    ps_std = np.std(ps_scores)
    caliper_width = caliper * ps_std

    # Step 2: Nearest neighbor matching
    matched_outcomes_treated = []
    matched_outcomes_control = []
    control_used = set()

    for idx_t, row_t in treated.iterrows():
        # Compute distances to all controls
        distances = np.abs(control["ps_score"].values - row_t["ps_score"])
        eligible = [
            (i, distances[i])
            for i in range(len(control))
            if distances[i] <= caliper_width and control.index[i] not in control_used
        ]

        if len(eligible) < k:
            continue

        eligible.sort(key=lambda x: x[1])
        for j in range(k):
            matched_idx = eligible[j][0]
            control_used.add(matched_idx)
            matched_outcomes_treated.append(row_t[outcome_col])
            matched_outcomes_control.append(control.iloc[eligible[j][0]][outcome_col])

    if len(matched_outcomes_treated) == 0:
        return {"error": "No matches found within caliper. Try a wider caliper.", "att": None}

    matched_t = np.array(matched_outcomes_treated)
    matched_c = np.array(matched_outcomes_control)
    att = np.mean(matched_t - matched_c)
    se = np.std(matched_t - matched_c, ddof=1) / np.sqrt(len(matched_t))
    t_stat = att / se if se > 0 else 0
    p_value = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df=len(matched_t) - 1))

    # Balance check: standardized mean difference before and after matching
    balance_before = _smd(X_scaled, y_treat, covariates)
    # After matching (approximate)
    matched_indices = list(control_used)
    balance_after = _smd(
        X_scaled[np.concatenate([treated.index.values[:len(matched_indices)], matched_indices])],
        np.concatenate([np.ones(len(matched_indices)), np.zeros(len(matched_indices))]),
        covariates,
    )

    return {
        "att": att,
        "std_error": se,
        "p_value": p_value,
        "n_matched": len(matched_t),
        "balance_before": balance_before,
        "balance_after": balance_after,
        "method": f"1:{k} Nearest Neighbor PSM (caliper={caliper})",
    }


def _smd(X, treatment, cov_names):
    """Compute standardized mean differences for balance check."""
    smds = {}
    for j, name in enumerate(cov_names):
        mean_t = np.mean(X[treatment == 1, j])
        mean_c = np.mean(X[treatment == 0, j])
        std_pooled = np.sqrt(
            (np.var(X[treatment == 1, j]) + np.var(X[treatment == 0, j])) / 2
        )
        smds[name] = abs(mean_t - mean_c) / std_pooled if std_pooled > 0 else 0
    return smds


# ---------------------------------------------------------------------------
# Instrumental Variables (2SLS)
# ---------------------------------------------------------------------------

@dataclass
class IVResult:
    """2SLS 工具变量估计结果"""
    iv_estimate: float
    ols_estimate: float       # 普通 OLS 估计（有偏，作对比）
    first_stage_f_stat: float  # 第一阶段 F 统计量 (>10 为强工具变量)
    std_error: float
    ci_lower: float
    ci_upper: float

    def summary(self) -> str:
        strength = "强 ✅" if self.first_stage_f_stat > 10 else "弱 ⚠️ (F<10)"
        return (
            f"\n{'='*50}\n"
            f"  2SLS Instrumental Variable Estimate\n"
            f"{'='*50}\n"
            f"  IV Estimate:   {self.iv_estimate:.4f}\n"
            f"  OLS Estimate:  {self.ols_estimate:.4f}  (likely biased)\n"
            f"  First-stage F: {self.first_stage_f_stat:.1f}  [{strength}]\n"
            f"  95% CI:        [{self.ci_lower:.4f}, {self.ci_upper:.4f}]\n"
            f"{'='*50}\n"
        )


def iv_2sls(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    instrument_col: str,
    controls: Optional[list] = None,
) -> IVResult:
    """
    两阶段最小二乘 (2SLS) 工具变量估计。

    使用场景：当处理变量存在内生性时，利用工具变量估计因果效应。
    例如：用"是否收到推送提醒"作为工具变量，估计"实际使用新功能"对留存的影响。

    Parameters
    ----------
    df : pd.DataFrame
    outcome_col : str — 结果变量 Y
    treatment_col : str — 内生处理变量 D
    instrument_col : str — 工具变量 Z (满足相关性和外生性)
    controls : list — 外生控制变量
    """
    from statsmodels.api import OLS, add_constant

    y = df[outcome_col].values
    D = df[treatment_col].values
    Z = df[instrument_col].values

    if controls:
        X1 = np.column_stack([Z] + [df[c].values for c in controls])
    else:
        X1 = Z.reshape(-1, 1)

    # First stage: D ~ Z + controls
    X1 = add_constant(X1)
    first_stage = OLS(D, X1).fit()
    D_hat = first_stage.fittedvalues
    f_stat = first_stage.fvalue

    # Second stage: Y ~ D_hat + controls
    if controls:
        X2 = np.column_stack([D_hat] + [df[c].values for c in controls])
    else:
        X2 = D_hat.reshape(-1, 1)
    X2 = add_constant(X2)
    second_stage = OLS(y, X2).fit()

    iv_est = second_stage.params[1]
    se = second_stage.bse[1]
    ci_lower = iv_est - 1.96 * se
    ci_upper = iv_est + 1.96 * se

    # Naive OLS for comparison
    if controls:
        X_ols = np.column_stack([D] + [df[c].values for c in controls])
    else:
        X_ols = D.reshape(-1, 1)
    X_ols = add_constant(X_ols)
    ols_est = OLS(y, X_ols).fit().params[1]

    return IVResult(
        iv_estimate=iv_est,
        ols_estimate=ols_est,
        first_stage_f_stat=f_stat,
        std_error=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


# ---------------------------------------------------------------------------
# Regression Discontinuity Design (RDD) — 概念演示
# ---------------------------------------------------------------------------

def sharp_rdd_local_linear(
    df: pd.DataFrame,
    running_var: str,
    outcome_col: str,
    cutoff: float,
    bandwidth: Optional[float] = None,
) -> Dict:
    """
    断点回归设计的局部线性估计 (Sharp RDD)。

    适用于有明确断点的策略评估。
    例如：信用分 ≥ 600 自动通过贷款 → 在 cutoff 附近比较违约率。

    Parameters
    ----------
    df : pd.DataFrame
    running_var : str — 驱动变量 (running variable)
    outcome_col : str — 结果变量
    cutoff : float — 断点值
    bandwidth : float — 带宽，None 则用 Imbens-Kalyanaraman 最优带宽近似
    """
    if bandwidth is None:
        # Rough IK bandwidth approximation
        n = len(df)
        bandwidth = 1.84 * np.std(df[running_var]) * (n ** (-1 / 5))

    # Subset to within bandwidth
    mask = (df[running_var] >= cutoff - bandwidth) & (df[running_var] <= cutoff + bandwidth)
    local = df[mask].copy()

    # Normalize running variable to be centered at cutoff
    local["X_centered"] = local[running_var] - cutoff
    local["above"] = (local[running_var] >= cutoff).astype(int)
    local["X_above"] = local["X_centered"] * local["above"]

    # Local linear regression: Y ~ X + above + X*above
    X = local[["X_centered", "above", "X_above"]].values
    from statsmodels.api import OLS, add_constant
    model = OLS(local[outcome_col].values, add_constant(X)).fit()

    treatment_effect = model.params[2]  # coefficient on "above" = jump at cutoff
    se = model.bse[2]
    p_value = model.pvalues[2]

    return {
        "rdd_estimate": treatment_effect,
        "std_error": se,
        "p_value": p_value,
        "bandwidth": bandwidth,
        "n_local": len(local),
        "method": "Sharp RDD (Local Linear)",
    }


# ---------------------------------------------------------------------------
# Utility: generate causal demo data
# ---------------------------------------------------------------------------

def generate_did_data(
    n_units: int = 200,
    n_periods: int = 2,
    treatment_effect: float = 3.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    生成 DID 演示数据。
    50% 处理组，50% 对照组；处理只发生在第 2 期。
    """
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n_units):
        treated = 1 if i < n_units // 2 else 0
        unit_fixed = rng.normal(0, 2)
        for t in range(n_periods):
            y = 10 + unit_fixed + 0.5 * t + rng.normal(0, 1)
            if treated and t == 1:
                y += treatment_effect  # treatment effect
            records.append({"unit": i, "time": t, "treated": treated, "outcome": y})
    return pd.DataFrame(records)
