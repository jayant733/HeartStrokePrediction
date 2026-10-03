"""
Statistical Hypothesis Testing & Audit Logging Engine
=====================================================
Provides rigorous statistical testing for model validation, unsupervised cluster
phenotyping, and clinical cohort separation:
  - Z-Test: One-sample & Two-sample tests for proportions and means
  - P-Test / P-Value: Formal hypothesis testing decision engine (H0 vs H1)
  - F-Test: One-way ANOVA across unsupervised clusters and Variance Ratio tests
  - Log Transformation: Variance stabilization & skewness reduction for biomarkers
  - Statistical Audit Log: Persistent, audit-trailed logging of all hypothesis tests
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

# Path configurations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE_PATH = os.path.join(BASE_DIR, "data", "statistical_test_audit.log")

# Setup Python file logger for statistical tests
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
_file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
stat_logger = logging.getLogger("statistical_tests")
stat_logger.setLevel(logging.INFO)
if not stat_logger.handlers:
    stat_logger.addHandler(_file_handler)

# In-memory log cache for fast UI retrieval
_IN_MEMORY_LOGS: List[Dict[str, Any]] = []


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOGGING & AUDIT TRAIL SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

def log_statistical_test(
    test_type: str,
    target_metric: str,
    statistic_name: str,
    statistic_value: float,
    p_value: float,
    decision: str,
    null_hypothesis: str,
    alternative_hypothesis: str,
    alpha: float = 0.05,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Log an executed statistical hypothesis test into persistent log and in-memory cache.
    """
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_type": test_type,
        "target_metric": target_metric,
        "statistic_name": statistic_name,
        "statistic_value": round(float(statistic_value), 4),
        "p_value": float(f"{p_value:.6e}"),
        "alpha": alpha,
        "null_hypothesis": null_hypothesis,
        "alternative_hypothesis": alternative_hypothesis,
        "decision": decision,
        "is_significant": bool(p_value < alpha),
        "details": details or {},
    }
    
    # Write formatted string to disk log
    stat_logger.info(
        f"TEST={test_type} | METRIC={target_metric} | {statistic_name}={entry['statistic_value']} | "
        f"P_VALUE={entry['p_value']} | DECISION={decision} | H0={null_hypothesis}"
    )
    
    _IN_MEMORY_LOGS.insert(0, entry)
    # Keep last 100 entries in memory
    if len(_IN_MEMORY_LOGS) > 100:
        _IN_MEMORY_LOGS.pop()
        
    return entry


def get_statistical_audit_logs() -> List[Dict[str, Any]]:
    """Retrieve in-memory and persistent statistical audit logs."""
    if not _IN_MEMORY_LOGS and os.path.exists(LOG_FILE_PATH):
        # Fallback load from disk if fresh process
        try:
            with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()[-50:]
                for line in reversed(lines):
                    _IN_MEMORY_LOGS.append({
                        "timestamp": line[:19].strip("[]"),
                        "raw_entry": line.strip()
                    })
        except Exception:
            pass
    return _IN_MEMORY_LOGS


# ─────────────────────────────────────────────────────────────────────────────
# 2. P-VALUE EVALUATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_p_value(p_val: float, alpha: float = 0.05) -> Tuple[str, str]:
    """
    Evaluate p-value against alpha threshold and return decision and clinical interpretation.
    """
    if p_val < 0.0001:
        strength = "Extremely strong statistical evidence (p < 0.0001)"
    elif p_val < 0.001:
        strength = "Very strong statistical evidence (p < 0.001)"
    elif p_val < 0.01:
        strength = "Strong statistical evidence (p < 0.01)"
    elif p_val < alpha:
        strength = f"Statistically significant evidence (p < {alpha})"
    else:
        strength = f"Insufficient statistical evidence (p >= {alpha})"

    if p_val < alpha:
        decision = f"Reject Null Hypothesis H0 (Statistically Significant at alpha={alpha})"
    else:
        decision = f"Fail to Reject Null Hypothesis H0 (Not Statistically Significant at alpha={alpha})"

    return decision, strength


# ─────────────────────────────────────────────────────────────────────────────
# 3. Z-TEST (TWO-SAMPLE PROPORTION & ONE-SAMPLE MEAN)
# ─────────────────────────────────────────────────────────────────────────────

def run_z_test_proportions(
    count1: int,
    n1: int,
    count2: int,
    n2: int,
    label1: str = "Cohort 1",
    label2: str = "Cohort 2 / Baseline",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Two-Sample Z-Test for Difference in Proportions:
    Tests whether the proportion in Cohort 1 is statistically different from Cohort 2/Baseline.
    Formula:
        p_pool = (count1 + count2) / (n1 + n2)
        SE = sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        z = (p1 - p2) / SE
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError("Sample sizes must be strictly positive.")

    p1 = count1 / n1
    p2 = count2 / n2
    p_pool = (count1 + count2) / (n1 + n2)
    
    # Avoid division by zero if all 0 or all 1
    if p_pool == 0 or p_pool == 1:
        se = 1e-9
    else:
        se = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))

    z_stat = (p1 - p2) / se
    p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))

    # 95% Confidence Interval for difference
    se_diff = np.sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2))
    z_crit = stats.norm.ppf(1 - alpha / 2.0)
    diff = p1 - p2
    ci_lower = diff - z_crit * se_diff
    ci_upper = diff + z_crit * se_diff

    h0 = f"Stroke rate in {label1} ({p1:.1%}) equals stroke rate in {label2} ({p2:.1%})"
    ha = f"Stroke rate in {label1} ({p1:.1%}) differs significantly from {label2} ({p2:.1%})"
    decision, strength = evaluate_p_value(p_val, alpha)

    res = {
        "test_name": "Two-Sample Z-Test for Proportions",
        "cohort_1": label1,
        "cohort_2": label2,
        "n1": n1,
        "count1": count1,
        "prop1": round(p1, 4),
        "n2": n2,
        "count2": count2,
        "prop2": round(p2, 4),
        "difference": round(diff, 4),
        "z_statistic": round(z_stat, 4),
        "p_value": p_val,
        "alpha": alpha,
        "ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
        "decision": decision,
        "interpretation": strength,
        "null_hypothesis": h0,
        "alternative_hypothesis": ha,
    }

    log_statistical_test(
        test_type="Z-Test (Proportions)",
        target_metric="Stroke Incidence Rate",
        statistic_name="Z-Score",
        statistic_value=z_stat,
        p_value=p_val,
        decision=decision,
        null_hypothesis=h0,
        alternative_hypothesis=ha,
        alpha=alpha,
        details={"prop1": p1, "prop2": p2, "diff": diff},
    )

    return res


def run_z_test_means(
    sample_mean: float,
    sample_std: float,
    n: int,
    mu0: float,
    metric_name: str = "Biomarker",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    One-Sample Z-Test for Mean against Reference Benchmark Value:
    Formula:
        z = (sample_mean - mu0) / (sample_std / sqrt(n))
    """
    if n <= 1:
        raise ValueError("Sample size must be > 1.")

    se = sample_std / np.sqrt(n)
    if se == 0:
        se = 1e-9
    z_stat = (sample_mean - mu0) / se
    p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))

    z_crit = stats.norm.ppf(1 - alpha / 2.0)
    ci_lower = sample_mean - z_crit * se
    ci_upper = sample_mean + z_crit * se

    h0 = f"Mean {metric_name} ({sample_mean:.2f}) equals reference baseline ({mu0:.2f})"
    ha = f"Mean {metric_name} ({sample_mean:.2f}) differs significantly from reference ({mu0:.2f})"
    decision, strength = evaluate_p_value(p_val, alpha)

    res = {
        "test_name": "One-Sample Z-Test for Mean",
        "metric_name": metric_name,
        "sample_mean": round(sample_mean, 2),
        "reference_mu0": round(mu0, 2),
        "sample_std": round(sample_std, 2),
        "n": n,
        "z_statistic": round(z_stat, 4),
        "p_value": p_val,
        "alpha": alpha,
        "ci_95": [round(ci_lower, 2), round(ci_upper, 2)],
        "decision": decision,
        "interpretation": strength,
        "null_hypothesis": h0,
        "alternative_hypothesis": ha,
    }

    log_statistical_test(
        test_type="Z-Test (Means)",
        target_metric=metric_name,
        statistic_name="Z-Score",
        statistic_value=z_stat,
        p_value=p_val,
        decision=decision,
        null_hypothesis=h0,
        alternative_hypothesis=ha,
        alpha=alpha,
        details={"sample_mean": sample_mean, "mu0": mu0},
    )

    return res


# ─────────────────────────────────────────────────────────────────────────────
# 4. LOG TRANSFORMATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def apply_log_transform(
    data: Union[pd.Series, np.ndarray, List[float]],
    offset: float = 1.0,
) -> Tuple[np.ndarray, float, float]:
    """
    Apply natural logarithmic transformation to reduce positive skewness
    and stabilize variance for clinical biomarkers (e.g. glucose, BMI).
    Returns:
        (transformed_array, skewness_before, skewness_after)
    """
    arr = np.array(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    # Ensure positive
    min_val = np.min(arr)
    if min_val <= 0:
        shift = abs(min_val) + offset
        transformed = np.log(arr + shift)
    else:
        transformed = np.log(arr)

    skew_before = float(stats.skew(arr))
    skew_after = float(stats.skew(transformed))

    return transformed, round(skew_before, 3), round(skew_after, 3)


# ─────────────────────────────────────────────────────────────────────────────
# 5. F-TEST (ANOVA ACROSS CLUSTERS & VARIANCE RATIO)
# ─────────────────────────────────────────────────────────────────────────────

def run_f_test_anova(
    groups_dict: Dict[str, Union[pd.Series, np.ndarray, List[float]]],
    feature_name: str = "Clinical Biomarker",
    log_transform: bool = False,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    One-Way ANOVA F-Test across Unsupervised Clinical Clusters:
    Tests whether the population mean of a clinical biomarker differs
    significantly across the unsupervised clusters/phenotypes.
    
    Formula:
        F = MS_between / MS_within
        df_between = k - 1
        df_within = N - k
    """
    cleaned_groups = []
    labels = []
    skews_before = []
    skews_after = []

    for name, grp in groups_dict.items():
        arr = np.array(grp, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) > 1:
            if log_transform:
                trans, s_b, s_a = apply_log_transform(arr)
                cleaned_groups.append(trans)
                skews_before.append(s_b)
                skews_after.append(s_a)
            else:
                cleaned_groups.append(arr)
                skews_before.append(float(stats.skew(arr)))
            labels.append(name)

    if len(cleaned_groups) < 2:
        raise ValueError("ANOVA requires at least 2 distinct cohorts.")

    k = len(cleaned_groups)
    total_n = sum(len(g) for g in cleaned_groups)
    df_between = k - 1
    df_within = total_n - k

    f_stat, p_val = stats.f_oneway(*cleaned_groups)

    h0 = f"Mean {feature_name} is identical across all {k} unsupervised patient cohorts"
    ha = f"At least one patient cohort has a significantly different mean {feature_name}"
    decision, strength = evaluate_p_value(p_val, alpha)

    res = {
        "test_name": "One-Way ANOVA F-Test",
        "feature_name": feature_name,
        "k_groups": k,
        "total_samples": total_n,
        "df_between": df_between,
        "df_within": df_within,
        "f_statistic": round(float(f_stat), 4),
        "p_value": float(p_val),
        "alpha": alpha,
        "log_transformed": log_transform,
        "mean_skew_before": round(float(np.mean(skews_before)), 3) if skews_before else 0.0,
        "mean_skew_after": round(float(np.mean(skews_after)), 3) if skews_after else None,
        "decision": decision,
        "interpretation": strength,
        "null_hypothesis": h0,
        "alternative_hypothesis": ha,
    }

    log_statistical_test(
        test_type=f"ANOVA F-Test{' (Log-Transformed)' if log_transform else ''}",
        target_metric=feature_name,
        statistic_name="F-Statistic",
        statistic_value=float(f_stat),
        p_value=float(p_val),
        decision=decision,
        null_hypothesis=h0,
        alternative_hypothesis=ha,
        alpha=alpha,
        details={
            "k_groups": k,
            "df_between": df_between,
            "df_within": df_within,
            "log_transformed": log_transform,
        },
    )

    return res


def run_f_test_variance(
    sample1: Union[pd.Series, np.ndarray, List[float]],
    sample2: Union[pd.Series, np.ndarray, List[float]],
    label1: str = "Cohort 1",
    label2: str = "Cohort 2",
    metric_name: str = "Biomarker Variance",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Two-Sample Fisher's F-Test for Equality of Variances:
    Formula:
        F = s1^2 / s2^2 (with s1^2 >= s2^2)
    """
    arr1 = np.array(sample1, dtype=float)
    arr2 = np.array(sample2, dtype=float)
    arr1 = arr1[~np.isnan(arr1)]
    arr2 = arr2[~np.isnan(arr2)]

    n1, n2 = len(arr1), len(arr2)
    if n1 <= 1 or n2 <= 1:
        raise ValueError("Sample sizes must be > 1.")

    var1 = float(np.var(arr1, ddof=1))
    var2 = float(np.var(arr2, ddof=1))

    # Place larger variance in numerator
    if var1 >= var2:
        f_stat = var1 / (var2 if var2 > 0 else 1e-9)
        df1, df2 = n1 - 1, n2 - 1
    else:
        f_stat = var2 / (var1 if var1 > 0 else 1e-9)
        df1, df2 = n2 - 1, n1 - 1

    p_val = 2.0 * min(stats.f.cdf(f_stat, df1, df2), 1.0 - stats.f.cdf(f_stat, df1, df2))

    h0 = f"Variance of {metric_name} in {label1} equals variance in {label2}"
    ha = f"Variance of {metric_name} in {label1} differs significantly from {label2}"
    decision, strength = evaluate_p_value(p_val, alpha)

    res = {
        "test_name": "Two-Sample F-Test for Equality of Variances",
        "metric_name": metric_name,
        "label1": label1,
        "var1": round(var1, 4),
        "label2": label2,
        "var2": round(var2, 4),
        "f_statistic": round(f_stat, 4),
        "df1": df1,
        "df2": df2,
        "p_value": p_val,
        "alpha": alpha,
        "decision": decision,
        "interpretation": strength,
        "null_hypothesis": h0,
        "alternative_hypothesis": ha,
    }

    log_statistical_test(
        test_type="F-Test (Variance Ratio)",
        target_metric=metric_name,
        statistic_name="F-Ratio",
        statistic_value=f_stat,
        p_value=p_val,
        decision=decision,
        null_hypothesis=h0,
        alternative_hypothesis=ha,
        alpha=alpha,
        details={"var1": var1, "var2": var2, "df1": df1, "df2": df2},
    )

    return res


# ─────────────────────────────────────────────────────────────────────────────
# 6. COMPREHENSIVE CLINICAL CLUSTER HYPOTHESIS TEST BATTERY
# ─────────────────────────────────────────────────────────────────────────────

def run_cluster_statistical_battery() -> Dict[str, Any]:
    """
    Run a complete battery of Z-tests and ANOVA F-tests on the training dataset
    to statistically validate unsupervised patient clustering & risk disparities.
    """
    csv_path = os.path.join(BASE_DIR, "data", "Training_data", "healthcare-dataset-stroke-data.csv")
    if not os.path.exists(csv_path):
        return {"error": "Dataset not found"}

    df = pd.read_csv(csv_path)
    df = df[df["gender"] != "Other"].copy()

    # Get unsupervised cluster labels
    from stroke_prediction.cluster_engine import get_clustering_model
    from stroke_prediction.data_processing import pipeline
    km = get_clustering_model()
    proc = pipeline(df.drop("stroke", axis=1))
    df["cluster"] = km.predict(proc)

    # 1. Z-Tests: Proportion of strokes in each cluster vs general baseline
    overall_stroke_count = int(df["stroke"].sum())
    overall_n = len(df)
    baseline_rate = overall_stroke_count / overall_n

    z_test_results = []
    for c_id in sorted(df["cluster"].unique()):
        cdf = df[df["cluster"] == c_id]
        c_count = int(cdf["stroke"].sum())
        c_n = len(cdf)
        # Compare cohort vs rest of dataset
        rest_df = df[df["cluster"] != c_id]
        rest_count = int(rest_df["stroke"].sum())
        rest_n = len(rest_df)

        from stroke_prediction.cluster_engine import CLUSTER_PROFILES
        c_name = CLUSTER_PROFILES.get(c_id, {}).get("name", f"Cohort {c_id}")

        z_res = run_z_test_proportions(
            count1=c_count,
            n1=c_n,
            count2=rest_count,
            n2=rest_n,
            label1=f"Cohort {c_id} ({c_name})",
            label2="Other Patient Cohorts",
        )
        z_test_results.append(z_res)

    # 2. ANOVA F-Tests: Differences across all 4 clusters
    groups_glucose = {f"Cohort {c}": df[df["cluster"] == c]["avg_glucose_level"] for c in sorted(df["cluster"].unique())}
    groups_age = {f"Cohort {c}": df[df["cluster"] == c]["age"] for c in sorted(df["cluster"].unique())}
    groups_bmi = {f"Cohort {c}": df[df["cluster"] == c]["bmi"].dropna() for c in sorted(df["cluster"].unique())}

    # Raw ANOVA F-Tests
    f_glucose = run_f_test_anova(groups_glucose, feature_name="Average Glucose Level", log_transform=False)
    f_glucose_log = run_f_test_anova(groups_glucose, feature_name="Log(Glucose Level)", log_transform=True)
    f_age = run_f_test_anova(groups_age, feature_name="Patient Age", log_transform=False)
    f_bmi = run_f_test_anova(groups_bmi, feature_name="Body Mass Index (BMI)", log_transform=False)

    return {
        "dataset_size": overall_n,
        "baseline_stroke_rate": round(baseline_rate, 4),
        "cluster_proportions_z_tests": z_test_results,
        "anova_f_tests": {
            "glucose": f_glucose,
            "glucose_log_transformed": f_glucose_log,
            "age": f_age,
            "bmi": f_bmi,
        },
    }
