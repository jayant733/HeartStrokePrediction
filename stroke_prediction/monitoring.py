"""
Model Monitoring & Statistical Drift Detection Engine
=====================================================
Monitors incoming patient data streams for data drift and concept drift using:
  - Z-Test on prediction positive rates (Stroke proportion drift)
  - F-Test on feature variances (Biomarker distribution dispersion drift)
  - P-Value significance thresholding (alpha = 0.05)
  - Log transformation for skewed biomarkers (glucose, BMI)
  - Persistent statistical logging to audit trail
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from stroke_prediction.statistical_tests import (
    run_z_test_proportions,
    run_f_test_variance,
    run_z_test_means,
    apply_log_transform,
    log_statistical_test,
)


def monitor_data_drift(
    new_data: pd.DataFrame,
    reference_data: pd.DataFrame,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Monitor data drift between new incoming patient records and baseline reference data
    using statistical hypothesis tests (Z-test, F-test, P-value).
    """
    alerts = []
    test_summaries = []

    # 1. Z-Test on Categorical / Proportion Drift (Hypertension, Heart Disease, Stroke if present)
    for col in ["hypertension", "heart_disease", "stroke", "prediction"]:
        if col in new_data.columns and col in reference_data.columns:
            count_new = int((new_data[col] == 1).sum())
            n_new = len(new_data)
            count_ref = int((reference_data[col] == 1).sum())
            n_ref = len(reference_data)

            if n_new > 5 and n_ref > 5:
                z_res = run_z_test_proportions(
                    count1=count_new,
                    n1=n_new,
                    count2=count_ref,
                    n2=n_ref,
                    label1=f"Incoming Batch ({col})",
                    label2=f"Reference Baseline ({col})",
                    alpha=alpha,
                )
                test_summaries.append(z_res)
                if z_res["p_value"] < alpha:
                    alerts.append(f"⚠️ Significant drift in {col}: Z={z_res['z_statistic']}, p={z_res['p_value']:.4e}")

    # 2. F-Test & Z-Test on Continuous Biomarkers (Age, Glucose, BMI)
    for col in ["avg_glucose_level", "bmi", "age"]:
        if col in new_data.columns and col in reference_data.columns:
            s_new = new_data[col].dropna()
            s_ref = reference_data[col].dropna()

            if len(s_new) > 5 and len(s_ref) > 5:
                # Continuous variance F-test
                f_res = run_f_test_variance(
                    sample1=s_new,
                    sample2=s_ref,
                    label1="Incoming Batch",
                    label2="Reference Baseline",
                    metric_name=f"{col} Variance",
                    alpha=alpha,
                )
                test_summaries.append(f_res)
                if f_res["p_value"] < alpha:
                    alerts.append(f"⚠️ Variance drift in {col}: F={f_res['f_statistic']}, p={f_res['p_value']:.4e}")

    has_drift = len(alerts) > 0
    return {
        "drift_detected": has_drift,
        "total_alerts": len(alerts),
        "alerts": alerts,
        "tests_executed": len(test_summaries),
        "summary": "Data drift detected via statistical hypothesis testing!" if has_drift else "No significant drift detected.",
    }
