"""
Sampling & Resampling Engine
============================
Comprehensive implementation of statistical sampling and machine learning
resampling techniques for clinical datasets and imbalanced classification:

1. Probability Sampling:
   - Simple Random Sampling (SRS)
   - Stratified Sampling (proportional class distribution preservation)
   - Systematic Sampling (k-interval selection)
   - Cluster Sampling (selecting entire primary sampling units)

2. Imbalanced Dataset Resampling (Critical for Stroke Prediction):
   - Oversampling: Random Oversampling, SMOTE, Borderline-SMOTE, ADASYN
   - Undersampling: Random Undersampling, Tomek Links, Edited Nearest Neighbors (ENN)
   - Hybrid: SMOTE + Tomek Links, SMOTE + ENN

3. Statistical Resampling for Validation:
   - Bootstrap Sampling (with replacement & out-of-bag calculation)
   - Stratified K-Fold Cross-Validation splits
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional, Union

# Optional imblearn imports with clean fallback
try:
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
    from imblearn.combine import SMOTETomek, SMOTEENN
    _HAS_IMBLEARN = True
except ImportError:
    _HAS_IMBLEARN = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. PROBABILITY SAMPLING METHODS
# ─────────────────────────────────────────────────────────────────────────────

def simple_random_sampling(
    df: pd.DataFrame,
    n: Optional[int] = None,
    frac: Optional[float] = None,
    replace: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Simple Random Sampling (SRS):
    Every individual in the population has an equal and independent probability
    of being selected into the sample.
    """
    if n is None and frac is None:
        frac = 0.2
    return df.sample(n=n, frac=frac, replace=replace, random_state=random_state)


def stratified_sampling(
    df: pd.DataFrame,
    strata_col: str,
    frac: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Stratified Random Sampling:
    Partitions the population into non-overlapping subgroups (strata) based on a
    characteristic (e.g. stroke = 0 vs 1, or hypertension status) and samples
    proportionately from each stratum to preserve subgroup representation.
    """
    if strata_col not in df.columns:
        raise ValueError(f"Column '{strata_col}' not found in dataframe.")
    
    return df.groupby(strata_col, group_keys=False).apply(
        lambda x: x.sample(frac=frac, random_state=random_state)
    )


def systematic_sampling(
    df: pd.DataFrame,
    step_k: Optional[int] = None,
    sample_size: Optional[int] = None,
    start_idx: int = 0,
) -> pd.DataFrame:
    """
    Systematic Sampling:
    Selects sample members at a regular periodic interval k = N / n from an ordered population list,
    starting from a randomly chosen initial offset.
    """
    n_pop = len(df)
    if step_k is None:
        if sample_size is None or sample_size <= 0:
            sample_size = int(n_pop * 0.2)
        step_k = max(1, n_pop // sample_size)
    
    indices = np.arange(start_idx % step_k, n_pop, step_k)
    return df.iloc[indices]


def cluster_sampling(
    df: pd.DataFrame,
    cluster_col: str,
    n_clusters_to_sample: int = 2,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Single-Stage Cluster Sampling:
    Divides the population into natural clusters (e.g. hospitals, work types, geographic regions).
    Randomly selects a subset of whole clusters and includes all individuals within selected clusters.
    """
    if cluster_col not in df.columns:
        raise ValueError(f"Column '{cluster_col}' not found in dataframe.")
    
    unique_clusters = df[cluster_col].unique()
    if n_clusters_to_sample > len(unique_clusters):
        n_clusters_to_sample = len(unique_clusters)
        
    np.random.seed(random_state)
    chosen_clusters = np.random.choice(unique_clusters, size=n_clusters_to_sample, replace=False)
    return df[df[cluster_col].isin(chosen_clusters)]


# ─────────────────────────────────────────────────────────────────────────────
# 2. IMBALANCED DATASET RESAMPLING (HEALTHCARE & STROKE CLASSIFICATION)
# ─────────────────────────────────────────────────────────────────────────────

def resample_imbalanced_data(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    method: str = "smote",
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Apply advanced resampling techniques for severe class imbalance.
    In stroke datasets, stroke prevalence is ~4.8% (minority class).
    
    Supported methods:
      - 'ros': Random Over-Sampling (duplicates minority instances)
      - 'smote': Synthetic Minority Over-sampling Technique (k-NN interpolation)
      - 'borderline_smote': Over-samples only ambiguous borderline minority samples
      - 'adasyn': Adaptive Synthetic Sampling (focuses on harder minority instances)
      - 'rus': Random Under-Sampling (discards majority instances)
      - 'tomek': Tomek Links removal (cleans boundary pairs)
      - 'enn': Edited Nearest Neighbors (removes noisy misclassified points)
      - 'smote_tomek': Hybrid SMOTE oversampling followed by Tomek Link undersampling
      - 'smote_enn': Hybrid SMOTE oversampling followed by ENN noise cleaning
    """
    if not _HAS_IMBLEARN:
        raise ImportError("imbalanced-learn package is required for resampling.")

    method = method.lower().strip()
    orig_dist = dict(pd.Series(y).value_counts())

    resamplers = {
        "ros": RandomOverSampler(random_state=random_state),
        "smote": SMOTE(random_state=random_state),
        "borderline_smote": BorderlineSMOTE(random_state=random_state),
        "adasyn": ADASYN(random_state=random_state),
        "rus": RandomUnderSampler(random_state=random_state),
        "tomek": TomekLinks(),
        "enn": EditedNearestNeighbours(),
        "smote_tomek": SMOTETomek(random_state=random_state),
        "smote_enn": SMOTEENN(random_state=random_state),
    }

    if method not in resamplers:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(resamplers.keys())}")

    sampler = resamplers[method]
    X_res, y_res = sampler.fit_resample(X, y)
    new_dist = dict(pd.Series(y_res).value_counts())

    info = {
        "method": method.upper(),
        "original_distribution": orig_dist,
        "resampled_distribution": new_dist,
        "original_samples": len(y),
        "resampled_samples": len(y_res),
        "minority_gain": new_dist.get(1, 0) - orig_dist.get(1, 0),
        "majority_reduction": orig_dist.get(0, 0) - new_dist.get(0, 0),
    }

    return X_res, y_res, info


# ─────────────────────────────────────────────────────────────────────────────
# 3. STATISTICAL RESAMPLING FOR MODEL VALIDATION (BOOTSTRAP & CROSS-VALIDATION)
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_sample(
    df: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Bootstrap Resampling (Bootstrapping / Bagging):
    Samples N instances with replacement from a population of size N.
    
    Statistical Property:
      Probability of an instance NOT being chosen = (1 - 1/N)^N ≈ 1/e ≈ 36.8% (Out-of-Bag / OOB).
      Probability of an instance being chosen at least once ≈ 63.2% (In-Bag sample).
    """
    n = len(df)
    in_bag = df.sample(n=n, replace=True, random_state=random_state)
    oob_indices = df.index.difference(in_bag.index)
    oob = df.loc[oob_indices]

    metrics = {
        "total_population_N": n,
        "in_bag_size": len(in_bag),
        "oob_size": len(oob),
        "oob_fraction": round(len(oob) / n, 4),
        "theoretical_oob_fraction": 0.3679,
    }

    return in_bag, oob, metrics
