"""Paired uncertainty estimates for comparing reranking methods across queries."""

from __future__ import annotations

import numpy as np
import pandas as pd


def paired_bootstrap_difference(
    metrics: pd.DataFrame,
    *,
    metric: str,
    treatment: str,
    control: str,
    query_col: str = "query_id",
    method_col: str = "method",
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int | str]:
    """Bootstrap the paired mean difference (treatment minus control) over queries.

    Pairing by query reduces noise because every method is evaluated on the same
    fixed candidate shortlist. NaN pairs are dropped before resampling.
    """
    if n_bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    required = {query_col, method_col, metric}
    if not required.issubset(metrics.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(metrics.columns))}")
    subset = metrics.loc[metrics[method_col].isin([treatment, control]), [query_col, method_col, metric]]
    pivot = subset.pivot_table(index=query_col, columns=method_col, values=metric, aggfunc="mean")
    if treatment not in pivot or control not in pivot:
        raise ValueError("Both treatment and control methods must be present")
    paired = pivot[[treatment, control]].dropna()
    if paired.empty:
        raise ValueError("No paired query observations available")
    differences = (paired[treatment] - paired[control]).to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(differences), size=(n_bootstrap, len(differences)))
    bootstrap_means = differences[sample_indices].mean(axis=1)
    alpha = 1.0 - confidence
    lower, upper = np.quantile(bootstrap_means, [alpha / 2, 1.0 - alpha / 2])
    return {
        "metric": metric,
        "treatment": treatment,
        "control": control,
        "n_queries": int(len(differences)),
        "mean_difference": float(differences.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": float(confidence),
        "n_bootstrap": int(n_bootstrap),
    }
