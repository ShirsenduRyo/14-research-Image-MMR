import pandas as pd
import pytest

from image_rerank.statistics import paired_bootstrap_difference


def test_paired_bootstrap_zero_difference():
    df = pd.DataFrame({
        "query_id": ["q1", "q1", "q2", "q2", "q3", "q3"],
        "method": ["a", "b", "a", "b", "a", "b"],
        "metric": [0.8, 0.8, 0.4, 0.4, 0.6, 0.6],
    })
    result = paired_bootstrap_difference(
        df, metric="metric", treatment="a", control="b", n_bootstrap=500
    )
    assert result["mean_difference"] == pytest.approx(0.0)
    assert result["ci_lower"] == pytest.approx(0.0)
    assert result["ci_upper"] == pytest.approx(0.0)


def test_paired_bootstrap_detects_positive_difference():
    df = pd.DataFrame({
        "query_id": ["q1", "q1", "q2", "q2", "q3", "q3"],
        "method": ["a", "b", "a", "b", "a", "b"],
        "metric": [0.9, 0.8, 0.6, 0.5, 0.7, 0.6],
    })
    result = paired_bootstrap_difference(
        df, metric="metric", treatment="a", control="b", n_bootstrap=500
    )
    assert result["mean_difference"] == pytest.approx(0.1)
    assert result["ci_lower"] > 0.0
