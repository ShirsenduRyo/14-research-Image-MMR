from pathlib import Path

import pandas as pd

from image_rerank.plots import plot_lambda_sweep, plot_metric_bars, plot_tradeoff


def test_plot_helpers_support_non_default_k_and_save_files(tmp_path: Path):
    metrics = pd.DataFrame({
        "query_id": ["q1", "q2", "q1", "q2"],
        "method": ["baseline", "baseline", "mmr", "mmr"],
        "ap@5": [0.70, 0.75, 0.68, 0.73],
        "ndcg@3": [0.80, 0.82, 0.78, 0.81],
        "recall@3": [0.30, 0.35, 0.28, 0.33],
        "ild@3": [0.25, 0.28, 0.38, 0.40],
        "rerank_latency_ms": [0.01, 0.01, 0.20, 0.22],
    })
    summary = metrics.groupby("method", as_index=False).agg(
        **{
            "ap@5_mean": ("ap@5", "mean"), "ap@5_std": ("ap@5", "std"),
            "ndcg@3_mean": ("ndcg@3", "mean"), "ndcg@3_std": ("ndcg@3", "std"),
            "recall@3_mean": ("recall@3", "mean"), "recall@3_std": ("recall@3", "std"),
            "ild@3_mean": ("ild@3", "mean"), "ild@3_std": ("ild@3", "std"),
        }
    )
    sweep = pd.DataFrame({
        "lambda_relevance": [0.3, 0.7, 0.3, 0.7],
        "ap@5": [0.65, 0.75, 0.63, 0.73],
        "ild@3": [0.42, 0.28, 0.40, 0.30],
        "rerank_latency_ms": [0.2, 0.2, 0.21, 0.21],
    })

    tradeoff_path = plot_tradeoff(metrics, tmp_path / "tradeoff.png")
    bars_path = plot_metric_bars(summary, tmp_path / "bars.png")
    sweep_path = plot_lambda_sweep(sweep, tmp_path / "lambda.png")
    assert tradeoff_path.exists() and tradeoff_path.stat().st_size > 0
    assert bars_path.exists() and bars_path.stat().st_size > 0
    assert sweep_path.exists() and sweep_path.stat().st_size > 0
