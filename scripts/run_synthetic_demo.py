"""Offline smoke test: run all rerankers and write plots/metrics without downloads."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd

from image_rerank.data import make_synthetic_queries
from image_rerank.metrics import evaluate_ranking
from image_rerank.rerankers import (
    adaptive_mmr_rerank, cluster_aspect_probabilities, greedy_dpp_rerank,
    mmr_rerank, original_rank, xquad_rerank,
)
from image_rerank.plots import plot_metric_bars, plot_tradeoff
from image_rerank.artifacts import ensure_dir, write_dataframe, write_json


def main() -> None:
    out = Path("artifacts/demo")
    ensure_dir(out / "metrics")
    ensure_dir(out / "rankings")
    ensure_dir(out / "figures")
    metrics_rows = []
    ranking_rows = []
    synthetic_data = make_synthetic_queries(n_queries=12, n_candidates=60, embedding_dim=64, seed=123)
    # Warm up algorithms once so measured latency does not include import/cold-start cost.
    warm = synthetic_data[0]
    warm_x = np.asarray(warm["image_embeddings"], dtype=np.float32)
    warm_q = np.asarray(warm["query_embedding"], dtype=np.float32)
    warm_scores = warm_x @ warm_q
    _ = original_rank(np.asarray(warm["original_scores"]), 50)
    _ = original_rank(warm_scores, 50)
    _ = mmr_rerank(warm_scores, warm_x, lambda_relevance=0.70, top_k=50)
    _ = adaptive_mmr_rerank(warm_scores, warm_x, top_k=50)
    _ = greedy_dpp_rerank(warm_scores, warm_x, top_k=50)
    _ = xquad_rerank(warm_scores, cluster_aspect_probabilities(warm_x, n_clusters=6, random_state=42), lambda_relevance=0.70, top_k=50)
    for data in synthetic_data:
        q_id = str(data["query_id"])
        q_text = str(data["query"])
        q = np.asarray(data["query_embedding"], dtype=np.float32)
        x = np.asarray(data["image_embeddings"], dtype=np.float32)
        base_scores = np.asarray(data["original_scores"], dtype=np.float32)
        labels = np.asarray(data["relevance"], dtype=np.int8)
        encoder_scores = x @ q
        embeddings = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        calls = {
            "initial_clip": lambda: original_rank(base_scores, 50),
            "encoder_similarity": lambda: original_rank(encoder_scores, 50),
            "mmr_fixed_0.70": lambda: mmr_rerank(encoder_scores, embeddings, lambda_relevance=0.70, top_k=50),
            "adaptive_mmr": lambda: adaptive_mmr_rerank(encoder_scores, embeddings, top_k=50)[0],
            "greedy_dpp": lambda: greedy_dpp_rerank(encoder_scores, embeddings, top_k=50),
            "cluster_xquad": lambda: xquad_rerank(
                encoder_scores,
                cluster_aspect_probabilities(embeddings, n_clusters=6, random_state=42),
                lambda_relevance=0.70,
                top_k=50,
            ),
        }
        for method, call in calls.items():
            start = time.perf_counter()
            order = call()
            latency = (time.perf_counter() - start) * 1000
            values = evaluate_ranking(labels, order, embeddings, k_relevance=50, k_diversity=10)
            metrics_rows.append({"query_id": q_id, "query": q_text, "method": method,
                                 "rerank_latency_ms": latency, **values, "data_type": "synthetic_smoke_test"})
            for rank, idx in enumerate(order[:50], 1):
                ranking_rows.append({"query_id": q_id, "method": method, "rank": rank,
                                     "image_id": str(data["image_ids"][idx]), "relevant": int(labels[idx])})

    metrics = pd.DataFrame(metrics_rows)
    rankings = pd.DataFrame(ranking_rows)
    summary = metrics.groupby("method", as_index=False).agg(
        **{"ap@50_mean": ("ap@50", "mean"), "ap@50_std": ("ap@50", "std"),
           "ndcg@10_mean": ("ndcg@10", "mean"), "ndcg@10_std": ("ndcg@10", "std"),
           "recall@10_mean": ("recall@10", "mean"), "recall@10_std": ("recall@10", "std"),
           "ild@10_mean": ("ild@10", "mean"), "ild@10_std": ("ild@10", "std"),
           "rerank_latency_ms_mean": ("rerank_latency_ms", "mean"),
           "rerank_latency_ms_p95": ("rerank_latency_ms", lambda x: float(np.percentile(x, 95)))})
    write_dataframe(out / "metrics" / "synthetic_per_query_metrics.csv", metrics)
    write_dataframe(out / "metrics" / "synthetic_summary.csv", summary)
    write_dataframe(out / "rankings" / "synthetic_rankings.csv", rankings)
    plot_tradeoff(metrics, out / "figures" / "synthetic_tradeoff.png")
    plot_metric_bars(summary, out / "figures" / "synthetic_metric_comparison.png")
    write_json(out / "metrics" / "synthetic_run_metadata.json", {
        "n_queries": 12, "n_candidates": 60, "seed": 123,
        "warning": "Synthetic smoke-test results only; do not interpret as benchmark performance.",
    })
    print(summary.to_string(index=False))
    print(f"Synthetic smoke-test artifacts: {out.resolve()}")


if __name__ == "__main__":
    main()
