"""Ranking metrics for relevance, semantic/visual diversity, and coverage."""

from __future__ import annotations

import numpy as np


def _ranked_relevance(relevance: np.ndarray, order: np.ndarray, k: int) -> np.ndarray:
    y = np.asarray(relevance, dtype=np.int8).reshape(-1)
    idx = np.asarray(order, dtype=np.int64).reshape(-1)
    if len(np.unique(idx)) != len(idx):
        raise ValueError("Ranking contains duplicate candidate indices")
    if np.any(idx < 0) or np.any(idx >= len(y)):
        raise ValueError("Ranking contains out-of-range candidate indices")
    if k <= 0:
        raise ValueError("k must be positive")
    return y[idx[:k]]


def average_precision_at_k(relevance: np.ndarray, order: np.ndarray, k: int = 50) -> float:
    """AP@k normalized by min(k, relevant candidates in the full shortlist)."""
    y = np.asarray(relevance, dtype=np.int8).reshape(-1)
    top = _ranked_relevance(y, order, k)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("relevance must be binary (0/1)")
    n_relevant = int(y.sum())
    if n_relevant == 0:
        return 0.0
    hit_positions = np.flatnonzero(top == 1)
    if len(hit_positions) == 0:
        return 0.0
    precisions = np.arange(1, len(top) + 1, dtype=np.float64)[hit_positions]
    precisions = precisions / (hit_positions + 1)
    denominator = min(k, n_relevant)
    return float(precisions.sum() / denominator)


def ndcg_at_k(relevance: np.ndarray, order: np.ndarray, k: int = 10) -> float:
    """Binary-relevance NDCG@k."""
    y = _ranked_relevance(relevance, order, k).astype(np.float64)
    discounts = 1.0 / np.log2(np.arange(2, len(y) + 2))
    dcg = float(np.sum(y * discounts))
    ideal_count = min(int(np.asarray(relevance).sum()), k)
    if ideal_count == 0:
        return 0.0
    ideal = float(np.sum(discounts[:ideal_count]))
    return dcg / ideal if ideal > 0 else 0.0


def recall_at_k(relevance: np.ndarray, order: np.ndarray, k: int = 10) -> float:
    y = np.asarray(relevance, dtype=np.int8).reshape(-1)
    top = _ranked_relevance(y, order, k)
    positives = int(y.sum())
    return float(top.sum() / positives) if positives else 0.0


def intra_list_diversity(embeddings: np.ndarray, order: np.ndarray, k: int = 10) -> float:
    """Mean pairwise cosine distance among top-k normalized image embeddings."""
    x = np.asarray(embeddings, dtype=np.float64)
    idx = np.asarray(order, dtype=np.int64)[:k]
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty 2-D array")
    if len(idx) < 2:
        return 0.0
    chosen = x[idx]
    chosen = chosen / np.maximum(np.linalg.norm(chosen, axis=1, keepdims=True), 1e-12)
    similarity = np.clip(chosen @ chosen.T, -1.0, 1.0)
    distances = 1.0 - similarity[np.triu_indices(len(idx), k=1)]
    return float(np.mean(distances))


def unique_coverage_at_k(values: np.ndarray, order: np.ndarray, k: int = 10) -> int:
    """Count distinct metadata values in top-k (descriptive, not a relevance metric)."""
    vals = np.asarray(values).astype(str)
    idx = np.asarray(order, dtype=np.int64)[:k]
    return int(len(set(vals[idx].tolist()) - {"", "nan", "None"}))


def evaluate_ranking(
    relevance: np.ndarray,
    order: np.ndarray,
    image_embeddings: np.ndarray,
    *,
    k_relevance: int = 50,
    k_diversity: int = 10,
    metadata: dict[str, np.ndarray] | None = None,
) -> dict[str, float | int]:
    """Evaluate a single ranking with clearly separated metric families."""
    metrics: dict[str, float | int] = {
        f"ap@{k_relevance}": average_precision_at_k(relevance, order, k_relevance),
        f"ndcg@{k_diversity}": ndcg_at_k(relevance, order, k_diversity),
        f"recall@{k_diversity}": recall_at_k(relevance, order, k_diversity),
        f"ild@{k_diversity}": intra_list_diversity(image_embeddings, order, k_diversity),
    }
    if metadata:
        for name, values in metadata.items():
            metrics[f"unique_{name}@{k_diversity}"] = unique_coverage_at_k(values, order, k_diversity)
    return metrics
