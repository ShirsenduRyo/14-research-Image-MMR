"""Fast, training-free rerankers.

All public ranking functions return an array of candidate indices in ranked order.
Relevance scores are min-max normalized internally where the objective mixes them
with pairwise similarity. Keep this choice fixed for reproducible comparisons.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _validate_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("scores must not be empty")
    if not np.isfinite(values).all():
        raise ValueError("scores contains NaN or infinity")
    return values


def _validate_embeddings(embeddings: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != n:
        raise ValueError("embeddings must have shape (len(scores), embedding_dim)")
    if not np.isfinite(x).all():
        raise ValueError("embeddings contains NaN or infinity")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def minmax_normalize(values: np.ndarray, neutral: float = 0.5) -> np.ndarray:
    """Map a vector to [0, 1]; constant vectors become the neutral value."""
    x = np.asarray(values, dtype=np.float64)
    span = float(np.max(x) - np.min(x))
    if span <= 1e-12:
        return np.full_like(x, fill_value=neutral, dtype=np.float64)
    return (x - np.min(x)) / span


def original_rank(scores: np.ndarray, top_k: Optional[int] = None) -> np.ndarray:
    """Return the original ranking in descending-score order."""
    x = _validate_scores(scores)
    order = np.argsort(-x, kind="stable")
    return order[:top_k] if top_k is not None else order


def _mmr_from_similarity(
    relevance: np.ndarray,
    pairwise_similarity: np.ndarray,
    *,
    lambda_relevance: float,
    top_k: int,
) -> np.ndarray:
    """Greedy MMR core that reuses a precomputed pairwise similarity matrix."""
    n = len(relevance)
    selected = np.zeros(n, dtype=bool)
    max_similarity_to_selected = np.zeros(n, dtype=np.float64)
    order = np.empty(top_k, dtype=np.int64)
    for rank in range(top_k):
        objective = lambda_relevance * relevance - (1.0 - lambda_relevance) * max_similarity_to_selected
        objective[selected] = -np.inf
        idx = int(np.argmax(objective))
        order[rank] = idx
        selected[idx] = True
        max_similarity_to_selected = np.maximum(max_similarity_to_selected, pairwise_similarity[idx])
    return order

def mmr_rerank(
    relevance_scores: np.ndarray,
    image_embeddings: np.ndarray,
    *,
    lambda_relevance: float = 0.7,
    top_k: Optional[int] = None,
) -> np.ndarray:
    """Greedy Maximal Marginal Relevance (MMR).

    Objective: lambda * normalized relevance - (1-lambda) * maximum similarity
    to an already selected image. Pairwise cosine is clipped to [0, 1], so negative
    cosine similarities are not rewarded as extra novelty.
    """
    scores = _validate_scores(relevance_scores)
    if not 0.0 <= lambda_relevance <= 1.0:
        raise ValueError("lambda_relevance must be in [0, 1]")
    x = _validate_embeddings(image_embeddings, len(scores))
    n = len(scores)
    k = n if top_k is None else int(top_k)
    if not 1 <= k <= n:
        raise ValueError("top_k must be in [1, n_candidates]")
    relevance = minmax_normalize(scores)

    # One matrix multiplication per query; all greedy iterations reuse it.
    pairwise = np.clip(x @ x.T, 0.0, 1.0)
    return _mmr_from_similarity(
        relevance, pairwise, lambda_relevance=lambda_relevance, top_k=k
    )


def adaptive_mmr_rerank(
    relevance_scores: np.ndarray,
    image_embeddings: np.ndarray,
    *,
    lambda_min: float = 0.25,
    lambda_max: float = 0.90,
    probe_k: int = 20,
    top_k: Optional[int] = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Query-adaptive MMR using score-gap confidence and shortlist redundancy.

    The controller is deliberately training-free in this first baseline. It is a
    hypothesis to test, not a guaranteed improvement. High score-gap confidence
    increases relevance weight; redundant candidate sets reduce it.
    """
    scores = _validate_scores(relevance_scores)
    x = _validate_embeddings(image_embeddings, len(scores))
    if not (0.0 <= lambda_min <= lambda_max <= 1.0):
        raise ValueError("Require 0 <= lambda_min <= lambda_max <= 1")
    if probe_k < 2:
        raise ValueError("probe_k must be >= 2")

    normalized_scores = minmax_normalize(scores)
    sorted_scores = np.sort(normalized_scores)[::-1]
    gap = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else 1.0
    score_std = float(np.std(normalized_scores))
    gap_z = gap / max(score_std, 1e-8)
    confidence = float(1.0 - np.exp(-gap_z))

    # One pairwise matrix serves both the redundancy estimate and MMR's greedy pass.
    pairwise_full = np.clip(x @ x.T, 0.0, 1.0)
    top_indices = np.argsort(-scores, kind="stable")[: min(probe_k, len(scores))]
    probe_similarity = pairwise_full[np.ix_(top_indices, top_indices)]
    off_diagonal = probe_similarity[~np.eye(len(top_indices), dtype=bool)]
    redundancy = float(np.mean(off_diagonal)) if off_diagonal.size else 0.0
    redundancy = float(np.clip(redundancy, 0.0, 1.0))

    lambda_value = lambda_min + (lambda_max - lambda_min) * confidence * (1.0 - redundancy)
    lambda_value = float(np.clip(lambda_value, lambda_min, lambda_max))
    k = len(scores) if top_k is None else int(top_k)
    if not 1 <= k <= len(scores):
        raise ValueError("top_k must be in [1, n_candidates]")
    order = _mmr_from_similarity(
        normalized_scores, pairwise_full,
        lambda_relevance=lambda_value,
        top_k=k,
    )
    diagnostics = {
        "lambda_relevance": lambda_value,
        "query_confidence": confidence,
        "candidate_redundancy": redundancy,
        "top_score_gap_z": float(gap_z),
    }
    return order, diagnostics


def greedy_dpp_rerank(
    relevance_scores: np.ndarray,
    image_embeddings: np.ndarray,
    *,
    top_k: Optional[int] = None,
    quality_floor: float = 1e-3,
) -> np.ndarray:
    """Greedy MAP approximation for a quality-weighted DPP kernel.

    L = diag(q) @ (X X.T) @ diag(q), where q is positive, normalized relevance.
    This is an exact PSD kernel before numerical regularization; greedy selection
    repeatedly chooses the largest conditional variance. It is an approximation
    to MAP inference, not exact global DPP optimization.
    """
    scores = _validate_scores(relevance_scores)
    x = _validate_embeddings(image_embeddings, len(scores))
    n = len(scores)
    k = n if top_k is None else int(top_k)
    if not 1 <= k <= n:
        raise ValueError("top_k must be in [1, n_candidates]")
    if quality_floor <= 0:
        raise ValueError("quality_floor must be positive")

    quality = np.maximum(minmax_normalize(scores), quality_floor)
    gram = x @ x.T
    kernel = (quality[:, None] * gram) * quality[None, :]
    residual = np.maximum(np.diag(kernel).copy(), 0.0)
    selected = np.zeros(n, dtype=bool)
    factors = np.zeros((k, n), dtype=np.float64)
    order = np.empty(k, dtype=np.int64)

    for rank in range(k):
        residual[selected] = -np.inf
        idx = int(np.argmax(residual))
        if not np.isfinite(residual[idx]) or residual[idx] <= 1e-12:
            # Degenerate/singular shortlist: complete deterministically by relevance.
            remaining = np.flatnonzero(~selected)
            remaining = remaining[np.argsort(-scores[remaining], kind="stable")]
            order[rank:] = remaining[: k - rank]
            break
        order[rank] = idx
        selected[idx] = True
        if rank == k - 1:
            break
        denom = np.sqrt(max(residual[idx], 1e-12))
        correction = factors[:rank, idx] @ factors[:rank, :] if rank else 0.0
        new_row = (kernel[idx] - correction) / denom
        factors[rank] = new_row
        residual = np.maximum(residual - new_row**2, 0.0)
        residual[selected] = -np.inf
    return order


def xquad_rerank(
    relevance_scores: np.ndarray,
    aspect_probabilities: np.ndarray,
    aspect_weights: Optional[np.ndarray] = None,
    *,
    lambda_relevance: float = 0.7,
    top_k: Optional[int] = None,
) -> np.ndarray:
    """xQuAD-style diversification over supplied candidate-aspect probabilities.

    `aspect_probabilities[i, a]` is P(aspect a is covered by candidate i), and
    `aspect_weights[a]` is the query's importance for aspect a. With inferred
    clusters these are exploratory pseudo-aspects, not human-annotated intents.
    """
    scores = _validate_scores(relevance_scores)
    aspects = np.asarray(aspect_probabilities, dtype=np.float64)
    if aspects.ndim != 2 or aspects.shape[0] != len(scores) or aspects.shape[1] == 0:
        raise ValueError("aspect_probabilities must have shape (n_candidates, n_aspects)")
    if not np.isfinite(aspects).all() or np.any(aspects < 0.0) or np.any(aspects > 1.0):
        raise ValueError("aspect_probabilities must be finite and in [0, 1]")
    if not 0.0 <= lambda_relevance <= 1.0:
        raise ValueError("lambda_relevance must be in [0, 1]")
    weights = np.ones(aspects.shape[1], dtype=np.float64) if aspect_weights is None else np.asarray(aspect_weights, dtype=np.float64)
    if weights.shape != (aspects.shape[1],) or np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("aspect_weights must be a non-negative vector of length n_aspects")
    if weights.sum() <= 0:
        raise ValueError("aspect_weights must sum to a positive value")
    weights = weights / weights.sum()
    k = len(scores) if top_k is None else int(top_k)
    if not 1 <= k <= len(scores):
        raise ValueError("top_k must be in [1, n_candidates]")

    relevance = minmax_normalize(scores)
    selected = np.zeros(len(scores), dtype=bool)
    aspect_uncovered = np.ones(aspects.shape[1], dtype=np.float64)
    order = np.empty(k, dtype=np.int64)
    for rank in range(k):
        novelty_gain = aspects @ (weights * aspect_uncovered)
        objective = lambda_relevance * relevance + (1.0 - lambda_relevance) * novelty_gain
        objective[selected] = -np.inf
        idx = int(np.argmax(objective))
        order[rank] = idx
        selected[idx] = True
        aspect_uncovered *= (1.0 - aspects[idx])
    return order


def cluster_aspect_probabilities(
    image_embeddings: np.ndarray, *, n_clusters: int = 8, random_state: int = 42
) -> np.ndarray:
    """Infer soft-ish pseudo-aspects from candidate embeddings using KMeans.

    Returns one-hot cluster membership. This is a diagnostic diversification
    baseline; the clusters are not guaranteed to correspond to useful query intents.
    """
    from sklearn.cluster import KMeans

    x = np.asarray(image_embeddings, dtype=np.float32)
    if x.ndim != 2 or len(x) == 0:
        raise ValueError("image_embeddings must be a non-empty 2-D array")
    n = len(x)
    k = int(np.clip(n_clusters, 1, n))
    if k == 1:
        return np.ones((n, 1), dtype=np.float32)
    labels = KMeans(n_clusters=k, random_state=random_state, n_init=5).fit_predict(x)
    return np.eye(k, dtype=np.float32)[labels]
