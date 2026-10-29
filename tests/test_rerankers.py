import numpy as np
import pytest

from image_rerank.rerankers import (
    adaptive_mmr_rerank,
    greedy_dpp_rerank,
    mmr_rerank,
    original_rank,
    xquad_rerank,
)


def normalized_rng(seed=7, n=20, d=16):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def assert_valid_order(order, n, k):
    assert order.shape == (k,)
    assert len(set(order.tolist())) == k
    assert np.all((order >= 0) & (order < n))


def test_original_rank_descending_and_stable():
    order = original_rank(np.array([0.2, 0.9, 0.9, -0.1]))
    assert order.tolist() == [1, 2, 0, 3]


def test_mmr_returns_unique_indices_and_lambda_one_matches_relevance():
    scores = np.linspace(0, 1, 20)
    x = normalized_rng()
    order = mmr_rerank(scores, x, lambda_relevance=1.0, top_k=10)
    assert_valid_order(order, 20, 10)
    assert order.tolist() == list(np.argsort(-scores, kind="stable")[:10])


def test_mmr_rejects_bad_inputs():
    with pytest.raises(ValueError):
        mmr_rerank(np.array([1.0, 2.0]), normalized_rng(n=2), lambda_relevance=1.1)
    with pytest.raises(ValueError):
        mmr_rerank(np.array([1.0, np.nan]), normalized_rng(n=2))


def test_adaptive_mmr_diagnostics_in_bounds():
    scores = np.linspace(0, 1, 20)
    order, diagnostics = adaptive_mmr_rerank(scores, normalized_rng(), top_k=8)
    assert_valid_order(order, 20, 8)
    assert 0.25 <= diagnostics["lambda_relevance"] <= 0.90
    assert 0.0 <= diagnostics["query_confidence"] <= 1.0
    assert 0.0 <= diagnostics["candidate_redundancy"] <= 1.0


def test_dpp_returns_unique_indices():
    scores = np.linspace(0, 1, 20)
    order = greedy_dpp_rerank(scores, normalized_rng(), top_k=10)
    assert_valid_order(order, 20, 10)


def test_xquad_diversifies_aspects():
    scores = np.array([0.99, 0.98, 0.50, 0.40])
    aspects = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
    order = xquad_rerank(scores, aspects, lambda_relevance=0.2, top_k=2)
    assert_valid_order(order, 4, 2)
    assert len(set(np.argmax(aspects[order], axis=1).tolist())) == 2


def test_xquad_rejects_bad_probabilities():
    with pytest.raises(ValueError):
        xquad_rerank(np.array([1.0, 0.0]), np.array([[2.0], [0.0]]))
