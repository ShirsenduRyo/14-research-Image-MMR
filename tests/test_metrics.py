import numpy as np
import pytest

from image_rerank.metrics import (
    average_precision_at_k,
    intra_list_diversity,
    ndcg_at_k,
    recall_at_k,
)


def test_metrics_perfect_ranking():
    y = np.array([1, 0, 1, 0])
    order = np.array([0, 2, 1, 3])
    x = np.eye(4, dtype=float)
    assert average_precision_at_k(y, order, 4) == pytest.approx(1.0)
    assert ndcg_at_k(y, order, 2) == pytest.approx(1.0)
    assert recall_at_k(y, order, 2) == pytest.approx(1.0)
    assert intra_list_diversity(x, order, 2) == pytest.approx(1.0)


def test_metrics_zero_relevant():
    y = np.zeros(4, dtype=np.int8)
    order = np.arange(4)
    assert average_precision_at_k(y, order, 3) == 0.0
    assert ndcg_at_k(y, order, 3) == 0.0
    assert recall_at_k(y, order, 3) == 0.0


def test_ild_is_zero_for_duplicate_embeddings():
    x = np.ones((4, 3), dtype=float)
    assert intra_list_diversity(x, np.arange(4), 4) == pytest.approx(0.0)


def test_duplicate_indices_rejected():
    with pytest.raises(ValueError):
        average_precision_at_k(np.array([1, 0, 1]), np.array([0, 0]), 2)
