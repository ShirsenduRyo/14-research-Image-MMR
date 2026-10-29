from pathlib import Path

import numpy as np

from image_rerank.artifacts import load_query_embeddings, save_query_embeddings
from image_rerank.schema import QueryEmbeddings


def test_embedding_cache_round_trip(tmp_path: Path):
    query = QueryEmbeddings(
        query_id="q-1",
        query="test query",
        query_embedding=np.ones(4, dtype=np.float32),
        image_embeddings=np.eye(4, dtype=np.float32),
        image_ids=np.array(["a", "b", "c", "d"]),
        original_scores=np.array([0.9, 0.8, 0.7, 0.6]),
        relevance=np.array([1, 0, 1, 0]),
        metadata={"category": np.array(["x", "x", "y", "z"])},
    )
    path = save_query_embeddings(tmp_path / "cache.npz", query)
    restored = load_query_embeddings(path)
    assert restored.query == query.query
    assert restored.query_id == query.query_id
    np.testing.assert_allclose(restored.query_embedding, query.query_embedding, atol=1e-3)
    np.testing.assert_array_equal(restored.metadata["category"], query.metadata["category"])


def test_cache_namespace_sanitizes_slashes_consistently(tmp_path: Path):
    from image_rerank.artifacts import query_cache_path, safe_name

    path = query_cache_path(tmp_path, "validation", "fashionclip/org/model", "qid")
    expected = tmp_path / "validation" / safe_name("fashionclip/org/model") / "qid.npz"
    assert path == expected
