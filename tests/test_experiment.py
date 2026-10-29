from pathlib import Path

from image_rerank.data import make_synthetic_queries
from image_rerank.experiment import evaluate_queries, load_cached_split
from image_rerank.schema import QueryEmbeddings


def test_evaluate_queries_writes_metrics_rankings_and_metadata(tmp_path: Path):
    query_objects = []
    for item in make_synthetic_queries(n_queries=2, n_candidates=20, embedding_dim=12, seed=10):
        query_objects.append(QueryEmbeddings(
            query_id=item["query_id"],
            query=item["query"],
            query_embedding=item["query_embedding"],
            image_embeddings=item["image_embeddings"],
            image_ids=item["image_ids"],
            original_scores=item["original_scores"],
            relevance=item["relevance"],
            metadata=item["metadata"],
        ))

    metrics, summary = evaluate_queries(
        query_objects, top_k=15, diversity_k=5,
        output_dir=tmp_path, run_name="test_run", fixed_mmr_lambda=0.50,
    )
    assert metrics["query_id"].nunique() == 2
    assert metrics["method"].nunique() == 6
    assert "mmr_fixed_0.50" in set(metrics["method"])
    assert "ap@15" in metrics.columns
    assert "ap@15_mean" in summary.columns
    assert (tmp_path / "metrics" / "test_run_per_query_metrics.csv").exists()
    assert (tmp_path / "metrics" / "test_run_summary.csv").exists()
    assert (tmp_path / "metrics" / "test_run_run_metadata.json").exists()
    assert (tmp_path / "rankings" / "test_run_rankings.csv").exists()
    import pandas as pd
    rankings = pd.read_csv(tmp_path / "rankings" / "test_run_rankings.csv")
    assert "metadata__synthetic_cluster" in rankings.columns


def test_load_cached_split_preserves_index_order_and_query_limit(tmp_path: Path):
    from image_rerank.artifacts import query_cache_path, save_query_embeddings, write_json

    items = make_synthetic_queries(n_queries=3, n_candidates=12, embedding_dim=8, seed=3)
    query_ids = [str(item["query_id"]) for item in items]
    encoder_key = "openclip-test"
    # Store files in an order that differs from lexical query ID order.
    for item in items:
        cache_path = query_cache_path(tmp_path, "validation", encoder_key, str(item["query_id"]))
        save_query_embeddings(cache_path, QueryEmbeddings(
            query_id=str(item["query_id"]),
            query=str(item["query"]),
            query_embedding=item["query_embedding"],
            image_embeddings=item["image_embeddings"],
            image_ids=item["image_ids"],
            original_scores=item["original_scores"],
            relevance=item["relevance"],
            metadata=item["metadata"],
        ))
    index_path = tmp_path / "validation" / encoder_key / "index.json"
    write_json(index_path, {
        "split": "validation", "encoder_key": encoder_key,
        "query_ids": [query_ids[2], query_ids[0], query_ids[1]],
    })

    loaded = load_cached_split(tmp_path, "validation", encoder_key)
    assert [item.query_id for item in loaded] == [query_ids[2], query_ids[0], query_ids[1]]
    limited = load_cached_split(tmp_path, "validation", encoder_key, max_queries=2)
    assert [item.query_id for item in limited] == [query_ids[2], query_ids[0]]
