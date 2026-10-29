"""Experiment orchestration: embedding cache, ranking, metrics, and artifacts."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .artifacts import load_query_embeddings, query_cache_path, safe_name, save_query_embeddings, write_dataframe, write_json
from .metrics import evaluate_ranking
from .rerankers import (
    adaptive_mmr_rerank,
    cluster_aspect_probabilities,
    greedy_dpp_rerank,
    mmr_rerank,
    original_rank,
    xquad_rerank,
)
from .schema import QueryEmbeddings, QueryRecord


def embed_record(record: QueryRecord, encoder) -> QueryEmbeddings:
    """Embed a single query shortlist; only one shortlist is held in memory."""
    image_embeddings = encoder.encode_images(record.images)
    query_embedding = encoder.encode_texts([record.query])[0]
    return QueryEmbeddings(
        query_id=record.query_id,
        query=record.query,
        query_embedding=query_embedding,
        image_embeddings=image_embeddings,
        image_ids=record.image_ids,
        original_scores=record.original_scores,
        relevance=record.relevance,
        metadata=record.metadata,
    )


def prepare_cached_embeddings(
    records: Iterable[QueryRecord], encoder, *, split: str, cache_root: str | Path
) -> tuple[list[Path], dict[str, int]]:
    """Create missing per-query embedding caches and reuse existing ones.

    Returns cache paths plus counts. Images are encoded and persisted one query at a
    time, so a partially completed run can resume after interruption.
    """
    paths: list[Path] = []
    query_ids: list[str] = []
    counts = {"cache_hits": 0, "encoded_queries": 0}
    for record in records:
        query_ids.append(record.query_id)
        path = query_cache_path(cache_root, split, encoder.cache_key, record.query_id)
        if path.exists():
            paths.append(path)
            counts["cache_hits"] += 1
            continue
        if record.images is None:
            raise RuntimeError(f"Cache miss for {record.query_id}, but images were not loaded")
        embedded = embed_record(record, encoder)
        save_query_embeddings(path, embedded)
        paths.append(path)
        counts["encoded_queries"] += 1

    # Keep dataset query order stable across runs. If a small run is followed by a
    # full run, the existing prefix is preserved and newly seen queries are appended.
    index_path = Path(cache_root) / safe_name(split) / safe_name(encoder.cache_key) / "index.json"
    prior_ids: list[str] = []
    if index_path.exists():
        from json import load
        with index_path.open("r", encoding="utf-8") as file:
            saved_index = load(file)
        prior_ids = [str(value) for value in saved_index.get("query_ids", [])]
    ordered_ids = list(dict.fromkeys([*prior_ids, *query_ids]))
    write_json(index_path, {
        "split": split,
        "encoder_key": encoder.cache_key,
        "query_ids": ordered_ids,
        "n_indexed_queries": len(ordered_ids),
    })
    return paths, counts


def load_cached_split(
    cache_root: str | Path,
    split: str,
    encoder_key: str,
    *,
    max_queries: int | None = None,
) -> list[QueryEmbeddings]:
    """Load caches in persisted dataset order, optionally limiting the query count."""
    folder = Path(cache_root) / safe_name(split) / safe_name(encoder_key)
    if not folder.exists():
        raise FileNotFoundError(f"Embedding cache not found: {folder}. Run notebook 01 first.")
    index_path = folder / "index.json"
    if index_path.exists():
        from json import load
        with index_path.open("r", encoding="utf-8") as file:
            query_ids = [str(value) for value in load(file).get("query_ids", [])]
        paths = [query_cache_path(cache_root, split, encoder_key, query_id) for query_id in query_ids]
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Cache index references {len(missing)} missing files; re-run the embedding notebook. "
                f"First missing path: {missing[0]}"
            )
    else:
        # Backward compatibility for caches created before the index was introduced.
        paths = sorted(path for path in folder.glob("*.npz") if path.is_file())
    if max_queries is not None:
        if max_queries < 1:
            raise ValueError("max_queries must be >= 1 or None")
        paths = paths[:max_queries]
    if not paths:
        raise FileNotFoundError(f"No .npz query caches found in {folder}")
    return [load_query_embeddings(path) for path in paths]


def evaluate_queries(
    queries: Iterable[QueryEmbeddings],
    *,
    top_k: int = 50,
    diversity_k: int = 10,
    output_dir: str | Path = "artifacts/experiment",
    run_name: str = "run",
    fixed_mmr_lambda: float = 0.70,
    adaptive_lambda_min: float = 0.25,
    adaptive_lambda_max: float = 0.90,
    adaptive_probe_k: int = 20,
    dpp_quality_floor: float = 1e-3,
    xquad_lambda: float = 0.70,
    xquad_n_clusters: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all methods and persist per-query metrics and ranked candidates.

    Reranker latency is measured on already available embeddings after one warm-up
    call per method. It excludes image decoding, model inference, Hugging Face
    downloads, and cache I/O.
    """
    if top_k < 1 or diversity_k < 1:
        raise ValueError("top_k and diversity_k must both be positive integers")
    output_dir = Path(output_dir)
    metrics_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []
    queries = list(queries)
    if not queries:
        raise ValueError("No query embeddings were supplied for evaluation")
    if queries:
        # Warm up imports, BLAS paths, and the KMeans implementation outside latency
        # measurements; otherwise the first measured call includes Python cold-start.
        warm_query = queries[0]
        warm_scores = warm_query.encoder_scores
        warm_embeddings = warm_query.normalized_image_embeddings
        warm_cutoff = min(max(top_k, diversity_k), len(warm_scores))
        _ = original_rank(warm_query.original_scores, warm_cutoff)
        _ = original_rank(warm_scores, warm_cutoff)
        _ = mmr_rerank(
            warm_scores, warm_embeddings, lambda_relevance=fixed_mmr_lambda, top_k=warm_cutoff
        )
        _ = adaptive_mmr_rerank(
            warm_scores, warm_embeddings, lambda_min=adaptive_lambda_min,
            lambda_max=adaptive_lambda_max, probe_k=adaptive_probe_k, top_k=warm_cutoff,
        )
        _ = greedy_dpp_rerank(
            warm_scores, warm_embeddings, quality_floor=dpp_quality_floor, top_k=warm_cutoff
        )
        if len(warm_scores) >= 4:
            warm_clusters = (
                xquad_n_clusters
                if xquad_n_clusters is not None
                else min(8, max(2, int(np.sqrt(len(warm_scores)))))
            )
            warm_aspects = cluster_aspect_probabilities(warm_embeddings, n_clusters=warm_clusters)
            _ = xquad_rerank(
                warm_scores, warm_aspects, lambda_relevance=xquad_lambda, top_k=warm_cutoff
            )
    for query in queries:
        scores = query.encoder_scores
        embeddings = query.normalized_image_embeddings
        cutoff = min(max(top_k, diversity_k), len(scores))
        # Each method is executed once per query; reported latency is the wall time
        # for its ranking call on cached embeddings, excluding model inference and I/O.
        method_calls = {
            "initial_clip": lambda: (original_rank(query.original_scores, cutoff), {}),
            "encoder_similarity": lambda: (original_rank(scores, cutoff), {}),
            f"mmr_fixed_{fixed_mmr_lambda:.2f}": lambda: (
                mmr_rerank(scores, embeddings, lambda_relevance=fixed_mmr_lambda, top_k=cutoff), {}
            ),
            "adaptive_mmr": lambda: adaptive_mmr_rerank(
                scores, embeddings, lambda_min=adaptive_lambda_min,
                lambda_max=adaptive_lambda_max, probe_k=adaptive_probe_k, top_k=cutoff,
            ),
            "greedy_dpp": lambda: (
                greedy_dpp_rerank(scores, embeddings, quality_floor=dpp_quality_floor, top_k=cutoff), {}
            ),
        }
        if len(scores) >= 4:
            def cluster_xquad_call():
                # Include pseudo-aspect discovery in latency because it is required
                # by this reranker for every new candidate shortlist.
                n_clusters = (
                    xquad_n_clusters
                    if xquad_n_clusters is not None
                    else min(8, max(2, int(np.sqrt(len(scores)))))
                )
                aspects = cluster_aspect_probabilities(embeddings, n_clusters=n_clusters)
                order = xquad_rerank(scores, aspects, lambda_relevance=xquad_lambda, top_k=cutoff)
                return order, {"n_pseudo_aspects": float(aspects.shape[1])}
            method_calls["cluster_xquad"] = cluster_xquad_call

        for method_name, call in method_calls.items():
            method_start = time.perf_counter()
            order, diagnostic = call()
            order = np.asarray(order, dtype=np.int64)
            elapsed_ms = (time.perf_counter() - method_start) * 1000.0
            eval_metrics = evaluate_ranking(
                query.relevance,
                order,
                embeddings,
                k_relevance=top_k,
                k_diversity=diversity_k,
                metadata=query.metadata,
            )
            metrics_rows.append({
                "query_id": query.query_id,
                "query": query.query,
                "method": method_name,
                "rerank_latency_ms": elapsed_ms,
                **eval_metrics,
                **diagnostic,
            })
            ranked_order = order[:max(top_k, diversity_k)]
            for rank, candidate_idx in enumerate(ranked_order, start=1):
                ranking_rows.append({
                    "query_id": query.query_id,
                    "query": query.query,
                    "method": method_name,
                    "rank": rank,
                    "image_id": str(query.image_ids[candidate_idx]),
                    "initial_clip_score": float(query.original_scores[candidate_idx]),
                    "encoder_similarity": float(scores[candidate_idx]),
                    "relevant": int(query.relevance[candidate_idx]),
                    **{
                        f"metadata__{key}": str(values[candidate_idx])
                        for key, values in query.metadata.items()
                    },
                })
    metrics_df = pd.DataFrame(metrics_rows)
    rankings_df = pd.DataFrame(ranking_rows)

    metrics_path = output_dir / "metrics" / f"{run_name}_per_query_metrics.csv"
    rankings_path = output_dir / "rankings" / f"{run_name}_rankings.csv"
    summary = summarize_metrics(metrics_df)
    summary_path = output_dir / "metrics" / f"{run_name}_summary.csv"
    write_dataframe(metrics_path, metrics_df)
    write_dataframe(rankings_path, rankings_df)
    write_dataframe(summary_path, summary)
    write_json(output_dir / "metrics" / f"{run_name}_run_metadata.json", {
        "run_name": run_name,
        "n_queries": len(queries),
        "top_k_relevance": top_k,
        "top_k_diversity": diversity_k,
        "reranking_parameters": {
            "fixed_mmr_lambda": fixed_mmr_lambda,
            "adaptive_lambda_min": adaptive_lambda_min,
            "adaptive_lambda_max": adaptive_lambda_max,
            "adaptive_probe_k": adaptive_probe_k,
            "dpp_quality_floor": dpp_quality_floor,
            "xquad_lambda": xquad_lambda,
            "xquad_n_clusters": xquad_n_clusters,
        },
        "latency_scope": "warm reranking stage on cached embeddings; excludes encoding, image decoding, downloads, and I/O",
        "warmup": "one unmeasured call per available method before query-level latency timing",
        "methods": sorted(metrics_df["method"].unique().tolist()),
    })
    return metrics_df, summary


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate means and bootstrap-free dispersion; use paired bootstrap separately."""
    metric_cols = [
        col for col in metrics_df.columns
        if col.startswith(("ap@", "ndcg@", "recall@", "ild@", "unique_"))
        or col == "rerank_latency_ms"
    ]
    grouped = metrics_df.groupby("method")[metric_cols].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns.to_flat_index()]
    result = grouped.reset_index()
    if "rerank_latency_ms" in metrics_df.columns:
        latency_quantiles = metrics_df.groupby("method")["rerank_latency_ms"].agg(
            rerank_latency_ms_p50="median",
            rerank_latency_ms_p95=lambda values: float(np.percentile(values, 95)),
        ).reset_index()
        result = result.merge(latency_quantiles, on="method", how="left")
    return result


def sweep_mmr_lambdas(
    queries: Iterable[QueryEmbeddings], *, lambdas: Iterable[float] = tuple(np.arange(0.1, 1.0, 0.1)), k: int = 50
) -> pd.DataFrame:
    """Return per-query and aggregate quality/latency for fixed lambda settings."""
    rows: list[dict[str, object]] = []
    query_list = list(queries)
    if query_list:
        warm_query = query_list[0]
        _ = mmr_rerank(
            warm_query.encoder_scores,
            warm_query.normalized_image_embeddings,
            lambda_relevance=0.5,
            top_k=min(k, len(warm_query.image_ids)),
        )
    for query in query_list:
        scores = query.encoder_scores
        embeddings = query.normalized_image_embeddings
        for weight in lambdas:
            start = time.perf_counter()
            order = mmr_rerank(scores, embeddings, lambda_relevance=float(weight), top_k=min(k, len(scores)))
            latency_ms = (time.perf_counter() - start) * 1000.0
            values = evaluate_ranking(query.relevance, order, embeddings, k_relevance=k, k_diversity=min(10, k))
            rows.append({"query_id": query.query_id, "method": "mmr", "lambda_relevance": float(weight), "rerank_latency_ms": latency_ms, **values})
    return pd.DataFrame(rows)
