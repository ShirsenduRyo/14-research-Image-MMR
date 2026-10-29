"""Public benchmark loader and deterministic synthetic data for smoke tests."""

from __future__ import annotations

import hashlib
from typing import Iterator

import numpy as np

from .schema import QueryRecord


DATASET_ID = "evendrow/INQUIRE-Rerank"


def stable_query_id(query: str) -> str:
    """Stable short ID, independent of row ordering."""
    return hashlib.sha1(query.strip().encode("utf-8")).hexdigest()[:12]


def _first_existing(columns: list[str], candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Dataset is missing all expected columns {candidates}. Available: {columns}")


def iter_inquire_queries(
    split: str = "validation", *, max_queries: int | None = None, dataset_id: str = DATASET_ID,
    streaming: bool = True,
) -> Iterator[QueryRecord]:
    """Yield one query at a time from the official Hugging Face INQUIRE-Rerank data.

    Streaming mode avoids eagerly materializing the entire split and permits a small
    smoke run with ``max_queries``. The official dataset is stored query-by-query;
    the loader checks that adjacent rows belong to the same query and emits one
    shortlist at a time. Check the dataset license (CC BY-NC 4.0) before use.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Install the 'datasets' package: pip install -e '.[dev]'") from exc

    dataset = load_dataset(dataset_id, split=split, streaming=streaming)
    columns = list(dataset.column_names)
    query_col = _first_existing(columns, ("query", "query_text", "text"))
    image_col = _first_existing(columns, ("image", "images", "pixel_values"))
    relevance_col = _first_existing(columns, ("relevant", "relevance", "is_relevant", "label"))
    score_col = _first_existing(columns, ("clip_score", "initial_score", "score"))
    image_id_col = _first_existing(columns, ("inat24_image_id", "image_id", "id"))
    metadata_cols = [
        col for col in ("supercategory", "category", "iconic_group", "iconic group", "inat24_species_name")
        if col in columns
    ]

    from itertools import groupby

    query_count = 0
    seen_queries: set[str] = set()
    row_iterator = iter(dataset)
    for query_text, group in groupby(row_iterator, key=lambda row: str(row[query_col])):
        if max_queries is not None and query_count >= max_queries:
            break
        # Each group is one query's fixed shortlist (normally 100 rows). Holding only
        # this list bounds memory regardless of total split size.
        if query_text in seen_queries:
            raise ValueError(
                "INQUIRE rows are not contiguous by query; cannot stream group safely. "
                "Use a sorted local manifest or non-streaming loader for this dataset revision."
            )
        seen_queries.add(query_text)
        rows = list(group)
        if not rows:
            continue
        images = [row[image_col] for row in rows]
        metadata = {
            col.replace(" ", "_"): np.asarray([row.get(col, "") for row in rows]).astype(str)
            for col in metadata_cols
        }
        ids = np.asarray([row[image_id_col] for row in rows]).astype(str)
        relevance = np.asarray([row[relevance_col] for row in rows], dtype=np.int8)
        scores = np.asarray([row[score_col] for row in rows], dtype=np.float32)
        yield QueryRecord(
            query_id=stable_query_id(query_text),
            query=query_text,
            images=images,
            image_ids=ids,
            original_scores=scores,
            relevance=relevance,
            metadata=metadata,
        )
        query_count += 1


def make_synthetic_queries(
    *, n_queries: int = 8, n_candidates: int = 50, embedding_dim: int = 64, seed: int = 42
) -> list[dict[str, object]]:
    """Generate deterministic synthetic retrieval cases for offline smoke tests.

    This checks ranking logic, plotting, and artifact persistence only. It is not a
    substitute for benchmark results and must never be described as model performance.
    """
    rng = np.random.default_rng(seed)
    result: list[dict[str, object]] = []
    for query_idx in range(n_queries):
        q = rng.normal(size=embedding_dim)
        q /= max(np.linalg.norm(q), 1e-12)
        images = rng.normal(size=(n_candidates, embedding_dim))
        # Assign relevance first, then create a noisy rather than perfect visual signal.
        # This prevents the synthetic metric demo from being trivially perfect.
        n_relevant = max(5, int(round(n_candidates * 0.35)))
        relevant_indices = rng.choice(n_candidates, size=min(n_relevant, n_candidates - 1), replace=False)
        relevance = np.zeros(n_candidates, dtype=np.int8)
        relevance[relevant_indices] = 1
        images[relevance == 1] += 0.65 * q
        images /= np.maximum(np.linalg.norm(images, axis=1, keepdims=True), 1e-12)
        similarity = images @ q
        # Slightly noisy initial score mimics a first-stage retriever.
        initial_scores = similarity + rng.normal(0, 0.14, size=n_candidates)
        result.append({
            "query_id": f"synthetic-{query_idx:03d}",
            "query": f"synthetic query {query_idx}",
            "query_embedding": q.astype(np.float32),
            "image_embeddings": images.astype(np.float32),
            "original_scores": initial_scores.astype(np.float32),
            "relevance": relevance,
            "image_ids": np.asarray([f"img-{query_idx}-{i}" for i in range(n_candidates)]),
            "metadata": {"synthetic_cluster": (np.argmax(images @ images.T, axis=1) % 4).astype(str)},
        })
    return result


def iter_local_manifest(
    manifest_path: str,
    *,
    image_root: str | None = None,
    max_queries: int | None = None,
) -> Iterator[QueryRecord]:
    """Load a user-prepared text-to-image retrieval manifest.

    Required CSV columns: ``query``, ``image_path``, ``relevant``, ``initial_score``.
    Optional ``image_id`` and any descriptive columns (e.g., category/color/style) are
    retained as metadata. The manifest can be built from licensed DeepFashion or
    other fashion data; this repository does not redistribute those images.
    """
    from pathlib import Path
    import pandas as pd
    from PIL import Image

    manifest = Path(manifest_path).expanduser().resolve()
    frame = pd.read_csv(manifest)
    required = {"query", "image_path", "relevant", "initial_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Manifest is empty")
    root = Path(image_root).expanduser().resolve() if image_root else manifest.parent
    metadata_cols = [
        col for col in frame.columns
        if col not in {"query", "image_path", "relevant", "initial_score", "image_id"}
    ]

    yielded = 0
    for query_text, rows in frame.groupby("query", sort=False):
        if max_queries is not None and yielded >= max_queries:
            break
        images = []
        paths = []
        for relative_path in rows["image_path"].astype(str):
            path = Path(relative_path)
            if not path.is_absolute():
                path = root / path
            if not path.exists():
                raise FileNotFoundError(f"Image referenced in manifest does not exist: {path}")
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
            paths.append(str(path))
        metadata = {col: rows[col].fillna("").astype(str).to_numpy() for col in metadata_cols}
        image_ids = rows["image_id"].astype(str).to_numpy() if "image_id" in rows else np.asarray(paths)
        yield QueryRecord(
            query_id=stable_query_id(str(query_text)),
            query=str(query_text),
            images=images,
            image_ids=image_ids,
            original_scores=rows["initial_score"].to_numpy(dtype=np.float32),
            relevance=rows["relevant"].to_numpy(dtype=np.int8),
            metadata=metadata,
        )
        yielded += 1
