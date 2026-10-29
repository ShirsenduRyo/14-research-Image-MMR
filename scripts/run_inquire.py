"""Run a small or full INQUIRE-Rerank experiment from the command line."""

from __future__ import annotations

import argparse
import json
import platform
from importlib import metadata as package_metadata
from pathlib import Path

from image_rerank.artifacts import ensure_dir, write_json
from image_rerank.config import load_yaml
from image_rerank.data import iter_inquire_queries
from image_rerank.embeddings import EncoderSettings, build_encoder, encoder_cache_key
from image_rerank.experiment import evaluate_queries, prepare_cached_embeddings, load_cached_split
from image_rerank.plots import plot_metric_bars, plot_tradeoff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--split", default=None, choices=["validation", "test"])
    parser.add_argument("--max-queries", type=int, default=None, help="Optional query limit for smoke testing")
    parser.add_argument("--backend", choices=["open_clip", "fashionclip"], default=None)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--skip-encode", action="store_true", help="Evaluate existing query embedding caches only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    exp_cfg = config.get("experiment", {})
    enc_cfg = config.get("encoder", {})
    data_cfg = config.get("data", {})
    artifact_cfg = config.get("artifacts", {})
    reranking_cfg = config.get("reranking", {})

    split = args.split or exp_cfg.get("split", "validation")
    max_queries = args.max_queries if args.max_queries is not None else exp_cfg.get("max_queries")
    backend = args.backend or enc_cfg.get("backend", "open_clip")
    device = args.device or enc_cfg.get("device", "auto")
    if max_queries == 0:
        max_queries = None  # Explicitly request the full split.
    run_name = args.run_name or f"{split}_{backend}_{'smoke' if max_queries else 'full'}"
    cache_root = Path(artifact_cfg.get("cache_root", "artifacts/cache"))
    output_root = ensure_dir(Path(artifact_cfg.get("output_root", "artifacts/experiments")) / run_name)

    settings = EncoderSettings(
        backend=backend,
        model_name=enc_cfg.get("model_name", "ViT-B-32") if backend == "open_clip" else "patrickjohncyh/fashion-clip",
        pretrained=enc_cfg.get("pretrained", "openai"),
        device=device,
        batch_size=int(enc_cfg.get("batch_size", 8)),
        use_amp=bool(enc_cfg.get("use_amp", True)),
    )
    cache_key = encoder_cache_key(settings)
    encoder = None
    if not args.skip_encode:
        encoder = build_encoder(settings)
        print(f"Loading INQUIRE-Rerank split={split!r}; max_queries={max_queries!r}; backend={backend!r}")
        _, cache_stats = prepare_cached_embeddings(
            iter_inquire_queries(
                split=split,
                max_queries=max_queries,
                dataset_id=data_cfg.get("inquire_dataset_id", "evendrow/INQUIRE-Rerank"),
                streaming=bool(data_cfg.get("streaming", True)),
            ),
            encoder,
            split=split,
            cache_root=cache_root,
        )
        print(f"Embedding cache: {json.dumps(cache_stats)}")

    queries = load_cached_split(
        cache_root, split, cache_key,
        max_queries=int(max_queries) if max_queries is not None else None,
    )
    metrics, summary = evaluate_queries(
        queries,
        top_k=int(exp_cfg.get("relevance_k", 50)),
        diversity_k=int(exp_cfg.get("diversity_k", 10)),
        output_dir=output_root,
        run_name=run_name,
        fixed_mmr_lambda=float(reranking_cfg.get("fixed_mmr_lambda", 0.70)),
        adaptive_lambda_min=float(reranking_cfg.get("adaptive_lambda_min", 0.25)),
        adaptive_lambda_max=float(reranking_cfg.get("adaptive_lambda_max", 0.90)),
        adaptive_probe_k=int(reranking_cfg.get("adaptive_probe_k", 20)),
        dpp_quality_floor=float(reranking_cfg.get("dpp_quality_floor", 1e-3)),
        xquad_lambda=float(reranking_cfg.get("xquad_lambda", 0.70)),
        xquad_n_clusters=(
            int(reranking_cfg["xquad_n_clusters"])
            if reranking_cfg.get("xquad_n_clusters") is not None else None
        ),
    )
    figure_dir = ensure_dir(output_root / "figures")
    plot_tradeoff(metrics, figure_dir / "relevance_diversity_latency.png")
    plot_metric_bars(summary, figure_dir / "metric_comparison.png")
    def installed_version(package_name: str) -> str | None:
        try:
            return package_metadata.version(package_name)
        except package_metadata.PackageNotFoundError:
            return None

    write_json(output_root / "metrics" / f"{run_name}_environment.json", {
        "python_version": platform.python_version(),
        "packages": {
            package: installed_version(package)
            for package in ("numpy", "pandas", "torch", "open-clip-torch", "transformers", "datasets")
        },
        "dataset": {
            "dataset_id": data_cfg.get("inquire_dataset_id", "evendrow/INQUIRE-Rerank"),
            "split": split,
            "max_queries": max_queries,
            "streaming": bool(data_cfg.get("streaming", True)),
        },
        "encoder": {
            "backend": settings.backend,
            "model_name": settings.model_name,
            "pretrained": settings.pretrained if backend == "open_clip" else None,
            "cache_key": cache_key,
            "requested_device": device,
            "resolved_device": getattr(encoder, "device", None),
            "batch_size": settings.batch_size,
            "use_amp": settings.use_amp,
        },
        "experiment_config": exp_cfg,
        "reranking_config": reranking_cfg,
    })
    print("\nSummary metrics:")
    print(summary.to_string(index=False))
    print(f"\nArtifacts written under: {output_root.resolve()}")


if __name__ == "__main__":
    main()
