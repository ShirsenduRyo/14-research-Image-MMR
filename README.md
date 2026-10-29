# Image Rerank Lab

A reproducible research repository for **image retrieval reranking under relevance, diversity, and latency constraints**. The first implementation uses frozen CLIP-family embeddings and compares the initial ranking, encoder-similarity ranking, fixed-weight MMR, query-adaptive MMR, greedy DPP, and an exploratory cluster-based xQuAD variant.

The reusable implementation is in `src/image_rerank/`. Guided, cell-by-cell notebooks are kept separately in `notebooks/`. Metric CSVs, ranked candidate CSVs, embedding caches, plots, and run metadata are written under `artifacts/`.

> **Research status:** the adaptive MMR controller is a hypothesis to evaluate, not a known improvement. Synthetic outputs only test the pipeline; they are not benchmark results.

## What is implemented

- `initial_clip`: original INQUIRE-Rerank `clip_score` order.
- `encoder_similarity`: candidate order from the chosen frozen encoder's text–image cosine similarity.
- `mmr_fixed_0.70`: fixed-weight MMR baseline.
- `adaptive_mmr`: MMR with a training-free query confidence / candidate redundancy controller.
- `greedy_dpp`: greedy MAP approximation for a quality-weighted DPP kernel.
- `cluster_xquad`: xQuAD-style diversity using KMeans pseudo-aspects inferred from candidates. This is an exploratory visual-cluster baseline, **not** human-annotated query-aspect coverage.

Metrics keep separate concepts separate: candidate-list AP@50, nDCG@10, Recall@10, intra-list diversity (ILD@10), optional taxonomy coverage, and CPU/GPU reranking latency. Latency excludes model inference, image decoding, downloads, and cache I/O; the run metadata records that scope. Do not compare the measured number directly to an end-to-end service SLA without measuring the full path.

## Repository layout

```text
image-rerank-lab/
├── configs/experiment.yaml
├── notebooks/
│   ├── 00_synthetic_smoke_test.ipynb
│   ├── 01_inquire_data_and_embeddings.ipynb
│   ├── 02_baselines_and_metrics.ipynb
│   ├── 03_adaptive_controller_and_tradeoffs.ipynb
│   ├── 04_fashionclip_transfer.ipynb
│   └── 05_algorithm_walkthrough.ipynb
├── scripts/run_inquire.py
├── scripts/run_synthetic_demo.py
├── src/image_rerank/
│   ├── data.py          # INQUIRE streaming loader + local CSV manifest loader
│   ├── embeddings.py    # frozen OpenCLIP and Hugging Face FashionCLIP
│   ├── rerankers.py     # MMR, adaptive MMR, greedy DPP, xQuAD-style baseline
│   ├── metrics.py       # relevance and diversity metrics
│   ├── experiment.py    # evaluation orchestration and durable artifacts
│   ├── artifacts.py     # compressed per-query embedding cache and metrics writers
│   ├── plots.py         # Seaborn darkgrid evaluation plots
│   └── statistics.py    # paired bootstrap confidence intervals
├── tests/
├── artifacts/demo/     # small generated synthetic smoke-test outputs
└── pyproject.toml       # runtime, notebook, and development dependencies
```

## Hardware profile

The default setup targets **4 GB dedicated GPU memory**:

- Frozen `ViT-B-32` OpenCLIP or FashionCLIP; no fine-tuning.
- Conservative image batch size of 8, configurable in YAML/CLI.
- Embeddings saved per query in compressed NPZ files as float16; loaded as float32 for numerical scoring.
- One query shortlist is processed at a time.
- Reranking uses a small candidate–candidate similarity matrix. At 100 candidates, this matrix is negligible compared with vision-encoder inference.

If CUDA memory is tight, set `batch_size: 4`, `use_amp: true`, and `device: auto` or `cpu`. Use the official [PyTorch install selector](https://pytorch.org/get-started/locally/) to choose a wheel matching the installed driver/CUDA runtime. The version selected in `pyproject.toml` is the dependency floor; the official selector determines the platform-specific install command.

## Installation

Project metadata permits Python 3.10–3.13; the CI matrix currently checks Python 3.10–3.12. Python 3.10 or 3.11 is a conservative choice for ML package compatibility.

```bash
# 1. Create and activate an environment (Windows PowerShell example)
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install a PyTorch build appropriate for your machine first.
#    Use the current command from https://pytorch.org/get-started/locally/
#    (CPU wheels also work for the smoke test.)

# 3. Install the project and all notebook/dev dependencies
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

# 4. Run unit tests and an offline end-to-end smoke experiment
pytest
python scripts/run_synthetic_demo.py
```

Linux/macOS venv activation uses `source .venv/bin/activate`.

The full `[project].dependencies` list and `[project.optional-dependencies].dev` list are in `pyproject.toml`. There is no untracked `requirements.txt` to get out of sync with the package metadata. Encoder, reranker, and experiment settings are in `configs/experiment.yaml`; the CLI records the selected reranking parameters in each run's JSON metadata.

## Start with the notebooks

Run from the repository root so relative paths resolve correctly:

1. **`00_synthetic_smoke_test.ipynb`** — offline demonstration of every reranker, metrics, and darkgrid charts. Run this first; it requires no dataset or model download.
2. **`01_inquire_data_and_embeddings.ipynb`** — fetches a small slice of INQUIRE-Rerank and writes per-query embedding caches. Start with `MAX_QUERIES = 3` or `5`.
3. **`02_baselines_and_metrics.ipynb`** — loads the saved cache, compares methods, and writes per-query/summary metrics and figures.
4. **`03_adaptive_controller_and_tradeoffs.ipynb`** — fixed-weight MMR sweep, paired bootstrap comparison, and relevance–diversity plot.
5. **`04_fashionclip_transfer.ipynb`** — shows how to use FashionCLIP and the local manifest interface for a licensed fashion dataset.
6. **`05_algorithm_walkthrough.ipynb`** — exposes intermediate arrays and implementation source line by line; writes a small metric table, ranked candidates, metadata, and a darkgrid chart.

The code persists artifacts rather than relying only on notebook state. After a process interruption, existing `.npz` query caches can be reused. Each encoder/split cache also has an `index.json` that preserves dataset query order, so query-limited reruns select a stable prefix even after larger cache runs.

## INQUIRE-Rerank benchmark

- Dataset: [`evendrow/INQUIRE-Rerank`](https://huggingface.co/datasets/evendrow/INQUIRE-Rerank)
- Official evaluation repository: [`inquire-benchmark/INQUIRE`](https://github.com/inquire-benchmark/INQUIRE)
- Official homepage: [INQUIRE](https://inquire-benchmark.github.io/)
- Benchmark task: reorder a fixed set of top-100 images per text query; the official first-stage candidates originate from OpenCLIP ViT-H/14-378.
- The Hugging Face dataset card specifies **CC BY-NC 4.0**. Review its current terms before use, especially for commercial work. Do not commit downloaded images, dataset shards, or large embeddings to this repository.

Start with validation, use it for exploratory analysis and controller design, then evaluate the locked method on test once. The benchmark is natural-world / ecology-focused; FashionCLIP results on it should be labeled **out-of-domain transfer**, not evidence of in-domain fashion performance.

```bash
# Small, streamed first pass (may still fetch a remote parquet/image shard)
python scripts/run_inquire.py --split validation --max-queries 5 --backend open_clip

# Full validation split; then tune/inspect before using test
python scripts/run_inquire.py --split validation --max-queries 0 --backend open_clip --run-name validation_openclip_full

# Once the method and thresholds are locked, run the held-out split
python scripts/run_inquire.py --split test --max-queries 0 --backend open_clip --run-name test_openclip_locked
```

Run with FashionCLIP using `--backend fashionclip`. This still uses INQUIRE candidates and is therefore an out-of-domain diagnostic. For a fashion-specific test, prepare a CSV manifest and use notebook 04.

### Local manifest format for fashion/custom data

Prepare one row per query–candidate pair. All images must be from a dataset whose license permits your research use.

```csv
query,image_path,relevant,initial_score,image_id,category,color,style
"red midi dress",images/item_001.jpg,1,0.73,item_001,dress,red,midi
"red midi dress",images/item_002.jpg,0,0.41,item_002,top,red,casual
```

`image_path` can be absolute or relative to the manifest directory (or `image_root` argument). `relevant` is binary. `initial_score` should come from the initial retrieval stage to ensure the original-ranking control is meaningful. Optional columns after the required fields are preserved as metadata.

## Artifact contract

For a run named `validation_openclip_smoke`, the main outputs look like:

```text
artifacts/
├── cache/validation/openclip-ViT-B-32-openai/
│   ├── index.json
│   └── <query-id>.npz
└── experiments/validation_openclip_smoke/
    ├── metrics/
    │   ├── validation_openclip_smoke_per_query_metrics.csv
    │   ├── validation_openclip_smoke_summary.csv
    │   ├── validation_openclip_smoke_run_metadata.json
    │   └── validation_openclip_smoke_environment.json
    ├── rankings/validation_openclip_smoke_rankings.csv
    └── figures/
        ├── relevance_diversity_latency.png
        └── metric_comparison.png
```

Embedding caches and benchmark run outputs are git-ignored because they are reproducible, may be large, and can be subject to data terms. Cache files and metric artifacts are written via temporary files and atomic replacement to reduce the chance that an interrupted run leaves a truncated file that looks complete. Small synthetic smoke-test outputs are versioned to demonstrate the format.

## Methodology caveats

1. **Same shortlist:** all rerankers operate on the same pre-retrieved candidates.
2. **Encoder comparison:** `initial_clip` uses the benchmark-provided `clip_score`; `encoder_similarity` uses the selected model's query–image cosine. MMR/DPP/xQuAD use the selected model's text–image cosine plus image–image similarity. Report both controls so embedding-model changes are not confused with ranking-policy changes.
3. **Diversity:** ILD is embedding-space dispersion, not proof of useful semantic/aspect coverage. Taxonomy counts are descriptive and can reward wrong species; interpret alongside relevance.
4. **xQuAD:** the included cluster-based variant uses unsupervised KMeans pseudo-aspects. It is not the same as xQuAD using human-defined query subtopics.
5. **Adaptive controller:** parameters are a hypothesis. Select them on validation data and report one locked test result; do not tune on test labels.
6. **Statistical comparison:** report per-query paired bootstrap confidence intervals; do not infer significance from aggregate means alone.
7. **Latency:** report p50/p95 as well as mean, and measure end-to-end query latency in a separate production-like benchmark if needed.

## Reproducibility commands

```bash
pytest --cov=image_rerank
ruff check src tests scripts
python scripts/run_synthetic_demo.py
```

## Citation

If publishing experiments using INQUIRE, cite Vendrow et al., *INQUIRE: A Natural World Text-to-Image Retrieval Benchmark*, NeurIPS 2024. Refer to the official benchmark repository for the current citation text and evaluation details.
