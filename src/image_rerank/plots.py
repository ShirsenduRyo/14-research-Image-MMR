"""Seaborn darkgrid plots used by the notebooks and CLI."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .artifacts import ensure_dir


def set_plot_style() -> None:
    """Set the project's required Seaborn darkgrid theme."""
    sns.set_theme(style="darkgrid", context="notebook", palette="deep")
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 180, "axes.titleweight": "bold"})


def _metric_column(columns: pd.Index, prefix: str, suffix: str = "") -> str:
    """Find one metric column by prefix/suffix, raising a clear error if absent."""
    matches = [str(column) for column in columns if str(column).startswith(prefix) and str(column).endswith(suffix)]
    if not matches:
        raise ValueError(f"No metric column matches prefix={prefix!r}, suffix={suffix!r}")
    # Prefer the most common smaller-k metric if multiple fields match by accident.
    return sorted(matches, key=lambda value: (int(value.split("@")[1].split("_")[0]) if "@" in value and value.split("@")[1].split("_")[0].isdigit() else 10**9, value))[0]


def plot_tradeoff(metrics: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot relevance against diversity and show reranking latency separately."""
    set_plot_style()
    ap_column = _metric_column(metrics.columns, "ap@")
    ild_column = _metric_column(metrics.columns, "ild@")
    summary = metrics.groupby("method", as_index=False).agg(
        relevance=(ap_column, "mean"), diversity=(ild_column, "mean"),
        latency_ms=("rerank_latency_ms", "mean"),
    )
    fig, (ax, ax_latency) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )
    method_palette = dict(zip(summary["method"], sns.color_palette(n_colors=len(summary))))
    sns.scatterplot(data=summary, x="diversity", y="relevance", hue="method", palette=method_palette, s=130, ax=ax)
    ax.set_title("Relevance–diversity trade-off")
    ax.set_xlabel(f"Intra-list diversity {ild_column.upper()} (higher = more dispersed)")
    ax.set_ylabel(f"{ap_column.upper()} (higher = better relevance)")
    ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    latency = summary.sort_values("latency_ms", ascending=True)
    sns.barplot(data=latency, x="latency_ms", y="method", hue="method", palette=method_palette, legend=False, ax=ax_latency)
    positive_latencies = latency.loc[latency["latency_ms"] > 0, "latency_ms"]
    if len(positive_latencies) and positive_latencies.max() / positive_latencies.min() > 20:
        ax_latency.set_xscale("log")
        ax_latency.set_xlabel("Mean reranking latency (ms, log scale)")
    else:
        ax_latency.set_xlabel("Mean reranking latency (ms)")
    ax_latency.set_ylabel("")
    ax_latency.set_title("Ranking stage only")
    fig.suptitle("Quality and cost on the same candidate shortlists", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = Path(output_path)
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_metric_bars(summary: pd.DataFrame, output_path: str | Path) -> Path:
    """Show available relevance/diversity metrics with query-level SD error bars."""
    set_plot_style()
    metric_specs = []
    for prefix, label in (
        ("ap@", "Average precision"),
        ("ndcg@", "nDCG"),
        ("recall@", "Recall"),
        ("ild@", "Intra-list diversity"),
    ):
        mean_column = _metric_column(summary.columns, prefix, "_mean") if any(
            str(column).startswith(prefix) and str(column).endswith("_mean") for column in summary.columns
        ) else None
        if mean_column:
            metric_name = mean_column.removesuffix("_mean")
            metric_specs.append((mean_column, f"{metric_name.upper()} — {label}"))
    if not metric_specs:
        raise ValueError(f"No expected mean metric columns found. Available: {summary.columns.tolist()}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes_flat = axes.flatten()
    method_palette = dict(zip(summary["method"], sns.color_palette(n_colors=len(summary))))
    for ax, (mean_col, title) in zip(axes_flat, metric_specs):
        sns.barplot(data=summary, x="method", y=mean_col, hue="method", palette=method_palette, legend=False, errorbar=None, ax=ax)
        std_col = mean_col.removesuffix("_mean") + "_std"
        if std_col in summary.columns:
            x_positions = range(len(summary))
            ax.errorbar(
                x=list(x_positions), y=summary[mean_col].to_numpy(),
                yerr=summary[std_col].fillna(0).to_numpy(),
                fmt="none", ecolor="black", elinewidth=1, capsize=3,
            )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Mean across queries")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
    for ax in axes_flat[len(metric_specs):]:
        ax.set_visible(False)
    fig.suptitle("Relevance and diversity metrics by method (error bars = query SD when available)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = Path(output_path)
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_lambda_sweep(sweep: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot relevance and diversity by fixed MMR relevance weight."""
    set_plot_style()
    ap_column = _metric_column(sweep.columns, "ap@")
    ild_column = _metric_column(sweep.columns, "ild@")
    summary = sweep.groupby("lambda_relevance", as_index=False).agg(
        ap=(ap_column, "mean"), ild=(ild_column, "mean"), latency_ms=("rerank_latency_ms", "mean")
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=summary, x="lambda_relevance", y="ap", marker="o", label=ap_column.upper(), ax=ax)
    ax.set_ylabel(f"Mean {ap_column.upper()}")
    ax.set_xlabel("MMR relevance weight λ")
    ax.set_title("Fixed-weight MMR: relevance as the diversity pressure changes")
    ax2 = ax.twinx()
    sns.lineplot(data=summary, x="lambda_relevance", y="ild", marker="s", label=ild_column.upper(), color="tab:orange", ax=ax2)
    ax2.set_ylabel(f"Mean {ild_column.upper()}")
    line1 = ax.lines[0]
    line2 = ax2.lines[0]
    ax.legend([line1, line2], [f"Mean {ap_column.upper()}", f"Mean {ild_column.upper()}"], loc="best")
    fig.tight_layout()
    path = Path(output_path)
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

