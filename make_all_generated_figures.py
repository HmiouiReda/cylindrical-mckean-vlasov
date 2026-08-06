"""Generate all manuscript figures that were created from the experiment outputs.

This script recreates the five generated figures:

    Figure 1  -> training_convergence_ou_gmm.png
    Figure 2  -> law_error_across_benchmarks.png
    Figure 3  -> functional_error_across_benchmarks.png
    Figure 4  -> signature_depth_double_well.png
    Figure 5  -> fourier_sensitivity_non_gaussian.png

Run from the repository root, for example:

    python make_all_generated_figures.py

By default the script expects:
    outputs/paper_full/aggregate/headline_summary.csv
    outputs/paper_full/aggregate/sweep_summary.csv

and it writes every figure to:
    outputs/assets/

If the aggregate CSV files do not exist yet, the script first runs

    python aggregate_full_results.py --root outputs/paper_full

Figure 1 is built from a training-history JSON file produced by an OU-GMM run.
The script tries to locate it automatically under outputs/. If auto-discovery
fails, pass it explicitly with --training-history.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BENCHMARK_ORDER = [
    "ou",
    "ou_uniform",
    "ou_gmm",
    "double_well",
    "multiplicative",
    "ou2d",
]

BENCHMARK_LABELS = {
    "ou": "OU-Gaussian",
    "ou_uniform": "OU-uniform",
    "ou_gmm": "OU-GMM",
    "double_well": "Double well",
    "multiplicative": "Multiplicative",
    "ou2d": "OU-2D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/paper_full"),
        help="Experiment campaign root for the paper runs (default: outputs/paper_full).",
    )
    parser.add_argument(
        "--aggregate-dir",
        type=Path,
        default=None,
        help="Directory containing headline_summary.csv and sweep_summary.csv. "
        "Default: <root>/aggregate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/assets"),
        help="Directory where all figures are written (default: outputs/assets).",
    )
    parser.add_argument(
        "--training-history",
        type=Path,
        default=None,
        help="Path to the OU-GMM training_history.json file used for Figure 1. "
        "If omitted, the script tries to find it automatically under outputs/.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="PNG resolution (default: 220).",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in {source}: {', '.join(missing)}. "
            "Re-run aggregate_full_results.py with the current main branch."
        )


def ensure_aggregate_files(root: Path, aggregate_dir: Path) -> tuple[Path, Path]:
    headline_csv = aggregate_dir / "headline_summary.csv"
    sweep_csv = aggregate_dir / "sweep_summary.csv"

    if headline_csv.exists() and sweep_csv.exists():
        return headline_csv, sweep_csv

    aggregator = Path("aggregate_full_results.py")
    if not aggregator.exists():
        raise FileNotFoundError(
            f"Could not find {headline_csv} and {sweep_csv}, and {aggregator} is absent. "
            "Run this script from the repository root."
        )

    print("Aggregate CSV files not found; running aggregate_full_results.py ...")
    subprocess.run([sys.executable, str(aggregator), "--root", str(root)], check=True)

    if not headline_csv.exists() or not sweep_csv.exists():
        raise FileNotFoundError(
            "Aggregation completed but the expected CSV files were not created: "
            f"{headline_csv}, {sweep_csv}."
        )

    return headline_csv, sweep_csv


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Created: {path}")


# -----------------------------------------------------------------------------
# Figure 1 utilities
# -----------------------------------------------------------------------------


def is_numeric_sequence(obj: Any) -> bool:
    if not isinstance(obj, (list, tuple)) or len(obj) == 0:
        return False
    try:
        [float(x) for x in obj]
    except (TypeError, ValueError):
        return False
    return True


def series_from_list_of_dicts(items: list[dict[str, Any]], candidates: list[str]) -> np.ndarray | None:
    for key in candidates:
        if all(isinstance(item, dict) and key in item for item in items):
            try:
                return np.asarray([float(item[key]) for item in items], dtype=float)
            except (TypeError, ValueError):
                pass
    return None


def extract_series(block: Any, candidates: list[str]) -> np.ndarray | None:
    """Try hard to extract a numeric series from several plausible JSON schemas."""
    if isinstance(block, dict):
        for key in candidates:
            if key in block and is_numeric_sequence(block[key]):
                return np.asarray(block[key], dtype=float)

        # Common nested names.
        for nested_name in ["history", "metrics", "curves", "logs", "data"]:
            if nested_name in block:
                result = extract_series(block[nested_name], candidates)
                if result is not None:
                    return result

        # Search recursively over all values.
        for value in block.values():
            result = extract_series(value, candidates)
            if result is not None:
                return result

    elif isinstance(block, list):
        if is_numeric_sequence(block):
            return np.asarray(block, dtype=float)
        if len(block) > 0 and all(isinstance(item, dict) for item in block):
            result = series_from_list_of_dicts(block, candidates)
            if result is not None:
                return result
            for item in block:
                result = extract_series(item, candidates)
                if result is not None:
                    return result

    return None


def choose_epochs(*series: np.ndarray | None) -> np.ndarray:
    for series_item in series:
        if series_item is not None and len(series_item) > 0:
            return np.arange(1, len(series_item) + 1, dtype=int)
    raise ValueError("Could not determine the epoch range from the training history.")


def find_training_history(outputs_root: Path) -> Path:
    candidates = list(outputs_root.rglob("training_history.json"))
    candidates += list(outputs_root.rglob("*training*history*.json"))

    candidates = [path for path in candidates if "ou_gmm" in str(path).lower()]
    if not candidates:
        raise FileNotFoundError(
            "Could not auto-discover an OU-GMM training-history JSON file under outputs/. "
            "Please pass it explicitly with --training-history."
        )

    def sort_key(path: Path) -> tuple[int, int, int, str]:
        text = str(path).lower()
        return (
            0 if "seed_0" in text else 1,
            0 if "ou_gmm" in text else 1,
            len(path.parts),
            text,
        )

    return sorted(candidates, key=sort_key)[0]




# -----------------------------------------------------------------------------
# Figure 2
# -----------------------------------------------------------------------------


def make_law_error_figure(headline: pd.DataFrame, source: Path, output_path: Path, dpi: int) -> None:
    required = [
        "group",
        "benchmark",
        "functional",
        "w2_mdn_mean_mean",
        "w2_mdn_mean_std_across_seeds",
        "w2_floor_mean_mean",
        "w2_floor_mean_std_across_seeds",
    ]
    require_columns(headline, required, source)

    data = headline.loc[
        (headline["group"] == "headline_moments")
        & (headline["functional"] == "moments")
        & (headline["benchmark"].isin(BENCHMARK_ORDER))
    ].copy()

    data["benchmark"] = pd.Categorical(data["benchmark"], categories=BENCHMARK_ORDER, ordered=True)
    data = data.sort_values("benchmark")

    labels = [BENCHMARK_LABELS[str(value)] for value in data["benchmark"]]
    x = np.arange(len(labels), dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(
        x - width / 2,
        data["w2_mdn_mean_mean"].to_numpy(float),
        width,
        yerr=data["w2_mdn_mean_std_across_seeds"].to_numpy(float),
        capsize=4,
        label="MDN",
    )
    ax.bar(
        x + width / 2,
        data["w2_floor_mean_mean"].to_numpy(float),
        width,
        yerr=data["w2_floor_mean_std_across_seeds"].to_numpy(float),
        capsize=4,
        label="Empirical cloud (N=200)",
    )

    ax.set_ylabel("Mean test discrepancy")
    ax.set_xticks(x, labels, rotation=22, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output_path, dpi)


# -----------------------------------------------------------------------------
# Figure 3
# -----------------------------------------------------------------------------


def make_functional_error_figure(headline: pd.DataFrame, source: Path, output_path: Path, dpi: int) -> None:
    required = [
        "group",
        "benchmark",
        "functional",
        "rel_l2_empirical_mean",
        "rel_l2_empirical_std_across_seeds",
        "rel_l2_full_mean",
        "rel_l2_full_std_across_seeds",
        "rel_l2_oracle_mean",
        "rel_l2_oracle_std_across_seeds",
    ]
    require_columns(headline, required, source)

    data = headline.loc[
        (headline["group"] == "headline_moments")
        & (headline["functional"] == "moments")
        & (headline["benchmark"].isin(BENCHMARK_ORDER))
    ].copy()

    data["benchmark"] = pd.Categorical(data["benchmark"], categories=BENCHMARK_ORDER, ordered=True)
    data = data.sort_values("benchmark")

    labels = [BENCHMARK_LABELS[str(value)] for value in data["benchmark"]]
    x = np.arange(len(labels), dtype=float)
    width = 0.25

    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    ax.bar(
    x - width,
    100.0 * data["rel_l2_empirical_mean"].to_numpy(float),
    width,
    yerr=100.0 * data[
        "rel_l2_empirical_std_across_seeds"
    ].to_numpy(float),
    capsize=4,
    label="Empirical",
)

    ax.bar(
        x,
        100.0 * data["rel_l2_full_mean"].to_numpy(float),
        width,
        yerr=100.0 * data[
            "rel_l2_full_std_across_seeds"
        ].to_numpy(float),
        capsize=4,
        label="MDN",
    )

    ax.bar(
        x + width,
        100.0 * data["rel_l2_oracle_mean"].to_numpy(float),
        width,
        yerr=100.0 * data[
            "rel_l2_oracle_std_across_seeds"
        ].to_numpy(float),
        capsize=4,
        label="Exact / indep. reference",
    )

    ax.set_ylabel(r"Relative $L^2$ error (%)")
    ax.set_xticks(x, labels, rotation=22, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output_path, dpi)


# -----------------------------------------------------------------------------
# Figure 4
# -----------------------------------------------------------------------------


def make_signature_depth_figure(
    sweeps: pd.DataFrame,
    source: Path,
    output_path: Path,
    dpi: int,
) -> None:
    required = [
        "group",
        "benchmark",
        "parameter",
        "value",
        "w2_mdn_mean_mean",
        "rel_l2_full_mean",
    ]
    require_columns(sweeps, required, source)

    data = sweeps.loc[
        (sweeps["group"] == "sweep_sig_level")
        & (sweeps["benchmark"] == "double_well")
        & (sweeps["parameter"] == "sig_level")
    ].copy()

    data = data.sort_values("value")
    levels = data["value"].to_numpy(float)

    fig, ax_w2 = plt.subplots(figsize=(6.6, 4.4))
    ax_l2 = ax_w2.twinx()

    line_w2 = ax_w2.plot(
        levels,
        data["w2_mdn_mean_mean"].to_numpy(float),
        marker="o",
        label=r"Mean $W_2$",
    )

    line_l2 = ax_l2.plot(
        levels,
        100.0 * data["rel_l2_full_mean"].to_numpy(float),
        marker="s",
        linestyle="--",
        label=r"Relative $L^2$ error",
    )

    ax_w2.set_xlabel("Signature truncation level")
    ax_w2.set_ylabel(r"Mean test $W_2$")
    ax_l2.set_ylabel(r"Relative $L^2$ error (%)")

    ax_w2.set_xticks(levels.astype(int))
    ax_w2.grid(alpha=0.3)

    lines = line_w2 + line_l2
    labels = [line.get_label() for line in lines]
    ax_w2.legend(lines, labels, loc="best")

    fig.tight_layout()
    save_figure(fig, output_path, dpi)


# -----------------------------------------------------------------------------
# Figure 5
# -----------------------------------------------------------------------------


def make_fourier_sensitivity_figure(sweeps: pd.DataFrame, source: Path, output_path: Path, dpi: int) -> None:
    required = [
        "group",
        "benchmark",
        "parameter",
        "value",
        "w2_mdn_mean_mean",
    ]
    require_columns(sweeps, required, source)

    data = sweeps.loc[
        (sweeps["group"] == "sweep_fourier")
        & (sweeps["parameter"] == "n_fourier")
        & (sweeps["benchmark"].isin(["ou_gmm", "ou_uniform"]))
    ].copy()

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for benchmark, label in [("ou_gmm", "OU-GMM"), ("ou_uniform", "OU-uniform")]:
        subset = data.loc[data["benchmark"] == benchmark].sort_values("value")
        ax.plot(
            subset["value"].to_numpy(float),
            subset["w2_mdn_mean_mean"].to_numpy(float),
            marker="o",
            label=label,
        )

    ax.set_xlabel(r"Number of positive Fourier frequencies $P$")
    ax.set_ylabel(r"Mean test $W_2$")
    ax.set_xticks(sorted(data["value"].astype(int).unique()))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output_path, dpi)


def main() -> None:
    args = parse_args()
    aggregate_dir = args.aggregate_dir or args.root / "aggregate"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    headline_csv, sweep_csv = ensure_aggregate_files(args.root, aggregate_dir)
    headline = pd.read_csv(headline_csv)
    sweeps = pd.read_csv(sweep_csv)

    training_history = args.training_history
    if training_history is None:
        training_history = find_training_history(Path("outputs"))
        print(f"Auto-discovered training history: {training_history}")
    elif not training_history.exists():
        raise FileNotFoundError(f"Training-history file not found: {training_history}")

    make_law_error_figure(
        headline,
        headline_csv,
        output_dir / "law_error_across_benchmarks.png",
        args.dpi,
    )
    make_functional_error_figure(
        headline,
        headline_csv,
        output_dir / "functional_error_across_benchmarks.png",
        args.dpi,
    )
    make_signature_depth_figure(
        sweeps,
        sweep_csv,
        output_dir / "signature_depth_double_well.png",
        args.dpi,
    )
    make_fourier_sensitivity_figure(
        sweeps,
        sweep_csv,
        output_dir / "fourier_sensitivity_non_gaussian.png",
        args.dpi,
    )


if __name__ == "__main__":
    main()
