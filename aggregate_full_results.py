r"""Aggregate the complete experiment campaign produced by ``run_all.ps1``.

The script scans ``metrics.json`` files below an output root, merges them with
``run_manifest.csv`` when available, computes multi-seed mean/std summaries,
and writes publication-ready comparison figures for every hyperparameter
sweep.  Raw experiment folders are never modified.

Typical use from the project root::

    .\.Mckean\Scripts\python.exe aggregate_full_results.py --root outputs/paper_full

Outputs are written below ``<root>/aggregate``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Metrics shown in the paper-level tables and sweep figures.
METRICS = [
    "mdn_val_nll",
    "w2_mdn_mean",
    "w2_mdn_std",
    "w2_floor_mean",
    "w2_floor_std",
    "reference_resolution",
    "rel_l2_full",
    "rel_l2_empirical",
    "rel_l2_oracle",
    "functional_reference_floor",
    "rel_l2_empirical_noise_adjusted",
    "rel_l2_mdn_noise_adjusted",
    "rel_l2_reference_noise_adjusted",
    "runtime_sec",
]

SWEEP_PLOT_METRICS = [
    ("w2_mdn_mean", r"test $W_2$ / sliced $W_2$"),
    ("mdn_val_nll", "best validation NLL"),
    ("rel_l2_full", r"test relative $L^2$"),
]

CONFIG_FIELDS = [
    "M",
    "N",
    "sig_level",
    "n_fourier",
    "J",
    "k_keep",
    "mdn_epochs",
    "psi_epochs",
    "hidden",
    "batch",
    "lr",
    "mdn_depth",
    "psi_depth",
    "S_quant",
    "k_x",
    "single_particle",
    "empirical_targets",
    "w2_metric",
]

GROUP_PARAMETER = {
    "sweep_J": "J",
    "sweep_fourier": "n_fourier",
    "sweep_lr": "lr",
    "sweep_mdn_depth": "mdn_depth",
    "sweep_psi_depth": "psi_depth",
    "sweep_sig_level": "sig_level",
    "sweep_mdn_depth_nonlinear": "mdn_depth",
    "sweep_psi_depth_nonlinear": "psi_depth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/paper_full"),
        help="campaign output root (default: outputs/paper_full)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="aggregation directory (default: <root>/aggregate)",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:  # pragma: no cover - defensive path
            print(f"[warning] cannot read {path}: {exc}", file=sys.stderr)
            return None
    except Exception as exc:
        print(f"[warning] cannot read {path}: {exc}", file=sys.stderr)
        return None
    if not isinstance(value, dict):
        print(f"[warning] expected a JSON object in {path}", file=sys.stderr)
        return None
    return value


def normalize_path(value: str | Path) -> str:
    """Normalize Windows/POSIX paths for manifest matching."""
    text = str(value).replace("\\", "/").rstrip("/")
    while "//" in text:
        text = text.replace("//", "/")
    return text.lower()


def read_manifest(path: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    if not path.exists():
        print(f"[warning] manifest not found: {path}; inferring metadata from folders")
        return [], {}

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(dict(row))

    # A resumed campaign can append several rows for the same output directory.
    # The last row is the latest status and therefore the one to retain.
    by_outdir: dict[str, dict[str, str]] = {}
    for row in rows:
        outdir = row.get("outdir", "")
        if outdir:
            by_outdir[normalize_path(outdir)] = row
    return rows, by_outdir


def match_manifest_row(
    run_dir: Path,
    root: Path,
    by_outdir: dict[str, dict[str, str]],
) -> dict[str, str] | None:
    candidates = {
        normalize_path(run_dir),
        normalize_path(run_dir.resolve()),
    }
    try:
        relative = run_dir.relative_to(root)
        candidates.add(normalize_path(root / relative))
        candidates.add(normalize_path(relative))
    except ValueError:
        pass

    for candidate in candidates:
        if candidate in by_outdir:
            return by_outdir[candidate]

    # The manifest normally stores a project-relative path such as
    # outputs/paper_full/..., whereas ``root`` may be absolute.
    normalized_run = normalize_path(run_dir)
    for key, row in by_outdir.items():
        if normalized_run.endswith(key) or key.endswith(normalized_run):
            return row
    return None


def parse_seed(text: str) -> int | None:
    match = re.search(r"seed[_-]?(\d+)$", text)
    return int(match.group(1)) if match else None


def infer_metadata(run_dir: Path, root: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    try:
        parts = list(run_dir.relative_to(root).parts)
    except ValueError:
        parts = list(run_dir.parts)

    config = metrics.get("config") if isinstance(metrics.get("config"), dict) else {}
    group = parts[0] if parts else "unknown"
    benchmark = str(metrics.get("benchmark") or (parts[1] if len(parts) > 1 else ""))
    functional = str(config.get("functional", ""))
    seed = config.get("seed")
    if seed is None:
        for part in reversed(parts):
            parsed = parse_seed(part)
            if parsed is not None:
                seed = parsed
                break

    parameter = GROUP_PARAMETER.get(group, "")
    value: str | float | int = ""
    if parameter:
        prefix = f"{parameter}_"
        for part in parts:
            if part.startswith(prefix):
                value = part[len(prefix) :]
                break

    return {
        "group": group,
        "benchmark": benchmark,
        "functional": functional,
        "parameter": parameter,
        "value": value,
        "seed": seed,
        "outdir": str(run_dir),
        "status": "found",
        "runtime_minutes_manifest": None,
        "message": "",
    }


def coerce_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def collect_runs(root: Path, manifest_index: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        if "aggregate" in metrics_path.relative_to(root).parts:
            continue
        metrics = read_json(metrics_path)
        if metrics is None:
            continue

        run_dir = metrics_path.parent
        metadata = infer_metadata(run_dir, root, metrics)
        manifest = match_manifest_row(run_dir, root, manifest_index)
        if manifest:
            for key in [
                "group",
                "benchmark",
                "functional",
                "parameter",
                "value",
                "status",
                "message",
            ]:
                if manifest.get(key, "") != "":
                    metadata[key] = manifest[key]
            if manifest.get("seed", "") != "":
                try:
                    metadata["seed"] = int(manifest["seed"])
                except ValueError:
                    metadata["seed"] = manifest["seed"]
            metadata["runtime_minutes_manifest"] = coerce_number(
                manifest.get("runtime_minutes")
            )

        config = metrics.get("config") if isinstance(metrics.get("config"), dict) else {}
        record: dict[str, Any] = {
            **metadata,
            "metrics_path": str(metrics_path),
            "mdn_beats_floor": metrics.get("mdn_beats_floor"),
            "reference_baseline_label": metrics.get("reference_baseline_label"),
            "n_ref_per_replicate": metrics.get("n_ref_per_replicate"),
            "n_ref_pooled": metrics.get("n_ref_pooled"),
            "sweep_kkeep": metrics.get("sweep_kkeep", {}),
        }
        for field in CONFIG_FIELDS:
            record[field] = config.get(field)
        for metric in METRICS:
            record[metric] = metrics.get(metric)
        runs.append(record)
    return runs


def mean_std(values: Iterable[Any]) -> tuple[float | None, float | None, int]:
    clean = [coerce_number(v) for v in values]
    arr = np.asarray([v for v in clean if v is not None], dtype=float)
    if arr.size == 0:
        return None, None, 0
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), std, int(arr.size)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            encoded = {
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list, tuple))
                else value
                for key, value in row.items()
            }
            writer.writerow(encoded)


def summarize_groups(
    runs: list[dict[str, Any]],
    keys: list[str],
    metrics: list[str] = METRICS,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[tuple(run.get(key, "") for key in keys)].append(run)

    summary: list[dict[str, Any]] = []
    for group_key, group_runs in grouped.items():
        row = {key: value for key, value in zip(keys, group_key)}
        seeds = sorted(
            {int(s) for s in (r.get("seed") for r in group_runs) if str(s).isdigit()}
        )
        row["n_runs"] = len(group_runs)
        row["seeds"] = ",".join(str(s) for s in seeds)
        for metric in metrics:
            mean, std, count = mean_std(r.get(metric) for r in group_runs)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std_across_seeds"] = std
            row[f"{metric}_n"] = count
        summary.append(row)
    return summary


def numeric_sort_key(value: Any) -> tuple[int, float | str]:
    number = coerce_number(value)
    return (0, number) if number is not None else (1, str(value))


def plot_series(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    mean_key: str,
    std_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output: Path,
) -> bool:
    filtered = [r for r in rows if coerce_number(r.get(mean_key)) is not None]
    if not filtered:
        return False
    filtered.sort(key=lambda r: numeric_sort_key(r.get(x_key)))

    x_numbers = [coerce_number(r.get(x_key)) for r in filtered]
    numeric_x = all(v is not None for v in x_numbers)
    if numeric_x:
        x: list[float] | np.ndarray = np.asarray(x_numbers, dtype=float)
        tick_positions = None
        tick_labels = None
    else:
        x = np.arange(len(filtered), dtype=float)
        tick_positions = x
        tick_labels = [str(r.get(x_key)) for r in filtered]

    y = np.asarray([float(r[mean_key]) for r in filtered], dtype=float)
    yerr = np.asarray(
        [coerce_number(r.get(std_key)) or 0.0 for r in filtered], dtype=float
    )

    plt.figure(figsize=(6.2, 4.2))
    plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4)
    if tick_positions is not None:
        plt.xticks(tick_positions, tick_labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=180)
    plt.close()
    return True


def make_sweep_plots(sweep_summary: list[dict[str, Any]], plot_dir: Path) -> int:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sweep_summary:
        grouped[(str(row["group"]), str(row["benchmark"]), str(row["parameter"]))].append(row)

    created = 0
    for (group, benchmark, parameter), rows in grouped.items():
        for metric, ylabel in SWEEP_PLOT_METRICS:
            output = plot_dir / f"{group}__{benchmark}__{parameter}__{metric}.png"
            ok = plot_series(
                rows,
                x_key="value",
                mean_key=f"{metric}_mean",
                std_key=f"{metric}_std_across_seeds",
                xlabel=parameter,
                ylabel=ylabel,
                title=f"{benchmark}: effect of {parameter}",
                output=output,
            )
            created += int(ok)
    return created


def make_headline_plots(headline_summary: list[dict[str, Any]], plot_dir: Path) -> int:
    created = 0
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in headline_summary:
        by_group[str(row["group"])].append(row)

    for group, rows in by_group.items():
        for metric, ylabel in [
            ("w2_mdn_mean", r"test $W_2$ / sliced $W_2$"),
            ("rel_l2_full", r"test relative $L^2$"),
        ]:
            rows_sorted = sorted(rows, key=lambda r: str(r.get("benchmark")))
            filtered = [
                r
                for r in rows_sorted
                if coerce_number(r.get(f"{metric}_mean")) is not None
            ]
            if not filtered:
                continue
            x = np.arange(len(filtered), dtype=float)
            y = np.asarray([r[f"{metric}_mean"] for r in filtered], dtype=float)
            yerr = np.asarray(
                [coerce_number(r.get(f"{metric}_std_across_seeds")) or 0.0 for r in filtered],
                dtype=float,
            )
            plt.figure(figsize=(7.2, 4.4))
            plt.errorbar(x, y, yerr=yerr, marker="o", linestyle="none", capsize=4)
            plt.xticks(x, [str(r.get("benchmark")) for r in filtered], rotation=25, ha="right")
            plt.ylabel(ylabel)
            plt.title(group.replace("_", " "))
            plt.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            plt.savefig(plot_dir / f"{group}__{metric}.png", dpi=180)
            plt.close()
            created += 1

        # Functional baseline comparison, when all three quantities exist.
        baseline_rows = [
            r
            for r in rows
            if any(
                coerce_number(r.get(f"{m}_mean")) is not None
                for m in ["rel_l2_empirical", "rel_l2_full", "rel_l2_oracle"]
            )
        ]
        if baseline_rows:
            baseline_rows.sort(key=lambda r: str(r.get("benchmark")))
            labels = [str(r.get("benchmark")) for r in baseline_rows]
            x = np.arange(len(labels), dtype=float)
            width = 0.24
            plt.figure(figsize=(8.0, 4.6))
            for offset, metric, label in [
                (-width, "rel_l2_empirical", "empirical plug-in"),
                (0.0, "rel_l2_full", "MDN pipeline"),
                (width, "rel_l2_oracle", "reference integrals"),
            ]:
                values = [coerce_number(r.get(f"{metric}_mean")) for r in baseline_rows]
                heights = [np.nan if v is None else v for v in values]
                plt.bar(x + offset, heights, width=width, label=label)
            plt.xticks(x, labels, rotation=25, ha="right")
            plt.ylabel(r"test relative $L^2$")
            plt.title(f"{group.replace('_', ' ')}: functional baselines")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / f"{group}__functional_baselines.png", dpi=180)
            plt.close()
            created += 1
    return created


def collect_kkeep(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        values = run.get("sweep_kkeep")
        if not isinstance(values, dict) or not values:
            continue
        for k, w2 in values.items():
            rows.append(
                {
                    "group": run.get("group"),
                    "benchmark": run.get("benchmark"),
                    "seed": run.get("seed"),
                    "K_keep": coerce_number(k),
                    "w2_mdn_mean": coerce_number(w2),
                    "outdir": run.get("outdir"),
                }
            )
    return rows


def make_kkeep_plots(rows: list[dict[str, Any]], plot_dir: Path) -> int:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("benchmark")), str(row.get("seed")))].append(row)
    created = 0
    for (benchmark, seed), group_rows in grouped.items():
        group_rows.sort(key=lambda r: numeric_sort_key(r.get("K_keep")))
        x = [r["K_keep"] for r in group_rows if r["K_keep"] is not None]
        y = [r["w2_mdn_mean"] for r in group_rows if r["K_keep"] is not None]
        if not x or any(v is None for v in y):
            continue
        plt.figure(figsize=(6.0, 4.1))
        plt.plot(x, y, marker="o")
        plt.xlabel(r"$K_{\mathrm{keep}}$")
        plt.ylabel(r"test $W_2$")
        plt.title(f"{benchmark}: effect of K_keep (seed {seed})")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(plot_dir / f"kkeep__{benchmark}__seed_{seed}.png", dpi=180)
        plt.close()
        created += 1
    return created


def collect_ito(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("ito_strat_check.json")):
        if "aggregate" in path.relative_to(root).parts:
            continue
        data = read_json(path)
        if data is None:
            continue
        row = {"path": str(path), **data}
        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    outdir = (args.outdir.resolve() if args.outdir else root / "aggregate")
    plot_dir = outdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        print(f"[error] output root does not exist: {root}", file=sys.stderr)
        return 2

    manifest_rows, manifest_index = read_manifest(root / "run_manifest.csv")
    runs = collect_runs(root, manifest_index)
    if not runs:
        print(f"[error] no metrics.json found below {root}", file=sys.stderr)
        return 2

    run_fields = [
        "group",
        "benchmark",
        "functional",
        "parameter",
        "value",
        "seed",
        "status",
        "outdir",
        "metrics_path",
        "runtime_minutes_manifest",
        "message",
        *CONFIG_FIELDS,
        *METRICS,
        "mdn_beats_floor",
        "reference_baseline_label",
        "n_ref_per_replicate",
        "n_ref_pooled",
        "sweep_kkeep",
    ]
    write_csv(outdir / "per_run_metrics.csv", runs, run_fields)

    sweep_runs = [r for r in runs if str(r.get("group", "")).startswith("sweep_")]
    sweep_summary = summarize_groups(
        sweep_runs, ["group", "benchmark", "functional", "parameter", "value"]
    )
    sweep_summary.sort(
        key=lambda r: (
            str(r.get("group")),
            str(r.get("benchmark")),
            numeric_sort_key(r.get("value")),
        )
    )
    write_csv(outdir / "sweep_summary.csv", sweep_summary)

    headline_runs = [
        r
        for r in runs
        if not str(r.get("group", "")).startswith("sweep_")
        and str(r.get("group", "")) not in {"kkeep", "ito_strat"}
    ]
    headline_summary = summarize_groups(
        headline_runs, ["group", "benchmark", "functional"]
    )
    headline_summary.sort(
        key=lambda r: (str(r.get("group")), str(r.get("benchmark")))
    )
    write_csv(outdir / "headline_summary.csv", headline_summary)
    # Compatibility name used in earlier instructions.
    write_csv(outdir / "summary.csv", headline_summary)

    kkeep_rows = collect_kkeep(runs)
    write_csv(outdir / "kkeep_summary.csv", kkeep_rows)

    ito_rows = collect_ito(root)
    write_csv(outdir / "ito_strat_summary.csv", ito_rows)

    plots_created = 0
    plots_created += make_sweep_plots(sweep_summary, plot_dir)
    plots_created += make_headline_plots(headline_summary, plot_dir)
    plots_created += make_kkeep_plots(kkeep_rows, plot_dir)

    status_counts: dict[str, int] = defaultdict(int)
    for row in manifest_rows:
        status_counts[row.get("status", "unknown")] += 1

    report = {
        "root": str(root),
        "n_metrics_files": len(runs),
        "n_sweep_runs": len(sweep_runs),
        "n_sweep_points": len(sweep_summary),
        "n_headline_groups": len(headline_summary),
        "n_kkeep_points": len(kkeep_rows),
        "n_ito_strat_checks": len(ito_rows),
        "plots_created": plots_created,
        "manifest_status_counts": dict(status_counts),
        "headline_summary": headline_summary,
        "sweep_summary": sweep_summary,
        "kkeep_summary": kkeep_rows,
        "ito_strat_summary": ito_rows,
    }
    (outdir / "summary.json").write_text(
        json.dumps(json_safe(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n[aggregate] completed successfully")
    print(f"  metrics files : {len(runs)}")
    print(f"  sweep points  : {len(sweep_summary)}")
    print(f"  plots created : {plots_created}")
    print(f"  output        : {outdir}")
    print(f"  main table    : {outdir / 'headline_summary.csv'}")
    print(f"  sweep table   : {outdir / 'sweep_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
