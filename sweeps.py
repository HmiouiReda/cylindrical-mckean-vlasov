"""One-factor-at-a-time hyper-parameter sweeps.

Examples
--------
python sweeps.py --benchmark ou_gmm --parameter J --values 1,2,3,5,8
python sweeps.py --benchmark ou_uniform --parameter n_fourier --values 1,2,4,6,8
python sweeps.py --benchmark ou_gmm --parameter mdn_depth --values 1,2,3,4
python sweeps.py --benchmark ou_gmm --parameter psi_depth --values 1,2,3,4
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


CLI_FLAG = {
    "J": "--J",
    "n_fourier": "--n-fourier",
    "mdn_depth": "--mdn-depth",
    "psi_depth": "--psi-depth",
    "lr": "--lr",
}


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument(
        "--parameter",
        required=True,
        choices=sorted(CLI_FLAG),
    )
    parser.add_argument("--values", required=True, help="comma-separated values")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--functional",
        default="moments",
        choices=["moments", "quantile"],
    )
    parser.add_argument(
        "--outdir",
        default="",
        help="default: outputs/sweeps/<benchmark>/<parameter>",
    )
    return parser.parse_args()


def main():
    args = get_args()
    values = [v.strip() for v in args.values.split(",") if v.strip()]

    if not values:
        raise ValueError("At least one sweep value is required.")

    root = (
        Path(args.outdir)
        if args.outdir
        else Path("outputs") / "sweeps" / args.benchmark / args.parameter
    )
    root.mkdir(parents=True, exist_ok=True)

    rows = []

    for value in values:
        run_dir = root / f"{args.parameter}_{value}"

        command = [
            sys.executable,
            "main.py",
            "--benchmark",
            args.benchmark,
            "--seed",
            str(args.seed),
            "--functional",
            args.functional,
            "--baselines",
            "none",
            "--outdir",
            str(run_dir),
            CLI_FLAG[args.parameter],
            value,
        ]

        if args.quick:
            command.append("--quick")

        print("\n[sweep]", " ".join(command))
        subprocess.run(command, check=True)

        metrics_path = run_dir / "metrics.json"

        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

        metrics = json.loads(metrics_path.read_text())

        rows.append({
            "parameter": args.parameter,
            "value": float(value),
            "mdn_val_nll": metrics["mdn_val_nll"],
            "w2_mdn_mean": metrics["w2_mdn_mean"],
            "rel_l2_full": metrics["rel_l2_full"],
            "runtime_sec": metrics["runtime_sec"],
            "run_dir": str(run_dir),
        })

    rows.sort(key=lambda row: row["value"])

    (root / "summary.json").write_text(json.dumps(rows, indent=2))

    with open(root / "summary.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    x_values = [row["value"] for row in rows]

    metrics_to_plot = [
        ("w2_mdn_mean", r"test $W_2$ / sliced $W_2$"),
        ("mdn_val_nll", "best validation NLL"),
        ("rel_l2_full", r"test relative $L^2$"),
    ]

    for metric, ylabel in metrics_to_plot:
        plt.figure(figsize=(5.5, 3.8))
        plt.plot(
            x_values,
            [row[metric] for row in rows],
            marker="o",
        )
        plt.xlabel(args.parameter)
        plt.ylabel(ylabel)
        plt.title(f"{args.benchmark}: effect of {args.parameter}")
        plt.tight_layout()
        plt.savefig(
            root / f"sweep_{args.parameter}_{metric}.png",
            dpi=140,
        )
        plt.close()

    print(f"\n[sweep] summary written to {root / 'summary.csv'}")


if __name__ == "__main__":
    main()