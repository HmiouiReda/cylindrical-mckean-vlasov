"""Multi-seed aggregation of the headline metrics.

    python run_seeds.py --benchmark ou --seeds 0,1,2,3,4 -- --baselines emp --psi-epochs 150

Runs `main.py` once per seed (identical arguments after the `--`
separator), then reports mean +/- std ACROSS SEEDS of the headline metrics
-- i.e. the training/sampling variability that the per-scenario +/- of a
single run does not capture.  Writes outputs/<benchmark>_seeds/summary.json.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HEADLINE = ["mdn_val_nll", "w2_mdn_mean", "w2_floor_mean",
            "rel_l2_full", "rel_l2_empirical","rel_l2_oracle", 
            "reference_resolution"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", type=str, required=True)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--outdir", type=str, default="",
                   help="default: outputs/<benchmark>_seeds")
    p.add_argument("rest", nargs=argparse.REMAINDER,
                   help="arguments forwarded to main.py (after --)")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    root = Path(args.outdir) if args.outdir else Path("outputs") / f"{args.benchmark}_seeds"
    root.mkdir(parents=True, exist_ok=True)
    fwd = [a for a in args.rest if a != "--"]

    per_seed = {}
    for s in seeds:
        out = root / f"seed{s}"
        cmd = [sys.executable, "main.py", "--benchmark", args.benchmark,
               "--seed", str(s), "--outdir", str(out)] + fwd
        print(f"\n===== seed {s}:  {' '.join(cmd)}")
        res = subprocess.run(cmd)
        if res.returncode != 0:
            raise SystemExit(f"seed {s} failed")
        per_seed[s] = json.loads((out / "metrics.json").read_text())

    summary = {"benchmark": args.benchmark, "seeds": seeds,
               "forwarded_args": fwd, "metrics": {}}
    print("\n" + "=" * 64)
    print(f"MULTI-SEED SUMMARY ({args.benchmark}, {len(seeds)} seeds)")
    print("=" * 64)
    for k in HEADLINE:
        vals = [per_seed[s].get(k) for s in seeds]
        if any(v is None for v in vals):
            continue
        vals = np.asarray(vals, dtype=float)
        summary["metrics"][k] = {"mean": float(vals.mean()),
                                 "std": float(vals.std(ddof=1)),
                                 "values": vals.tolist()}
        print(f"  {k:22s}: {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}")
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWritten to {root/'summary.json'}")


if __name__ == "__main__":
    main()
