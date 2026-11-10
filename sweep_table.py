"""Read the hyperparameter sweep, apply the selection rule, write the tables.

    python sweep_table.py OUT --stage select   # after the one-factor stage
    python sweep_table.py OUT --stage final    # after the combined runs

Run names are <benchmark>__s<seed>__<kind>[__<value>], made by run_sweep.sh:
    default                 the current configuration
    ts__<k>                 the default retrained with --train-seed k
    <factor>__<value>       one factor changed
    combined                every accepted change together
    order__default, order__final    kuramoto, order functional, check only

The rule (tasks_plan.html, sweep section). Choose on validation relative L2
of the functional only; test numbers are written out but never read here.
sigma = sd of the validation error of the default over train seeds 0..4, data
seed 0. A change is real if it beats the default on both data seeds and the
mean paired improvement exceeds 2 sigma. It is vetoed if the law (validation
NLL or validation W2) gets worse by more than 2 of its own sigma.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

BENCHMARKS = ["ou", "kuramoto"]
SEEDS = [0, 1]
FACTORS = ["J", "n_fourier", "sig_level", "mdn_depth", "psi_depth", "hidden", "lr"]
GLOBAL = ["J", "n_fourier", "mdn_depth", "psi_depth", "hidden", "lr"]
CLI = {"J": "--J", "n_fourier": "--n-fourier", "sig_level": "--sig-level",
       "mdn_depth": "--mdn-depth", "psi_depth": "--psi-depth",
       "hidden": "--hidden", "lr": "--lr"}
# where a sig_level result applies: the benchmark's class
CLASS = {"ou": "OU family (ou, ou_uniform, ou_gmm, ou2d)",
         "kuramoto": "particle family (double_well, multiplicative, kuramoto)"}
LAW = {"mdn_val_nll": "validation NLL", "w2_mdn_val_mean": "validation W2"}


def load_runs(out: Path) -> list[dict]:
    runs = []
    for path in sorted((out / "runs").glob("*/metrics.json")):
        parts = path.parent.name.split("__")
        if len(parts) < 3 or parts[0] not in BENCHMARKS:
            continue
        m = json.loads(path.read_text())
        runs.append(dict(name=path.parent.name, benchmark=parts[0],
                         seed=int(parts[1][1:]), kind=parts[2],
                         value=(parts[3] if len(parts) > 3 else None), m=m))
    return runs


def find(runs, benchmark, seed, kind, value=None):
    for r in runs:
        if (r["benchmark"], r["seed"], r["kind"], r["value"]) == \
                (benchmark, seed, kind, value):
            return r
    return None


def same(a, b) -> bool:
    return abs(float(a) - float(b)) <= 1e-12 * max(1.0, abs(float(b)))


class Bench:
    """Everything the rule needs for one benchmark."""

    def __init__(self, runs, b):
        self.b = b
        self.runs = runs
        self.default = {s: find(runs, b, s, "default") for s in SEEDS}
        d0 = self.default[0]
        self.config = d0["m"]["config"] if d0 else None
        reps = [d0] + [find(runs, b, 0, "ts", str(k)) for k in range(1, 5)]
        reps = [r for r in reps if r is not None]
        self.n_reps = len(reps)
        self.sigma = {}
        for key in ["rel_l2_val", *LAW]:
            vals = [r["m"][key] for r in reps]
            self.sigma[key] = float(np.std(vals, ddof=1)) if len(vals) >= 2 else None

    def default_value(self, factor):
        return self.config[factor]

    def compare(self, run_for_seed) -> dict:
        """Paired differences of a candidate against the default, per seed."""
        out = dict(n=0, gain=None, both=False, gain_sigma=None, law_worse=[])
        diffs, law = [], {k: [] for k in LAW}
        for s in SEEDS:
            r, d = run_for_seed(s), self.default[s]
            if r is None or d is None:
                continue
            diffs.append(d["m"]["rel_l2_val"] - r["m"]["rel_l2_val"])  # >0 = better
            for k in LAW:
                law[k].append(r["m"][k] - d["m"][k])                    # >0 = worse
        out["n"] = len(diffs)
        if not diffs:
            return out
        out["gain"] = float(np.mean(diffs))
        out["both"] = len(diffs) == len(SEEDS) and all(x > 0 for x in diffs)
        sig = self.sigma["rel_l2_val"]
        if sig:
            out["gain_sigma"] = out["gain"] / sig
        for k in LAW:
            if self.sigma[k] and law[k] and np.mean(law[k]) > 2 * self.sigma[k]:
                out["law_worse"].append(LAW[k])
        return out

    def factor_run(self, factor, value):
        return lambda s: (self.default[s] if same(value, self.default_value(factor))
                          else find(self.runs, self.b, s, factor, value))

    def real(self, c) -> bool:
        sig = self.sigma["rel_l2_val"]
        return bool(c["both"] and sig and c["gain"] > 2 * sig)

    def values(self, factor):
        vals = {r["value"] for r in self.runs
                if r["benchmark"] == self.b and r["kind"] == factor}
        if self.config:
            vals.add(str(self.default_value(factor)))
        return sorted(vals, key=float)


def select(runs) -> dict:
    """One-factor results -> accepted changes, flags, per-benchmark config."""
    benches = {b: Bench(runs, b) for b in BENCHMARKS}
    flags, table = [], {}
    winner = {b: {} for b in BENCHMARKS}
    for b, B in benches.items():
        if B.config is None:
            flags.append(f"{b}: default run missing, nothing selected")
            continue
        if B.n_reps < 3:
            flags.append(f"{b}: only {B.n_reps} train-seed replicates for sigma")
        for f in FACTORS:
            best = None
            for v in B.values(f):
                if same(v, B.default_value(f)):
                    continue
                c = B.compare(B.factor_run(f, v))
                table[(b, f, v)] = c
                if B.real(c):
                    if c["law_worse"]:
                        flags.append(f"{b}: {f}={v} wins on the functional but "
                                     f"{', '.join(c['law_worse'])} worse; vetoed")
                        continue
                    if best is None or c["gain"] > table[(b, f, best)]["gain"]:
                        best = v
            if best is not None:
                winner[b][f] = best

    changes = []
    for f in GLOBAL:
        w = {b: winner[b].get(f) for b in BENCHMARKS if winner[b].get(f)}
        if not w:
            continue
        if len(w) == 2:
            (b1, v1), (b2, v2) = w.items()
            if same(v1, v2):
                changes.append(dict(param=f, new=v1, scope="all benchmarks",
                                    benchmarks=BENCHMARKS, source=[b1, b2]))
            else:
                flags.append(f"{f}: ou wants {w['ou']}, kuramoto wants "
                             f"{w['kuramoto']}; kept the default")
            continue
        (b, v), = w.items()
        other = [x for x in BENCHMARKS if x != b][0]
        O = benches[other]
        if O.config is None:
            flags.append(f"{f}={v}: {other} missing, not transferred")
            continue
        c = O.compare(O.factor_run(f, v))
        sig = O.sigma["rel_l2_val"]
        if c["n"] == 0 or sig is None:
            flags.append(f"{f}={v}: no {other} run to check transfer")
        elif c["gain"] < -2 * sig:
            flags.append(f"{f}={v} real on {b} but worse on {other} by "
                         f"{-c['gain'] / sig:.1f} sigma; kept the default")
        else:
            changes.append(dict(param=f, new=v, scope="all benchmarks",
                                benchmarks=BENCHMARKS, source=[b]))
    for b in BENCHMARKS:
        v = winner[b].get("sig_level")
        if v:
            changes.append(dict(param="sig_level", new=v, scope=CLASS[b],
                                benchmarks=[b], source=[b]))
    for ch in changes:
        ch["old"] = {b: benches[b].default_value(ch["param"])
                     for b in ch["benchmarks"] if benches[b].config}
        ch["gain_sigma"] = {b: table[(b, ch["param"], ch["new"])]["gain_sigma"]
                            for b in ch["source"]}
    per_bench = {b: [ch for ch in changes if b in ch["benchmarks"]]
                 for b in BENCHMARKS}
    return dict(benches=benches, table=table, winner=winner, changes=changes,
                per_bench=per_bench, flags=flags)


def cli_args(chs) -> str:
    return " ".join(f"{CLI[ch['param']]} {ch['new']}" for ch in chs)


def stage_select(out: Path):
    sel = select(load_runs(out))
    jobs = []
    for b in BENCHMARKS:
        chs = sel["per_bench"][b]
        if len(chs) >= 2:                      # one change: its own run exists
            for s in SEEDS:
                jobs.append(f"{b}__s{s}__combined --benchmark {b} --seed {s} "
                            f"{cli_args(chs)}")
    for s in SEEDS:
        jobs.append(f"kuramoto__s{s}__order__default --benchmark kuramoto "
                    f"--seed {s} --functional order")
        chs = sel["per_bench"]["kuramoto"]
        if chs:
            jobs.append(f"kuramoto__s{s}__order__final --benchmark kuramoto "
                        f"--seed {s} --functional order {cli_args(chs)}")
    (out / "stage_c_jobs.txt").write_text("\n".join(jobs) + "\n")
    (out / "selection.json").write_text(json.dumps(
        dict(changes=sel["changes"], flags=sel["flags"]), indent=2))
    print(f"[select] {len(sel['changes'])} change(s), "
          f"{len(jobs)} stage C job(s); flags: {len(sel['flags'])}")
    for ch in sel["changes"]:
        print(f"  {ch['param']}: {ch['old']} -> {ch['new']}  ({ch['scope']})")
    for fl in sel["flags"]:
        print(f"  flag: {fl}")


def verdict(sel, runs) -> tuple[list[dict], list[str]]:
    """Combined config, or the best single change alone."""
    notes = []
    changes = sel["changes"]
    if len(changes) <= 1:
        return changes, notes
    benches = sel["benches"]
    ok = True
    for b in BENCHMARKS:
        chs = sel["per_bench"][b]
        if len(chs) < 2:
            continue
        B = benches[b]
        c = B.compare(lambda s: find(runs, b, s, "combined"))
        if c["n"] < len(SEEDS):
            notes.append(f"{b}: combined run missing")
            ok = False
            continue
        best_single = max(sel["table"][(b, ch["param"], ch["new"])]["gain"]
                          for ch in chs if (b, ch["param"], ch["new"]) in sel["table"])
        sig = B.sigma["rel_l2_val"]
        if not B.real(c):
            notes.append(f"{b}: combined does not beat the default by the rule "
                         f"({c['gain_sigma']:.1f} sigma)")
            ok = False
        elif c["gain"] < best_single - 2 * sig:
            notes.append(f"{b}: combined worse than the best single change by "
                         f"{(best_single - c['gain']) / sig:.1f} sigma")
            ok = False
        else:
            notes.append(f"{b}: combined beats the default by "
                         f"{c['gain_sigma']:.1f} sigma")
    if ok:
        notes.append("verdict: take the combined configuration")
        return changes, notes
    best = max(changes, key=lambda ch: max(v or 0 for v in ch["gain_sigma"].values()))
    notes.append(f"verdict: changes do not add up; take {best['param']}="
                 f"{best['new']} alone")
    return [best], notes


def stage_final(out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load_runs(out)
    sel = select(runs)
    final, notes = verdict(sel, runs)
    benches = sel["benches"]

    # one row per run
    keys = ["rel_l2_val", "rel_l2_full", "w2_mdn_val_mean", "w2_mdn_mean",
            "mdn_val_nll", "runtime_sec"]
    with open(out / "sweep_runs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "benchmark", "seed", "kind", "value", "train_seed",
                    "functional", *keys, "torch_threads", "simulate_cached"])
        for r in runs:
            m, env = r["m"], r["m"]["timings"]["environment"]
            w.writerow([r["name"], r["benchmark"], r["seed"], r["kind"], r["value"],
                        m["config"].get("train_seed"), m["config"]["functional"],
                        *[m.get(k) for k in keys], env["start"].get("torch_threads"),
                        env.get("simulate_cached")])

    # mean over the two data seeds, per factor value
    rows = []
    for b, B in benches.items():
        if B.config is None:
            continue
        for f in FACTORS:
            for v in B.values(f):
                rs = [B.factor_run(f, v)(s) for s in SEEDS]
                rs = [r for r in rs if r is not None]
                if not rs:
                    continue
                mean = {k: float(np.mean([r["m"][k] for r in rs])) for k in keys}
                c = sel["table"].get((b, f, v))
                rows.append(dict(benchmark=b, factor=f, value=v, n_seeds=len(rs),
                                 default=same(v, B.default_value(f)),
                                 gain_sigma=(None if c is None else c["gain_sigma"]),
                                 **mean))
    with open(out / "sweep_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["empty"])
        w.writeheader()
        w.writerows(rows)

    chosen = {b: {f: str(benches[b].default_value(f)) for f in FACTORS}
              for b in BENCHMARKS if benches[b].config}
    for ch in final:
        for b in ch["benchmarks"]:
            if b in chosen:
                chosen[b][ch["param"]] = ch["new"]

    # section 4.5 table, test numbers only
    name = {"J": "Mixture size $J$", "n_fourier": "Fourier frequencies $P$",
            "sig_level": "Signature level", "mdn_depth": "MDN depth",
            "psi_depth": r"$\Psi$ depth", "hidden": "Width", "lr": "Learning rate"}
    pretty = {"ou": "OU", "kuramoto": "Kuramoto"}
    lines = [r"\begin{tabular}{p{0.24\linewidth}p{0.22\linewidth}p{0.22\linewidth}p{0.22\linewidth}}",
             r"\toprule",
             r"Sweep and benchmark & Settings & test rel.\ $L^2$ (\%) & test $W_2$ \\",
             r"\midrule"]
    for f in FACTORS:
        for b in BENCHMARKS:
            rr = [r for r in rows if r["benchmark"] == b and r["factor"] == f]
            if not rr:
                continue
            mark = lambda r, s: (rf"\textbf{{{s}}}" if same(r["value"], chosen[b][f]) else s)
            vals = ", ".join(mark(r, r["value"]) for r in rr)
            rel = ", ".join(mark(r, f"{100 * r['rel_l2_full']:.2f}") for r in rr)
            w2 = ", ".join(mark(r, f"{r['w2_mdn_mean']:.4f}") for r in rr)
            lines.append(f"{name[f]}, {pretty[b]} & ${vals}$ & {rel} & {w2} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "sweep_table.tex").write_text("\n".join(lines) + "\n")

    # one figure per factor: validation and test functional error
    for f in FACTORS:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        for ax, b in zip(axes, BENCHMARKS):
            rr = [r for r in rows if r["benchmark"] == b and r["factor"] == f]
            if not rr:
                ax.set_visible(False)
                continue
            x = [float(r["value"]) for r in rr]
            ax.plot(x, [100 * r["rel_l2_val"] for r in rr], "o-", label="validation")
            ax.plot(x, [100 * r["rel_l2_full"] for r in rr], "s--", label="test")
            ax.axvline(float(chosen[b][f]), color="grey", lw=0.8)
            if f == "lr":
                ax.set_xscale("log")
            ax.set_title(pretty[b])
            ax.set_xlabel(f)
            ax.set_ylabel(r"relative $L^2$ (%)")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"sweep_{f}.png", dpi=140)
        plt.close(fig)

    # order functional check
    order = {}
    for s in SEEDS:
        for tag in ["default", "final"]:
            r = find(runs, "kuramoto", s, "order", tag)
            if r:
                order[f"s{s}_{tag}"] = dict(rel_l2_val=r["m"]["rel_l2_val"],
                                            rel_l2_test=r["m"]["rel_l2_full"])

    proposal = dict(changes=final, notes=notes, flags=sel["flags"],
                    candidates=sel["changes"], order_check=order,
                    sigma={b: B.sigma for b, B in benches.items()},
                    n_replicates={b: B.n_reps for b, B in benches.items()})
    (out / "freeze_proposal.json").write_text(json.dumps(proposal, indent=2))

    # human-readable report
    rep = ["hyperparameter sweep: report", ""]
    for b, B in benches.items():
        sig = B.sigma["rel_l2_val"]
        rep.append(f"{b}: training-noise sigma of validation rel L2 = "
                   + ("n/a" if sig is None else f"{100 * sig:.3f} %")
                   + f" ({B.n_reps} replicates)")
    rep.append("")
    rep.append("gain = default minus candidate, validation rel L2, mean of seeds, "
               "in sigma; * = real by the rule")
    for b, B in benches.items():
        if B.config is None:
            continue
        rep.append(f"\n{b}")
        for f in FACTORS:
            cells = []
            for v in B.values(f):
                if same(v, B.default_value(f)):
                    cells.append(f"[{v}]")
                    continue
                c = sel["table"].get((b, f, v))
                if c is None or c["gain_sigma"] is None:
                    cells.append(f"{v}:?")
                else:
                    cells.append(f"{v}:{c['gain_sigma']:+.1f}" + ("*" if B.real(c) else ""))
            rep.append(f"  {f:10s} " + "  ".join(cells))
    rep.append("\nflags:" if sel["flags"] else "\nflags: none")
    rep += [f"  {fl}" for fl in sel["flags"]]
    rep.append("\ncombination:")
    rep += [f"  {n}" for n in notes] or ["  nothing to combine"]
    rep.append("\nproposed freeze:" if final else "\nproposed freeze: no change")
    for ch in final:
        rep.append(f"  {ch['param']}: {ch['old']} -> {ch['new']}  ({ch['scope']})")
    rep.append("\nkuramoto order functional (check only, % rel L2 val / test):")
    for k, v in order.items():
        rep.append(f"  {k}: {100 * v['rel_l2_val']:.2f} / {100 * v['rel_l2_test']:.2f}")
    status = out / "status.txt"
    failed = ([ln for ln in status.read_text().splitlines() if "FAILED" in ln]
              if status.exists() else [])
    rep.append(f"\nfailed steps (status.txt): {len(failed)}")
    rep += [f"  {ln}" for ln in failed]
    (out / "sweep_report.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out")
    p.add_argument("--stage", choices=["select", "final"], required=True)
    a = p.parse_args()
    (stage_select if a.stage == "select" else stage_final)(Path(a.out))


if __name__ == "__main__":
    main()
