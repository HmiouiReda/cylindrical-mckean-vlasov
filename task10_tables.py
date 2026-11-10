"""Read task 10's runs and write the paper's tables, figures and checks.

    python task10_tables.py outputs/task10
    python task10_tables.py outputs/task10 --sweep outputs/sweep --task5 outputs/task5

Run names are made by run_task10.sh:
    <benchmark>__<functional>__s<seed>   the 75 runs (seed 0 moments = cost runs)
    task5__check_<b>, task5__dw_M<m>, task5__dw_N<n>, task5__ou2d_e1200
    single__ou__s<k>, kkeep__ou__s<k>, ito__multiplicative, reda__<name>

Writes OUT/paper/: the LaTeX tables, the figures under the names AHR.tex uses,
numbers.txt (inline numbers, old -> new), task5_rows.txt, checks.txt,
validation_report.txt, reda_runs.csv; and OUT/task10_runs.csv. Missing runs
are listed, never fatal. Comparisons against the sweep, task 5 and the old
kuramoto runs are skipped when their directories are absent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ALL7 = ["ou", "ou_uniform", "ou_gmm", "double_well", "multiplicative", "ou2d", "kuramoto"]
ONE_D = ["ou", "ou_uniform", "ou_gmm", "double_well", "multiplicative", "kuramoto"]
ORDER = ["double_well", "kuramoto"]
PARTICLE = ["double_well", "multiplicative", "kuramoto"]
SEEDS = [0, 1, 2, 3, 4]
LABEL = {"ou": "OU--Gaussian", "ou_uniform": "OU--uniform", "ou_gmm": "OU--GMM",
         "double_well": "Double well", "multiplicative": "Multiplicative",
         "ou2d": "OU--2D", "kuramoto": "Kuramoto"}
PLAIN = {"ou": "OU-Gaussian", "ou_uniform": "OU-uniform", "ou_gmm": "OU-GMM",
         "double_well": "Double well", "multiplicative": "Multiplicative",
         "ou2d": "OU-2D", "kuramoto": "Kuramoto"}

# the paper's five-seed numbers before task 10 (AHR.tex, tab:law-main,
# tab:order-main; kuramoto quantile from outputs/kuramoto_seeds_quantile)
OLD_LAW = {  # benchmark: (nll, sd, mdn w2, sd)
    "ou": (0.2951, 0.0045, 0.0057, 0.0008), "ou_uniform": (0.2187, 0.0033, 0.0054, 0.0008),
    "ou_gmm": (0.3642, 0.0023, 0.0082, 0.0009), "double_well": (0.8373, 0.0064, 0.0240, 0.0011),
    "multiplicative": (0.0232, 0.0086, 0.0127, 0.0006), "ou2d": (0.1977, 0.0072, 0.0064, 0.0003),
    "kuramoto": (1.6933, 0.0023, 0.0310, 0.0011)}
OLD_FUNC = {  # (benchmark, functional): (mdn rel l2 %, sd); same evaluation code
    ("kuramoto", "moments"): (1.81, 0.70), ("kuramoto", "quantile"): (1.70, 0.04),
    ("kuramoto", "order"): (3.18, 0.07), ("double_well", "order"): (1.61, 0.06)}
SWEEP_SIGMA = {"ou": 0.012, "kuramoto": 0.028}   # points of rel L2, sweep_report


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def load(out: Path) -> dict:
    runs = {}
    for d in sorted((out / "runs").glob("*")):
        if not d.is_dir():
            continue
        runs[d.name] = dict(
            dir=d, done=(d / ".done").exists(), m=read_json(d / "metrics.json"),
            es=read_json(d / "error_sources.json"), ito=read_json(d / "ito_strat_check.json"))
    return runs


def metric(runs, name, key):
    r = runs.get(name)
    if not r or not r["m"]:
        return None
    v = r["m"]
    for k in key.split("."):
        if not isinstance(v, dict) or k not in v:
            return None
        v = v[k]
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def over_seeds(runs, b, f, key, seeds=SEEDS, prefix=None):
    """values of one key over the seeds that exist, and the missing names"""
    vals, missing = [], []
    for s in seeds:
        name = prefix.format(s=s) if prefix else f"{b}__{f}__s{s}"
        v = metric(runs, name, key)
        (missing.append(name) if v is None else vals.append(float(v)))
    return np.asarray(vals), missing


def ms(vals):
    if len(vals) == 0:
        return None, None
    return float(vals.mean()), (float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)


def pm(vals, scale=1.0, nd=4):
    m, s = ms(vals)
    if m is None:
        return "--"
    return f"${m * scale:.{nd}f}\\pm{s * scale:.{nd}f}$"


def short(vals, scale=1.0, nd=4):
    m, s = ms(vals)
    return "missing" if m is None else f"{m * scale:.{nd}f} +/- {s * scale:.{nd}f} (n={len(vals)})"


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def law_table(runs) -> str:
    rows = []
    for b in ALL7:
        cells = [pm(over_seeds(runs, b, "moments", k)[0]) for k in
                 ("mdn_val_nll", "w2_mdn_mean", "w2_floor_mean")]
        res = over_seeds(runs, b, "moments", "reference_resolution")[0]
        cells.append(pm(res) if b in PARTICLE else "--")
        rows.append(f"{LABEL[b]} & " + " & ".join(cells) + r" \\")
    head = (r"Benchmark & Validation NLL & MDN discrepancy & Empirical \(N=200\) & "
            r"Reference resolution \\")
    return tabular("lcccc", head, rows)


FUNC_KEYS = ["rel_l2_empirical", "rel_l2_full", "rel_l2_oracle",
             "functional_reference_floor", "rel_l2_mdn_noise_adjusted"]


def functional_table(runs, f, benches) -> str:
    rows = []
    for b in benches:
        cells = []
        for k in FUNC_KEYS:
            vals = over_seeds(runs, b, f, k)[0]
            cells.append(pm(vals, 100, 2) if (b in PARTICLE or k in FUNC_KEYS[:3]) else "--")
        rows.append(f"{LABEL[b]} & " + " & ".join(cells) + r" \\")
    head = r"Benchmark & Empirical & MDN & Exact/indep. & Target floor & MDN adjusted \\"
    return tabular("lccccc", head, rows)


def tabular(cols, head, rows) -> str:
    return "\n".join([f"\\begin{{tabular}}{{{cols}}}", r"\toprule", head, r"\midrule",
                      *rows, r"\bottomrule", r"\end{tabular}", ""])


# ---------------------------------------------------------------------------
# cost (task 8): seed-0 moments runs, run alone, cache misses
# ---------------------------------------------------------------------------
def accuracy_matched(runs, b, es_new):
    """N' at which a plug-in cloud matches the MDN's W2, assuming N^(-1/2)"""
    if b in PARTICLE:
        es = es_new.get(b)
        if not es:
            return None
        cs = [es[k]["bias_rms"] * math.sqrt(int(k.split("_")[1]))
              for k in es if k.startswith("plugin_")]
        target = es["mdn"]["vs_H"]["bias_rms"]
        return (float(np.mean(cs)) / target) ** 2 if cs and target > 0 else None
    w2 = metric(runs, f"{b}__moments__s0", "w2_mdn_mean")
    floor = metric(runs, f"{b}__moments__s0", "w2_floor_mean")
    n = metric(runs, f"{b}__moments__s0", "config.N")
    if not (w2 and floor and n):
        return None
    return (floor * math.sqrt(n) / w2) ** 2


def cost(runs, es_new):
    rows, notes, tex = [], [], []
    for b in ALL7:
        name = f"{b}__moments__s0"
        m = (runs.get(name) or {}).get("m")
        if not m or "timings" not in m:
            notes.append(f"{b}: cost run missing")
            continue
        t = m["timings"]
        env = t.get("environment", {})
        if env.get("simulate_cached"):
            notes.append(f"{b}: cost run was a cache hit, no simulation time; delete "
                         f"sim_cache/{b}_s0_*.pkl and the run, rerun it alone")
            continue
        load = (env.get("start", {}).get("loadavg"), env.get("end", {}).get("loadavg"))
        notes.append(f"{b}: torch threads {env.get('start', {}).get('torch_threads')}, "
                     f"load average start {load[0]}, end {load[1]}")
        sim, ev, steps = t.get("simulate", {}), t.get("eval_points", {}), t.get("particle_steps", {})
        g = lambda d, k: float(d.get(k, 0.0) or 0.0)  # noqa: E731
        train_cloud = g(sim, "train_cloud") or g(sim, "total")
        ref_a, ref_b = g(sim, "ref_A"), g(sim, "ref_B")
        c_sys = g(ev, "total")
        mdn, psi = g(t.get("mdn_train", {}), "total"), float(t.get("psi_train", 0.0))
        inf = t.get("inference", {})
        latency = g(inf, "single_scenario_latency")
        upfront = (train_cloud + ref_b + g(ev, "train") + g(ev, "validation")
                   + float(t.get("features", 0.0)) + mdn + psi)
        # cost of one particle-step: a 2000-particle reference where there is
        # one, else the training cloud
        if ref_a and steps.get("ref_A"):
            rate = ref_a / steps["ref_A"]
        else:
            rate = train_cloud / steps["train_cloud"]
        n_steps = m["config"]["benchmark_params"]["n_steps"]
        n_match = accuracy_matched(runs, b, es_new)
        be = {}
        for label, n in (("200", 200), ("2000", 2000), ("matched", n_match)):
            if n is None:
                be[label] = None
                continue
            c = rate * n * n_steps
            be[label] = upfront / (c - latency) if c > latency else float("inf")
        rows.append(dict(benchmark=b, train_cloud=train_cloud, ref_A=ref_a, ref_B=ref_b,
                         system_C=c_sys, mdn_train=mdn, psi_train=psi,
                         batched_per_scenario=g(inf, "batched_per_scenario"),
                         single_scenario_latency=latency, upfront=upfront,
                         us_per_particle_step=1e6 * rate, n_steps=n_steps,
                         n_matched=n_match, breakeven_200=be["200"],
                         breakeven_2000=be["2000"], breakeven_matched=be["matched"],
                         total=float(t.get("total", 0.0))))

    def num(x, fmt="{:.0f}"):
        if x is None:
            return "--"
        if x == float("inf"):
            return r"$\infty$"
        return fmt.format(x)

    for r in rows:
        sim = (f"{num(r['train_cloud'])} / {num(r['ref_A'] or None)} / "
               f"{num(r['ref_B'] or None)} / {num(r['system_C'] or None)}")
        tex.append(f"{LABEL[r['benchmark']]} & {sim} & {num(r['mdn_train'])} & "
                   f"{num(r['psi_train'])} & {num(1e3 * r['single_scenario_latency'], '{:.2f}')} & "
                   f"{num(r['upfront'])} & {num(r['breakeven_200'], '{:,.0f}')} & "
                   f"{num(r['breakeven_2000'], '{:,.0f}')} & "
                   f"{num(r['n_matched'], '{:,.0f}')}: {num(r['breakeven_matched'], '{:,.0f}')}"
                   + r" \\")
    head = (r"Benchmark & Simulation (s): train / $A$ / $B$ / $C$ & MDN (s) & $\Psi$ (s) & "
            r"Inference (ms) & Upfront (s) & $n^\ast_{200}$ & $n^\ast_{2000}$ & "
            r"$N'$: $n^\ast_{N'}$ \\")
    return rows, notes, tabular("lcccccccc", head, tex)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def figures(runs, paper: Path):
    have = [b for b in ALL7 if len(over_seeds(runs, b, "moments", "w2_mdn_mean")[0])]
    if not have:
        return ["figures: no moments runs"]
    x = np.arange(len(have), dtype=float)
    labels = [PLAIN[b] for b in have]

    def bars(ax, keys, names, width, scale=1.0):
        for i, (k, n) in enumerate(zip(keys, names)):
            stats = [ms(over_seeds(runs, b, "moments", k)[0]) for b in have]
            mean = np.array([np.nan if a is None else a * scale for a, _ in stats])
            sd = np.array([0.0 if s is None else s * scale for _, s in stats])
            off = (i - (len(keys) - 1) / 2) * width
            ax.bar(x + off, mean, width, yerr=sd, capsize=4, label=n)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bars(ax, ["w2_mdn_mean", "w2_floor_mean"], ["MDN", "Empirical cloud (N=200)"], 0.36)
    ax.set_ylabel("Mean test discrepancy")
    fig.tight_layout()
    fig.savefig(paper / "law_error_across_benchmarks.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    bars(ax, ["rel_l2_empirical", "rel_l2_full", "rel_l2_oracle"],
         ["Empirical", "MDN", "Exact / indep. reference"], 0.25, 100.0)
    ax.set_ylabel(r"Relative $L^2$ error (%)")
    fig.tight_layout()
    fig.savefig(paper / "functional_error_across_benchmarks.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    notes = []
    for src, dst in (("ou_gmm__moments__s0/fig6_convergence.png", "Figure1.png"),
                     ("double_well__moments__s0/fig2_mdn_vs_ref.png",
                      "double_well_conditional_law.png"),
                     ("ou2d__moments__s0/fig2_mdn_vs_ref.png", "ou_2d_conditional_ellipses.png")):
        p = paper.parent / "runs" / src
        if p.exists():
            shutil.copyfile(p, paper / dst)
        else:
            notes.append(f"figure {dst}: {src} missing")
    return notes


# ---------------------------------------------------------------------------
# inline numbers of section 4
# ---------------------------------------------------------------------------
def numbers(runs) -> list[str]:
    L = []

    def mom(b, k):
        return over_seeds(runs, b, "moments", k)[0]

    def line(anchor, old, new):
        L.append(f"{anchor}\n    old: {old}\n    new: {new}")

    L.append("== sec:law-results ==")
    line("'mean W2 is 0.0054 for the uniform initial law'", "0.0054", short(mom("ou_uniform", "w2_mdn_mean")))
    line("'and 0.0082 for the two component mixture'", "0.0082", short(mom("ou_gmm", "w2_mdn_mean")))
    line("'empirical cloud discrepancies 0.0450 and 0.0550'", "0.0450, 0.0550",
         f"{short(mom('ou_uniform', 'w2_floor_mean'))}; {short(mom('ou_gmm', 'w2_floor_mean'))}")
    k_w2, k_fl = ms(mom("kuramoto", "w2_mdn_mean"))[0], ms(mom("kuramoto", "w2_floor_mean"))[0]
    if k_w2:
        line("kuramoto block: 'Relative to the width of its own law its discrepancy is 2.0%' "
             "(W2 / 1.5, the paper's 'about 1.5')", "2.0 (0.0310 / 1.5 = 2.07)",
             f"{100 * k_w2 / 1.5:.2f}")
    ratios = []
    for b in ("kuramoto", "double_well", "multiplicative"):
        w, f = ms(mom(b, "w2_mdn_mean"))[0], ms(mom(b, "w2_floor_mean"))[0]
        ratios.append(f"{b} {f / w:.2f}" if w and f else f"{b} missing")
    line("'its margin over the N=200 particle cloud, a factor 3.3, matches 3.4 and 3.2'",
         "kuramoto 3.3, double_well 3.4, multiplicative 3.2", ", ".join(ratios))
    line("'the measured 1.6933 lies below the matched Gaussian value' (1.84)", "1.6933",
         short(mom("kuramoto", "mdn_val_nll")))
    line("'It equals 0.2896 +/- 0.0012 across seeds (0.2901 at seed 0)' (ou entropy floor)",
         "0.2896 +/- 0.0012; 0.2901", f"{short(mom('ou', 'entropy_floor'))}; seed 0 "
         f"{metric(runs, 'ou__moments__s0', 'entropy_floor')}")
    line("'while the validation NLL is 0.2951 +/- 0.0045' (ou)", "0.2951 +/- 0.0045",
         short(mom("ou", "mdn_val_nll")))
    line("'the exact entropy is 0.1800 +/- 0.0026' (ou2d); NLL beside it",
         "0.1800 +/- 0.0026; NLL 0.1977 +/- 0.0072",
         f"{short(mom('ou2d', 'entropy_floor'))}; NLL {short(mom('ou2d', 'mdn_val_nll'))}")
    above = []
    for b in PARTICLE:
        w, r = ms(mom(b, "w2_mdn_mean"))[0], ms(mom(b, "reference_resolution"))[0]
        above.append(f"{b} MDN {w:.4f} vs resolution {r:.4f}" if w and r else f"{b} missing")
    line("'the MDN errors are close to but above the measured pooled reference resolutions' "
         "-- RE-READ", "dw 0.0240/0.0179, mult 0.0127/0.0092, kur 0.0310/0.0230", "; ".join(above))

    L.append("\n== sec:functional-results ==")
    for b, old in (("double_well", "2.24 / 1.36"), ("multiplicative", "1.40 / 1.11")):
        line(f"'{b} MDN error ... after subtracting ... the error is' (observed / adjusted)", old,
             f"{short(mom(b, 'rel_l2_full'), 100, 2)} / "
             f"{short(mom(b, 'rel_l2_mdn_noise_adjusted'), 100, 2)}")
    for b in ORDER:
        vals = {k: ms(over_seeds(runs, b, "order", k)[0])[0]
                for k in ("rel_l2_empirical", "rel_l2_full", "rel_l2_oracle")}
        if all(v is not None for v in vals.values()):
            ahead = "MDN" if vals["rel_l2_full"] < vals["rel_l2_oracle"] else "reference"
            line(f"order paragraph, {b}: who is ahead, MDN or independent reference -- RE-READ",
                 "dw: MDN; kuramoto: reference 'marginally ahead'",
                 f"{ahead} (emp {100 * vals['rel_l2_empirical']:.2f}, MDN "
                 f"{100 * vals['rel_l2_full']:.2f}, ref {100 * vals['rel_l2_oracle']:.2f})")
    line("'It is reported for seed 0 because ...' (quantile) -- REWRITE",
         "seed 0", "five seeds, tab_quantile_main.tex")

    L.append("\n== sec:single-observation ==")
    pre = "single__ou__s{s}"
    line("'it reaches W2 = 0.0362 +/- 0.0054' (single observation)", "0.0362 +/- 0.0054 (3 seeds)",
         short(over_seeds(runs, None, None, "w2_mdn_mean", prefix=pre)[0]))
    line("'below the N=200 empirical cloud discrepancy 0.0514 +/- 0.0010'", "0.0514 +/- 0.0010",
         short(over_seeds(runs, None, None, "w2_floor_mean", prefix=pre)[0]))
    line("'The downstream moment functional error is 2.92% +/- 0.34%'", "2.92 +/- 0.34",
         short(over_seeds(runs, None, None, "rel_l2_full", prefix=pre)[0], 100, 2))
    line("'compared with 3.74% +/- 0.16% for the empirical plug in'", "3.74 +/- 0.16",
         short(over_seeds(runs, None, None, "rel_l2_empirical", prefix=pre)[0], 100, 2))
    kk = {}
    for s in SEEDS:
        sw = metric_dict(runs, f"kkeep__ou__s{s}", "sweep_kkeep")
        for k, v in (sw or {}).items():
            kk.setdefault(int(k), []).append(float(v))
    line("K_keep table, mean W2 at K_keep = 1, 8, 64, 200", "0.0069 0.0060 0.0056 0.0058 (seed 0)",
         "; ".join(f"{k}: {short(np.asarray(v))}" for k, v in sorted(kk.items())) or "missing")

    L.append("\n== sec:ito-strat-num ==")
    ito = (runs.get("ito__multiplicative") or {}).get("ito")
    if ito:
        line("corrected gap, resolution, naive gap", "0.00449 +/- 0.00348; 0.01360; 0.04941 +/- 0.02048",
             f"{ito['gap_corrected_mean']:.5f} +/- {ito['gap_corrected_std']:.5f}; "
             f"{ito['mc_resolution']:.5f}; {ito['gap_naive_mean']:.5f} +/- {ito['gap_naive_std']:.5f}")
    else:
        line("corrected gap, resolution, naive gap", "0.00449; 0.01360; 0.04941", "missing")
    return L


def metric_dict(runs, name, key):
    r = runs.get(name)
    v = (r or {}).get("m") or {}
    return v.get(key) if isinstance(v.get(key), dict) else None


# ---------------------------------------------------------------------------
# task 5 rows
# ---------------------------------------------------------------------------
def task5_rows(runs, es_new, task5: Path | None) -> list[str]:
    L = []
    old = {}
    for b in PARTICLE:
        old[b] = read_json(task5 / b / "error_sources.json") if task5 else None
    for b in PARTICLE:
        es = es_new.get(b)
        if not es:
            L.append(f"{b}: error_sources missing")
            continue
        m, o = es["mdn"], (old[b] or {}).get("mdn", {})
        def f(d, *ks):
            for k in ks:
                d = (d or {}).get(k)
            return "--" if d is None else f"{d:.4f}"
        L.append(f"{b}: MDN vs H bias {f(m, 'vs_H', 'bias_rms')} (old {f(o, 'vs_H', 'bias_rms')}); "
                 f"vs reference test W2 {f(m, 'vs_reference_test', 'w2_mean')} "
                 f"(old {f(o, 'vs_reference_test', 'w2_mean')}), train "
                 f"{f(m, 'vs_reference_train', 'w2_mean')} (old {f(o, 'vs_reference_train', 'w2_mean')}); "
                 f"reference vs H error {f(es, 'reference', 'error_rms')}; "
                 f"accuracy-matched N' {accuracy_matched(runs, b, es_new) or float('nan'):.0f}")
    L.append("")
    base = "double_well__moments__s0"
    def run_line(name, label):
        m = (runs.get(name) or {}).get("m")
        if not m:
            return f"{label}: missing"
        # task 5's base run was outputs/task5/double_well
        old_dir = "double_well" if name == base else name.replace("task5__", "")
        o = read_json(task5 / old_dir / "metrics.json") if task5 else None
        o = o or {}
        return (f"{label}: W2 {m['w2_mdn_mean']:.4f} (old {o.get('w2_mdn_mean', float('nan')):.4f}), "
                f"rel L2 {100 * m['rel_l2_full']:.2f}% (old {100 * o.get('rel_l2_full', float('nan')):.2f}%)")
    L.append("training particles, double_well seed 0:")
    for name, label in (("task5__dw_N50", "  N = 50"), (base, "  N = 200"), ("task5__dw_N800", "  N = 800")):
        L.append(run_line(name, label))
    L.append("scenarios, double_well seed 0:")
    for name, label in (("task5__dw_M1000", "  M = 1000"), ("task5__dw_M2000", "  M = 2000"),
                        (base, "  M = 4000"), ("task5__dw_M8000", "  M = 8000")):
        L.append(run_line(name, label))
    m4, m12 = (runs.get("ou2d__moments__s0") or {}).get("m"), (runs.get("task5__ou2d_e1200") or {}).get("m")
    if m4 and m12:
        floor = m4.get("entropy_floor")
        L.append(f"ou2d seed 0: NLL 400 epochs {m4['mdn_val_nll']:.4f}, 1200 epochs "
                 f"{m12['mdn_val_nll']:.4f}, entropy {floor:.4f}; excess "
                 f"{m4['mdn_val_nll'] - floor:.4f} -> {m12['mdn_val_nll'] - floor:.4f} "
                 f"(old 0.0177, closed 0.0014)")
    else:
        L.append("ou2d 1200 epochs: missing")
    return L


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def numbers_equal(a, b, path="", out=None, rtol=1e-9):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}{k}: only on one side")
            else:
                numbers_equal(a[k], b[k], f"{path}{k}.", out, rtol)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not math.isclose(a, b, rel_tol=rtol, abs_tol=1e-12):
            out.append(f"{path[:-1]}: {a} vs {b}")
    elif a != b:
        out.append(f"{path[:-1]}: {a!r} vs {b!r}")
    return out


def checks(runs, es_new, sweep, task5, old_kur) -> list[str]:
    L = []
    # 1. the MDN does not see the functional
    worst, n = 0.0, 0
    for b in ALL7:
        for s in SEEDS:
            vals = [(metric(runs, f"{b}__{f}__s{s}", "mdn_val_nll"),
                     metric(runs, f"{b}__{f}__s{s}", "w2_mdn_mean"))
                    for f in ("moments", "quantile", "order")]
            vals = [v for v in vals if v[0] is not None]
            for v in vals[1:]:
                worst = max(worst, abs(v[0] - vals[0][0]), abs(v[1] - vals[0][1]))
                n += 1
    L.append(f"[{'ok' if worst < 1e-6 else 'DIFF'}] same MDN across functionals: {n} pairs, "
             f"largest difference in NLL or W2 {worst:.2e}")
    # 2. ou equals the sweep's frozen ou config
    if sweep and (sweep / "runs").exists():
        for s in (0, 1):
            old = read_json(sweep / "runs" / f"ou__s{s}__combined" / "metrics.json")
            new = (runs.get(f"ou__moments__s{s}") or {}).get("m")
            if old and new:
                keys = ["rel_l2_val", "w2_mdn_val_mean", "mdn_val_nll", "rel_l2_full", "w2_mdn_mean"]
                diff = {k: abs(new[k] - old[k]) for k in keys}
                ok = max(diff.values()) < 1e-9
                L.append(f"[{'ok' if ok else 'DIFF'}] ou seed {s} vs sweep combined: " +
                         ", ".join(f"{k} {new[k]:.6f}/{old[k]:.6f}" for k in keys))
            else:
                L.append(f"[skip] ou seed {s} vs sweep: a run is missing")
    else:
        L.append("[skip] sweep directory not found")
    # 3. error_sources: everything but the network must equal task 5
    for b in PARTICLE:
        old = read_json(task5 / b / "error_sources.json") if task5 else None
        new = es_new.get(b)
        if not (old and new):
            L.append(f"[skip] {b} error_sources vs task 5: missing")
            continue
        drop = {"mdn", "cost", "config"}
        diff = numbers_equal({k: v for k, v in new.items() if k not in drop},
                             {k: v for k, v in old.items() if k not in drop})
        L.append(f"[{'ok' if not diff else 'DIFF'}] {b} error_sources non-network rows equal to "
                 f"task 5" + ("" if not diff else ": " + "; ".join(diff[:5])))
    # 4. ito-stratonovich reproduces the paper
    ito = (runs.get("ito__multiplicative") or {}).get("ito")
    if ito:
        got = (round(ito["gap_corrected_mean"], 5), round(ito["gap_corrected_std"], 5),
               round(ito["mc_resolution"], 5), round(ito["gap_naive_mean"], 5),
               round(ito["gap_naive_std"], 5))
        want = (0.00449, 0.00348, 0.01360, 0.04941, 0.02048)
        L.append(f"[{'ok' if got == want else 'DIFF'}] ito-stratonovich {got} vs paper {want}")
    else:
        L.append("[skip] ito-stratonovich missing")
    # 5. data-only numbers equal the old kuramoto five-seed runs
    if old_kur and old_kur.exists():
        for s in SEEDS:
            old = read_json(old_kur / f"seed{s}" / "metrics.json")
            new = (runs.get(f"kuramoto__moments__s{s}") or {}).get("m")
            if old and new:
                d = [k for k in ("reference_resolution", "entropy_floor", "w2_floor_mean")
                     if old.get(k) is not None and not math.isclose(old[k], new.get(k) or 0, rel_tol=1e-9)]
                L.append(f"[{'ok' if not d else 'DIFF'}] kuramoto seed {s} data-only numbers vs old "
                         f"run" + ("" if not d else ": " + ", ".join(d)))
    else:
        L.append("[skip] old kuramoto runs not found")
    # 6. cost runs uncached
    for b in ALL7:
        m = (runs.get(f"{b}__moments__s0") or {}).get("m")
        if m:
            cached = m.get("timings", {}).get("environment", {}).get("simulate_cached")
            L.append(f"[{'ok' if cached is False else 'DIFF'}] {b} cost run uncached: {not cached}")
    return L


# ---------------------------------------------------------------------------
# validation report: the frozen config's first look; report only
# ---------------------------------------------------------------------------
def validation(runs, sweep) -> list[str]:
    L = ["first look at the frozen configuration; nothing here selects anything", ""]
    if sweep and (sweep / "runs").exists():
        for b in ("ou", "kuramoto"):
            for s in (0, 1):
                old = read_json(sweep / "runs" / f"{b}__s{s}__combined" / "metrics.json")
                new = (runs.get(f"{b}__moments__s{s}") or {}).get("m")
                if not (old and new):
                    L.append(f"{b} seed {s}: missing")
                    continue
                d = 100 * (new["rel_l2_val"] - old["rel_l2_val"])
                L.append(f"{b} seed {s}: val rel L2 frozen {100 * new['rel_l2_val']:.3f}% vs sweep "
                         f"combined {100 * old['rel_l2_val']:.3f}% (n_fourier "
                         f"{new['config']['n_fourier']} vs {old['config']['n_fourier']}): "
                         f"{d:+.3f} points = {d / SWEEP_SIGMA[b]:+.1f} sigma; val W2 "
                         f"{new['w2_mdn_val_mean']:.4f} vs {old['w2_mdn_val_mean']:.4f}")
        L.append("")
    L.append("all benchmarks, five seeds, frozen config; old = paper before task 10")
    for b in ALL7:
        v = over_seeds(runs, b, "moments", "rel_l2_val")[0]
        t = over_seeds(runs, b, "moments", "rel_l2_full")[0]
        w = over_seeds(runs, b, "moments", "w2_mdn_mean")[0]
        nll_o, nll_sd, w_o, w_sd = OLD_LAW[b]
        wm = ms(w)[0]
        flag = ""
        if wm is not None and wm > w_o + 2 * w_sd:
            flag = "  FLAG: W2 worse than old by more than 2 old sd"
        L.append(f"{b:15s} moments val {short(v, 100, 3)}  test {short(t, 100, 3)}  "
                 f"W2 {short(w)} (old {w_o:.4f} +/- {w_sd:.4f}){flag}")
    for (b, f), (o, osd) in OLD_FUNC.items():
        t = over_seeds(runs, b, f, "rel_l2_full")[0]
        tm = ms(t)[0]
        flag = "  FLAG: worse than old by more than 2 old sd" if tm and 100 * tm > o + 2 * osd else ""
        L.append(f"{b:15s} {f:8s} test {short(t, 100, 2)} (old {o:.2f} +/- {osd:.2f}){flag}")
    per = over_seeds(runs, "kuramoto", "moments", "rel_l2_full")[0]
    L.append("")
    L.append("kuramoto moments per seed (old seed-1 outlier 3.06%): " +
             ", ".join(f"{100 * x:.2f}%" for x in per))
    return L


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path)
    p.add_argument("--sweep", type=Path, default=Path("outputs/sweep"))
    p.add_argument("--task5", type=Path, default=Path("outputs/task5"))
    p.add_argument("--old-kuramoto", type=Path, default=Path("outputs/kuramoto_seeds_moments"))
    a = p.parse_args()
    out = a.out
    paper = out / "paper"
    paper.mkdir(parents=True, exist_ok=True)
    task5 = a.task5 if a.task5.exists() else None
    runs = load(out)
    es_new = {b: (runs.get(f"task5__check_{b}") or {}).get("es") for b in PARTICLE}
    es_new = {b: v for b, v in es_new.items() if v}

    # what is missing among the 75
    expected = ([(b, "moments") for b in ALL7] + [(b, "quantile") for b in ONE_D]
                + [(b, "order") for b in ORDER])
    seeds_found = sorted({int(n.rsplit("__s", 1)[1]) for n in runs
                          if "__s" in n and n.rsplit("__s", 1)[1].isdigit()})
    missing = [f"{b}__{f}__s{s}" for b, f in expected for s in SEEDS
               if not (runs.get(f"{b}__{f}__s{s}") or {}).get("m")]

    (paper / "tab_law_main.tex").write_text(law_table(runs))
    (paper / "tab_functional_main.tex").write_text(functional_table(runs, "moments", ALL7))
    (paper / "tab_quantile_main.tex").write_text(functional_table(runs, "quantile", ONE_D))
    (paper / "tab_order_main.tex").write_text(functional_table(runs, "order", ORDER))
    cost_rows, cost_notes, cost_tex = cost(runs, es_new)
    (paper / "tab_cost.tex").write_text(cost_tex)
    fig_notes = figures(runs, paper)
    (paper / "numbers.txt").write_text("\n".join(numbers(runs)) + "\n")
    (paper / "task5_rows.txt").write_text("\n".join(task5_rows(runs, es_new, task5)) + "\n")
    chk = checks(runs, es_new, a.sweep, task5, a.old_kuramoto)
    if out.name.endswith("_smoke"):
        chk = ["smoke run: comparisons against the full-size sweep, task 5 and old runs "
               "are expected to show DIFF", ""] + chk
    chk += ["", "cost runs:"] + ["  " + n for n in cost_notes] + fig_notes
    chk += ["", f"seeds found: {seeds_found}",
            f"missing of the 75: {len(missing)}"] + ["  " + n for n in missing]
    failed = [n for n, r in runs.items() if not r["done"]]
    chk += [f"jobs without .done (failed or running): {len(failed)}"] + ["  " + n for n in failed]
    (paper / "checks.txt").write_text("\n".join(chk) + "\n")
    (paper / "validation_report.txt").write_text("\n".join(validation(runs, a.sweep)) + "\n")

    # one row per run
    keys = ["mdn_val_nll", "w2_mdn_mean", "w2_mdn_val_mean", "w2_floor_mean",
            "reference_resolution", "entropy_floor", "rel_l2_val", "rel_l2_full",
            "rel_l2_empirical", "rel_l2_oracle", "functional_reference_floor",
            "rel_l2_mdn_noise_adjusted", "rel_l2_direct", "rel_l2_fourier", "runtime_sec"]
    with open(out / "task10_runs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "done", "functional", "seed", "simulate_cached", "torch_threads"] + keys)
        for n, r in runs.items():
            m = r["m"] or {}
            cfg, env = m.get("config", {}), m.get("timings", {}).get("environment", {})
            w.writerow([n, r["done"], cfg.get("functional"), cfg.get("seed"),
                        env.get("simulate_cached"), env.get("start", {}).get("torch_threads")]
                       + [m.get(k) if not isinstance(m.get(k), dict) else "" for k in keys])
    with open(paper / "cost.csv", "w", newline="") as fh:
        if cost_rows:
            w = csv.DictWriter(fh, fieldnames=list(cost_rows[0]))
            w.writeheader()
            w.writerows(cost_rows)

    # reda's runs: every number at the top level and the timing stages
    reda = {n: r for n, r in runs.items() if n.startswith("reda__")}
    with open(paper / "reda_runs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        rk = sorted({k for r in reda.values() for k, v in (r["m"] or {}).items()
                     if isinstance(v, (int, float)) and not isinstance(v, bool)})
        w.writerow(["name", "done"] + rk + ["comparisons_timings"])
        for n, r in reda.items():
            m = r["m"] or {}
            w.writerow([n, r["done"]] + [m.get(k) for k in rk]
                       + [json.dumps(m.get("timings", {}).get("comparisons", {}))])

    print("\n".join(chk))
    print(f"\nwritten to {paper}")


if __name__ == "__main__":
    main()
