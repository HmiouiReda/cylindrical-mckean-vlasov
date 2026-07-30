"""Full pipeline: simulation -> features -> MDN -> cylindrical Psi -> eval.

    python main.py --benchmark ou                     # paper config, CPU
    python main.py --benchmark double_well
    python main.py --benchmark ou --quick             # fast smoke test
    python main.py --benchmark ou --single-particle   # one trajectory per scenario
    python main.py --benchmark ou --overfit-test      # gradient-flow check
    python main.py --benchmark ou --sweep-kkeep 1,8,64,200

Outputs in --outdir (default outputs/<benchmark>): fig1..fig5 (PNG) and
metrics.json.  Hyper-parameters are read from configs/<benchmark>.json
(overridable via --config and CLI flags); everything is seeded.
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch

from benchmarks import PARAMS, get_benchmark
from features import (FREQS, FREQS2D, Scaler, build_features,
                      sig_dim)
from mdn import MDN, train_mdn
from cylindrical import (Psi, empirical_fourier,
                         make_psi_inputs, psi_dataset, train_regressor)
from evaluate import (make_baseline_figure,make_figures,relative_l2,
                      sw2_mc_floor,sw2_mdn_vs_ref,w2_mc_floor,w2_mdn_vs_ref)

# Global defaults; per-benchmark values live in configs/<benchmark>.json.
DEFAULT_RUN = dict(M=4000, N=200, sig_level=2, J=5, k_keep=64,
                   mdn_epochs=400, psi_epochs=350, hidden=128,
                   S_quant=2048, batch=256, lr=1e-3, k_x=16)


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", type=str, default="ou",
                   choices=sorted(PARAMS), help="which benchmark to run")
    p.add_argument("--config", type=str, default="",
                   help="JSON config (default: configs/<benchmark>.json)")
    p.add_argument("--quick", action="store_true",
                   help="small configuration for a fast smoke test")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="",
                   help="output directory (default: outputs/<benchmark>)")
    # config overrides (0 / empty = use the config value)
    p.add_argument("--M", type=int, default=0, help="number of scenarios")
    p.add_argument("--N", type=int, default=0, help="particles per scenario")
    p.add_argument("--sig-level", type=int, default=0, choices=[0, 1, 2, 3])
    p.add_argument("--J", type=int, default=0, help="mixture components")
    p.add_argument("--kkeep", type=int, default=0,
                   help="particles per scenario used in the NLL "
                        "(gradient variance only)")
    p.add_argument("--mdn-epochs", type=int, default=0)
    p.add_argument("--psi-epochs", type=int, default=0)
    p.add_argument("--functional", type=str, default="moments",
                   choices=["moments", "quantile"],
                   help="target functional V(x, mu): 'moments' (easy: two "
                        "first moments) or 'quantile' (hard: x*IQR(mu) + "
                        "F_mu(0) + sin x, d = 1 only)")
    p.add_argument("--baselines", type=str, default="full",
                    choices=["full", "emp", "none"],
                    help="V-hat comparison modes: "
                    "'full' evaluates the MDN pipeline, the empirical plug-in, "
                    "and oracle integrals when available; "
                    "'emp' evaluates the MDN pipeline and the empirical plug-in only; "
                    "'none' evaluates only the MDN pipeline.")
    # modes
    p.add_argument("--ito-strat-check", action="store_true",
                   help="three-scheme Ito/Stratonovich drift-correction "
                        "check (multiplicative benchmark), then exit")
    p.add_argument("--single-particle", action="store_true",
                   help="strict single-trajectory training: the MDN sees ONE "
                        "fixed particle per common-noise realization "
                        "(k_keep forced to 1); evaluation is unchanged")
    p.add_argument("--empirical-targets", action="store_true",
                   help="Psi targets from empirical moments of the N "
                        "training particles (no oracle at all)")
    p.add_argument("--overfit-test", action="store_true",
                   help="deliberate-overfit gradient-flow check, then exit")
    p.add_argument("--sweep-kkeep", type=str, default="",
                   help="e.g. 1,8,64,200 -- retrain the MDN for each K")
    p.add_argument("--sweep-only", action="store_true",
                   help="ONLY run the K_keep sweep, then merge metrics.json "
                        "(identical data: same seed)")
    p.add_argument("--sweep-epochs", type=int, default=0,
                   help="epochs per K in the sweep (0 = mdn_epochs // 2)")
    return p.parse_args()


def load_config(args):
    """defaults <- configs/<benchmark>.json <- run_quick (if --quick) <- CLI."""
    path = Path(args.config) if args.config else Path(f"configs/{args.benchmark}.json")
    cfg = json.loads(path.read_text()) if path.exists() else {}
    bench = dict(cfg.get("benchmark", {}))
    run = {**DEFAULT_RUN, **cfg.get("run", {})}
    if args.quick:
        bench_fields = {f.name for f in dataclasses.fields(PARAMS[args.benchmark])}
        for k, v in cfg.get("run_quick", {}).items():
            (bench if k in bench_fields else run)[k] = v
    for cli, key in [("M", "M"), ("N", "N"), ("J", "J"), ("kkeep", "k_keep"),
                     ("sig_level", "sig_level"), ("mdn_epochs", "mdn_epochs"),
                     ("psi_epochs", "psi_epochs")]:
        v = getattr(args, cli)
        if v:
            run[key] = v
    return bench, run


def main():
    args = get_args()
    t0 = time.time()
    outdir = Path(args.outdir) if args.outdir else Path("outputs") / args.benchmark
    outdir.mkdir(parents=True, exist_ok=True)

    bench_params, run = load_config(args)
    if args.single_particle:
        run["k_keep"] = 1
    benchmark = get_benchmark(args.benchmark, **bench_params)
    M, N = run["M"], run["N"]
    torch.set_num_threads(2)   # small tensors => limit CPU contention

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 72)
    print(f"Benchmark: {benchmark.pretty}  ({args.benchmark})")
    print(f"Config   : M={M} scenarios, N={N} particles, "
          f"n={benchmark.p.n_steps} steps, sig level {run['sig_level']}, "
          f"J={run['J']}, K_keep={run['k_keep']}, seed={args.seed}")
    print("=" * 72)


    # ---- optional: Ito-Stratonovich three-scheme check --------------------
    if args.ito_strat_check:
        assert hasattr(benchmark, "ito_stratonovich_check"), \
            "this benchmark has constant sigma0: the correction vanishes"
        Mc = args.M if args.M else 300
        Nc = args.N if args.N else 2000
        print(f"[ito-strat] same increments, three schemes "
              f"(M={Mc} scenarios, N={Nc} particles, "
              f"n={benchmark.p.n_steps} steps)")
        r = benchmark.ito_stratonovich_check(M=Mc, N=Nc, rng=rng)
        print(f"[ito-strat] W2(Ito EM, corrected Heun) = "
              f"{r['gap_corrected_mean']:.4f} +/- {r['gap_corrected_std']:.4f}"
              f"   (MC resolution = {r['mc_resolution']:.4f})")
        print(f"[ito-strat] W2(Ito EM, naive Heun)     = "
              f"{r['gap_naive_mean']:.4f} +/- {r['gap_naive_std']:.4f}")
        ok = (r["gap_naive_mean"]
              > 2.0 * max(r["gap_corrected_mean"], r["mc_resolution"]))
        print(f"[ito-strat] correction is non-trivial and correctly "
              f"handled: {'YES' if ok else 'INCONCLUSIVE'}")
        with open(outdir / "ito_strat_check.json", "w") as f:
            json.dump(r, f, indent=2)
        return

    # ---- simulation -------------------------------------------------------
    t = time.time()
    data = benchmark.simulate(M, N, rng)
    extra = (f", N_ref={benchmark.p.n_ref}"
             if hasattr(benchmark.p, "n_ref") else "")
    print(f"[simulate] {M}x{N} particles{extra} in {time.time() - t:.1f}s")

    # ---- features + split + standardisation -------------------------------
    t = time.time()
    feats = build_features(data, run["sig_level"])
    F = feats.shape[1]
    n_fourier = 2 * len(FREQS2D if benchmark.d == 2 else FREQS)
    print(f"[features] dimension = {F} "
        f"({n_fourier} Fourier + "
        f"{sig_dim(run['sig_level'])} signature) "
        f"in {time.time() - t:.1f}s")

    perm = rng.permutation(M)
    n_tr, n_va = int(0.7 * M), int(0.15 * M)
    itr, iva, ite = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

    scaler = Scaler().fit(feats[itr])              # fit on TRAIN only
    Fs = scaler.transform(feats).astype(np.float32)
    zva = Fs[iva]
    print(f"[features] standardisation check (val): "
          f"max |mean| = {np.abs(zva.mean(0)).max():.3f}, "
          f"mean std = {zva.std(0).mean():.3f}")

    to_t = lambda a: torch.as_tensor(np.ascontiguousarray(a),
                                     dtype=torch.float32)
    F_tr, F_va, F_te = to_t(Fs[itr]), to_t(Fs[iva]), to_t(Fs[ite])
    X_tr, X_va = to_t(data["XT"][itr]), to_t(data["XT"][iva])
    if args.single_particle:
        # Strict version of the paper's training procedure: the learner is
        # given ONE fixed trajectory per common-noise realization, for the
        # whole training (and validation) -- the NLL gradient is unbiased
        # already at a single draw, since minimizing the NLL minimizes the
        # conditional KL.  The N-particle clouds remain in `data` only
        # because (i) they ARE the interacting system generating the
        # dynamics and (ii) evaluation (MC floor, L2 norm of the theorem)
        # needs a reference.  The proposed estimator never touches them.
        X_tr = X_tr[:, :1].contiguous()
        X_va = X_va[:, :1].contiguous()
        print("[data] SINGLE-PARTICLE mode: 1 fixed trajectory per scenario "
              "for MDN training/validation (k_keep = 1); evaluation "
              "unchanged")
    d_tr = benchmark.subset(data, itr)
    d_va = benchmark.subset(data, iva)
    d_te = benchmark.subset(data, ite)
    def model_w2(model, feats_test, data_test):
        """Use exact 1D W2 or sliced W2 according to the state dimension."""
        if benchmark.d == 2:
            return sw2_mdn_vs_ref(model,feats_test,benchmark,
                        data_test,S=run["S_quant"])

        return w2_mdn_vs_ref( model,feats_test,benchmark,
                             data_test,S=run["S_quant"])


    def mc_w2_floor(data_test):
        """Monte-Carlo floor using the metric appropriate to the dimension."""
        if benchmark.d == 2:
            return sw2_mc_floor(benchmark, data_test)
        return w2_mc_floor(benchmark, data_test)

    # ---- K_keep sweep (shared between combined mode and --sweep-only) -----
    def run_sweep():
        res = {}
        ep_sw = (args.sweep_epochs if args.sweep_epochs > 0
                 else max(run["mdn_epochs"] // 2, 100))
        print(f"\n[sweep K_keep] variance/redundancy trade-off "
              f"({ep_sw} epochs per K)")
        for K in [int(s) for s in args.sweep_kkeep.split(",")]:
            torch.manual_seed(args.seed)          # same net init for each K
            m_k = MDN(F, hidden=run["hidden"], J=run["J"],d=benchmark.d)
            train_mdn(m_k, F_tr, X_tr, F_va, X_va, epochs=ep_sw,
                      batch_size=run["batch"], lr=run["lr"], k_keep=K,
                      seed=args.seed, show=False)
            w2k = model_w2(m_k, F_te, d_te)
            res[K] = float(w2k.mean())
            print(f"    K_keep = {K:4d}  ->  W2 test = {w2k.mean():.4f}")
        return res

    if args.sweep_only:
        assert args.sweep_kkeep, "--sweep-only requires --sweep-kkeep"
        sweep_results = run_sweep()
        mfile = outdir / "metrics.json"
        metrics = json.loads(mfile.read_text()) if mfile.exists() else {}
        metrics["sweep_kkeep"] = sweep_results
        mfile.write_text(json.dumps(metrics, indent=2))
        print(f"[sweep] results merged into {mfile} "
              f"({round(time.time() - t0, 1)}s)")
        return

    # ---- optional: deliberate overfit test (gradient-flow sanity) ---------
    if args.overfit_test:
        print("\n[overfit-test] 50 scenarios, 4 FROZEN particles, 800 epochs "
              "-- the train NLL MUST collapse (otherwise: gradient-flow bug).")
        X_fix = X_tr[:50, :4].contiguous()   # FIXED set => memorisation possible
        m_small = MDN(F, hidden=64, J=run["J"], d=benchmark.d)
        h, _ = train_mdn(m_small, F_tr[:50], X_fix, F_tr[:50], X_fix,
                         epochs=800, batch_size=50, lr=1e-3, k_keep=4,
                         seed=args.seed, log_every=200)
        # With 4 frozen atoms, the MDN can shrink sigmas onto them and push
        # the NLL far below the differential-entropy floor of the true law
        # (~0.30 for the OU benchmark, higher for the double well): a train
        # NLL < 0 is impossible without memorisation => gradients flow.
        ok = h["train"][-1] < 0.0
        verdict = ("OK: memorisation confirmed, gradient flow is healthy"
                   if ok else "FAILURE -- debug the training loop!")
        print(f"[overfit-test] train NLL: {h['train'][0]:.3f} -> "
              f"{h['train'][-1]:.3f} ({verdict})")
        return

    # ---- MDN --------------------------------------------------------------
    print("\n[MDN] training "
        "(minimizing expected NLL is equivalent to minimizing conditional KL; "
        "unbiased gradients from a single trajectory)")
    mdn = MDN(F, hidden=run["hidden"], J=run["J"], d=benchmark.d)
    hist_mdn, best_nll = train_mdn(mdn, F_tr, X_tr, F_va, X_va,
                                   epochs=run["mdn_epochs"],
                                   batch_size=run["batch"], lr=run["lr"],
                                   k_keep=run["k_keep"], seed=args.seed)
    mdn.eval()
    floor = benchmark.entropy_floor(data)
    msg = (f"(entropy floor 0.5 log(2 pi e v_T) = {floor:.4f})"
           if floor is not None else "(no closed-form entropy floor)")
    print(f"[MDN] best val NLL = {best_nll:.4f} {msg}")

    # ---- W2 vs Monte-Carlo floor ------------------------------------------
    w2_model = model_w2(mdn, F_te, d_te)
    w2_floor = mc_w2_floor(d_te)

    beats = w2_model.mean() < w2_floor.mean()
    w2_name = "sliced W2" if benchmark.d == 2 else "W2"
    print(f"[{w2_name}] MDN = {w2_model.mean():.4f} +/- {w2_model.std():.4f}   |   "
          f"MC floor (N={N}) = {w2_floor.mean():.4f} +/- {w2_floor.std():.4f}"
          f"   =>  MDN beats the floor: {'YES' if beats else 'NO'}")
    ref_res = benchmark.reference_resolution(d_te)
    if ref_res is not None:
        print(f"[W2] reference resolution (half-sample estimate of the "
              f"reference's own W2 error) = {ref_res:.4f}")

    # ---- optional: K_keep sweep -------------------------------------------
    sweep_results = run_sweep() if args.sweep_kkeep else {}

    # ---- cylindrical network Psi ------------------------------------------
    print("\n[Psi] closed-form <phi, mixture> integration + MSE training "
          "(MDN frozen)")
    freqs_np = FREQS2D if benchmark.d == 2 else FREQS
    freqs_t = torch.as_tensor(freqs_np, dtype=torch.float32)
    assert not (args.functional == "quantile" and benchmark.d > 1), \
        "the quantile functional is defined for d = 1"

    def ref_pair(dd):
        """Per-scenario reference values of the two functionals feeding V."""
        if args.functional == "quantile":
            q = benchmark.ref_quantiles(dd, np.array([0.25, 0.75]))
            return q[1] - q[0], benchmark.ref_cdf0(dd)
        return benchmark.ref_moments(dd)

    def mk(F_s, d, kx):
        f1, f2 = ref_pair(d)
        return psi_dataset(mdn, F_s, d["XT"], f1, f2, kx, freqs_t, rng,
                           empirical_targets=args.empirical_targets,
                           functional=args.functional)

    Xin_tr, y_tr = mk(F_tr, d_tr, run["k_x"])
    Xin_va, y_va = mk(F_va, d_va, run["k_x"])
    Xin_te, y_te = mk(F_te, d_te, 0)          # test: ALL particles

    psi = Psi(n_integrals=2 * len(freqs_np), hidden=run["hidden"],
              x_dim=benchmark.d)
    hist_psi, _ = train_regressor(psi, Xin_tr, y_tr, Xin_va, y_va,
                                  epochs=run["psi_epochs"], batch_size=1024,
                                  lr=run["lr"], seed=args.seed)

    # ---- measure-representation comparisons for V-hat ------------------------
    # The regression target is unchanged. The variants differ only through the
    # representation supplied to the measure channel of Psi:
    #
    #   (1) <phi, mu_hat_N> -> empirical particle plug-in;
    #   (2) MDN pipeline    -> integrals against the predicted mixture;
    #   (3) <phi, mu_T>     -> oracle integrals, when available.
    #
    # --baselines full: empirical + oracle comparisons
    # --baselines emp : empirical comparison only
    # --baselines none: MDN pipeline only
    rel_emp = rel_orc = None

    def variant_sets(ints_fn):
        out = []
        for dd, kx in ((d_tr, run["k_x"]), (d_va, run["k_x"]), (d_te, 0)):
            f1, f2 = ref_pair(dd)
            ints = torch.as_tensor(ints_fn(dd), dtype=torch.float32)
            out.append(make_psi_inputs(ints, dd["XT"], f1, f2, kx, rng,
                                       empirical_targets=args.empirical_targets,
                                       functional=args.functional))
        return out



    if args.baselines in ("full", "emp"):
        print(f"[comparisons] empirical plug-in <phi, mu_hat_N> "
              f"(N={N} particles per scenario)")
        (Xe_tr, ye_tr), (Xe_va, ye_va), (Xe_te, ye_te) = variant_sets(
            lambda dd: empirical_fourier(dd["XT"], freqs_np))
        psi_emp = Psi(n_integrals=2 * len(freqs_np), hidden=run["hidden"],
                      x_dim=benchmark.d)
        train_regressor(psi_emp, Xe_tr, ye_tr, Xe_va, ye_va,
                        epochs=run["psi_epochs"], batch_size=1024,
                        lr=run["lr"], seed=args.seed,show=False)
        rel_emp = relative_l2(psi_emp, Xe_te, ye_te)

    if args.baselines == "full":
        print("[comparisons] oracle integrals <phi, mu_T>")

        def oracle_fourier(dd):
            if benchmark.has_closed_form:
                return benchmark.ref_fourier(dd, freqs_np)
            return empirical_fourier(dd["X_ref"], freqs_np)

        (Xo_tr, yo_tr), (Xo_va, yo_va), (Xo_te, yo_te) = variant_sets(
                                    oracle_fourier)

        psi_orc = Psi(n_integrals=2 * len(freqs_np),
                    hidden=run["hidden"],x_dim=benchmark.d)

        train_regressor(psi_orc,Xo_tr,yo_tr,Xo_va,yo_va,
            epochs=run["psi_epochs"],batch_size=1024,lr=run["lr"],
            seed=args.seed,show=False)
        rel_orc = relative_l2(psi_orc, Xo_te, yo_te)

    rel_full = relative_l2(psi, Xin_te, y_te)
    pct = lambda v: "   --  " if v is None else f"{100 * v:6.2f}%"
    print("[L2] relative error of V-hat (test):")
    print(f"     empirical plug-in (N={N:4d})     : {pct(rel_emp)}")
    print(f"     MDN pipeline (ours)             : {pct(rel_full)}")
    print(f"     oracle integrals                : {pct(rel_orc)}")

    # ---- figures + metrics ------------------------------------------------
    make_figures(outdir, benchmark, d_te, mdn, F_te, hist_mdn, hist_psi,
                 w2_model, w2_floor)
    if args.baselines == "full":
        make_baseline_figure(outdir, benchmark, {
            f"empirical plug-in ($N$={N})": rel_emp,
            "MDN pipeline (ours)": rel_full,
            "oracle integrals": rel_orc,
        })

    _f = lambda v: None if v is None else float(v)
    metrics = dict(
        benchmark=args.benchmark,
        config=dict(**run, seed=args.seed, functional=args.functional,
                    w2_metric=("sliced" if benchmark.d == 2 else "quantile"),
                    single_particle=args.single_particle,
                    empirical_targets=args.empirical_targets,
                    benchmark_params={f.name: getattr(benchmark.p, f.name)
                                      for f in dataclasses.fields(benchmark.p)}),
        mdn_val_nll=float(best_nll),
        entropy_floor=(float(floor) if floor is not None else None),
        w2_mdn_mean=float(w2_model.mean()), w2_mdn_std=float(w2_model.std()),
        w2_floor_mean=float(w2_floor.mean()),
        w2_floor_std=float(w2_floor.std()),
        mdn_beats_floor=bool(beats),
        reference_resolution=(float(ref_res) if ref_res is not None else None),
        rel_l2_full=float(rel_full),
        rel_l2_empirical=_f(rel_emp), rel_l2_oracle=_f(rel_orc),
        sweep_kkeep=sweep_results,
        runtime_sec=round(time.time() - t0, 1),
    )
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print(f"  W2 (conditional law): MDN {w2_model.mean():.4f}  vs  "
          f"MC floor {w2_floor.mean():.4f}   "
          f"({'MDN < floor: OK' if beats else 'MDN >= floor'})")
    if ref_res is not None:
        print(f"  Reference resolution: {ref_res:.4f} "
              f"(lower bound on measurable W2)")
    print(f"empirical plug-in {pct(rel_emp).strip()}  |  "
          f"MDN pipeline {pct(rel_full).strip()}  |  "
          f"oracle {pct(rel_orc).strip()}")
    print(f"  Figures + metrics.json: {outdir}/   "
          f"({metrics['runtime_sec']}s)")
    print("=" * 72)


if __name__ == "__main__":
    main()
