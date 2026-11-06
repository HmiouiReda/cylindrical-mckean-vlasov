"""Error sources and a high-fidelity check of the particle references (task 5).

    python main.py --benchmark double_well --save-model --outdir outputs/dw
    python error_sources.py --run outputs/dw
    python error_sources.py --kl-only --benchmark ou2d

For the saved test scenarios of a particle-reference benchmark, each coarse
common-noise path is refined by a Brownian bridge and a high-fidelity system
H is simulated on it (default 2 x 16,000 particles, 8x finer grid).  Every
error is then a W2 distance to H, or between two simulated laws:

  reference      pooled A+B (published)            vs H
  ref_fine       2 x n_ref particles, fine grid     vs H   (discretization)
  hi_coarse      2 x n_hi/2 particles, coarse grid  vs H   (discretization)
  train_coarse   copies x N, coarse grid            vs H   (particle + disc.)
  train_fine     copies x N, fine grid              vs H   (particle alone)
  plugin_N'      one system of N', coarse grid      vs H   (accuracy of a plug-in)
  mixture_J      EM fit of a J-mixture to half of H vs the other half (representation)
  mdn            the saved network                  vs H and vs A+B

W2 between finite clouds is bias plus sampling noise.  Each system is two
independent halves, so its noise is W2(half 1, half 2) / 2, and the bias is
reported as sqrt(max(mean W2^2 - noise_X^2 - noise_H^2, 0)).

--kl-only fits a diagonal J-mixture to the exact law of a closed-form
benchmark (ou2d) and reports the KL divergence; no simulation of H.

Writes <run>/error_sources.json and prints one table.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from benchmarks import get_benchmark
from benchmarks.base import order_parameters, refine_path
from cylindrical import V_target, V_target_order, V_target_quantile
from evaluate import midpoint_probs, mixture_quantiles
from mdn import MDN

LOG_SIGMA_MIN = -5.0          # same floor as the MDN (mdn.py)


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=str, default="",
                   help="run directory written by main.py --save-model")
    p.add_argument("--refine", type=int, default=8, help="grid refinement factor")
    p.add_argument("--n-hi", type=int, default=32000,
                   help="particles in H, split into two independent halves")
    p.add_argument("--copies", type=int, default=50,
                   help="independent N-particle systems pooled for the training law")
    p.add_argument("--plugin-sizes", type=str, default="200,800,2000,8000")
    p.add_argument("--J-list", type=str, default="1,2,3,4,5,6,7,8")
    p.add_argument("--k", type=int, default=0,
                   help="use only the first k saved scenarios (0 = all)")
    p.add_argument("--em-iters", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kl-only", action="store_true",
                   help="closed-form benchmark: KL of the best diagonal mixture")
    p.add_argument("--benchmark", type=str, default="ou2d",
                   help="benchmark for --kl-only")
    p.add_argument("--kl-samples", type=int, default=100000,
                   help="EM fitting noise adds ~ params / (2 n) nats to the KL")
    p.add_argument("--out", type=str, default="")
    return p.parse_args()


# ---------------------------------------------------------------------------
# W2 and noise
# ---------------------------------------------------------------------------
def w2_clouds(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Exact per-row W2 between two 1-d empirical measures, (K, n), (K, m)."""
    xs, ys = np.sort(X, axis=1), np.sort(Y, axis=1)
    n, m = xs.shape[1], ys.shape[1]
    edges = np.union1d(np.linspace(0.0, 1.0, n + 1), np.linspace(0.0, 1.0, m + 1))
    mids, w = 0.5 * (edges[:-1] + edges[1:]), np.diff(edges)
    ix = np.minimum((mids * n).astype(int), n - 1)
    iy = np.minimum((mids * m).astype(int), m - 1)
    return np.sqrt((((xs[:, ix] - ys[:, iy]) ** 2) * w).sum(axis=1))


def w2_quantiles_vs_cloud(q: np.ndarray, X: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """W2 between a law given by quantiles q (S, K) and a cloud X (K, n), by
    the midpoint rule, as evaluate.w2_mdn_vs_ref does."""
    qx = np.quantile(X, probs, axis=1)                        # (S, K)
    return np.sqrt(((q - qx) ** 2).mean(axis=0))


class System:
    """A cloud made of two independent halves on the same paths."""

    def __init__(self, a: np.ndarray, b: np.ndarray, seconds: float, steps: int):
        self.a, self.b = a, b
        self.cloud = np.concatenate([a, b], axis=1)
        self.seconds = seconds
        self.particle_steps = self.cloud.size * steps

    def noise2(self) -> np.ndarray:
        """Per-row squared sampling noise of the pooled cloud."""
        return (w2_clouds(self.a, self.b) / 2.0) ** 2


def bias_summary(w2: np.ndarray, noise2_x, noise2_h) -> dict:
    """Raw W2 and the noise-removed bias, all as rms over scenarios."""
    raw2 = float((w2 ** 2).mean())
    n2x = 0.0 if noise2_x is None else float(np.mean(noise2_x))
    n2h = float(np.mean(noise2_h))
    return dict(w2_mean=float(w2.mean()), w2_rms=float(np.sqrt(raw2)),
                noise_x_rms=float(np.sqrt(n2x)), noise_h_rms=float(np.sqrt(n2h)),
                bias_rms=float(np.sqrt(max(raw2 - n2x - n2h, 0.0))),
                resolved=bool(raw2 - n2x - n2h > n2x + n2h))


# ---------------------------------------------------------------------------
# Simulation on given paths
# ---------------------------------------------------------------------------
def simulate_on(bench, scen: dict, W0, times, n: int, copies: int,
                rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """`copies` independent n-particle systems per scenario on the paths W0,
    pooled: (K, copies * n), and the wall-clock seconds."""
    K = len(scen["m0"])
    data = {"m0": np.tile(scen["m0"], copies), "s0": np.tile(scen["s0"], copies),
            "W0": np.tile(W0, (copies, 1)), "times": times}
    t = time.perf_counter()
    X = bench.sample_evaluation_points(data, n, rng, n_particles=n)
    seconds = time.perf_counter() - t
    X = X.reshape(copies, K, n).transpose(1, 0, 2).reshape(K, copies * n)
    return X, seconds


def system(bench, scen, W0, times, n_total, copies, rng) -> System:
    """Two independent halves, each `copies` systems of n_total / (2 copies)."""
    n = n_total // (2 * copies)
    a, sa = simulate_on(bench, scen, W0, times, n, copies, rng)
    b, sb = simulate_on(bench, scen, W0, times, n, copies, rng)
    return System(a, b, sa + sb, len(times) - 1)


# ---------------------------------------------------------------------------
# Gaussian mixtures by EM, diagonal, vectorized over scenarios
# ---------------------------------------------------------------------------
def em_mixture(X: np.ndarray, J: int, iters: int = 300, chunk: int = 10):
    """Fit a J-component diagonal Gaussian mixture to each row of X,
    (K, n) or (K, n, d).  Returns weights (K, J), means and sigmas (K, J, d).
    Initialised at quantiles of the first coordinate; sigma floored at the
    MDN's own floor.  Maximum likelihood, so in W2 an upper bound on the
    best mixture."""
    X = X[..., None] if X.ndim == 2 else X
    K, n, d = X.shape
    pis, mus, sigs = [], [], []
    floor2 = np.exp(2 * LOG_SIGMA_MIN)
    for s in range(0, K, chunk):
        x = X[s:s + chunk]                                    # (k, n, d)
        k = x.shape[0]
        order = np.argsort(x[..., 0], axis=1)
        idx = ((np.arange(J) + 0.5) / J * n).astype(int)
        mu = np.take_along_axis(x, order[:, idx, None], axis=1)  # (k, J, d)
        var = np.repeat(x.var(axis=1, keepdims=True) / J ** 2 if J > 1
                        else x.var(axis=1, keepdims=True), J, axis=1)
        pi = np.full((k, J), 1.0 / J)
        prev = -np.inf
        for _ in range(iters):
            diff = x[:, :, None, :] - mu[:, None, :, :]       # (k, n, J, d)
            logp = (np.log(pi)[:, None, :]
                    - 0.5 * (np.log(2 * np.pi * var)[:, None, :, :]
                             + diff ** 2 / var[:, None, :, :]).sum(-1))
            mx = logp.max(axis=2, keepdims=True)
            lse = mx + np.log(np.exp(logp - mx).sum(axis=2, keepdims=True))
            r = np.exp(logp - lse)                            # (k, n, J)
            nk = r.sum(axis=1) + 1e-12                        # (k, J)
            pi = nk / n
            mu = np.einsum("knj,knd->kjd", r, x) / nk[..., None]
            diff = x[:, :, None, :] - mu[:, None, :, :]
            var = np.maximum(np.einsum("knj,knjd->kjd", r, diff ** 2)
                             / nk[..., None], floor2)
            ll = float(lse.mean())
            if ll - prev < 1e-7:
                break
            prev = ll
        pis.append(pi), mus.append(mu), sigs.append(np.sqrt(var))
    return np.concatenate(pis), np.concatenate(mus), np.concatenate(sigs)


def mixture_logpdf(x, pi, mu, sig):
    """log density of a diagonal mixture at x (K, n, d)."""
    diff = x[:, :, None, :] - mu[:, None, :, :]
    logp = (np.log(pi)[:, None, :]
            - 0.5 * (np.log(2 * np.pi * sig ** 2)[:, None, :, :]
                     + diff ** 2 / sig[:, None, :, :] ** 2).sum(-1))
    mx = logp.max(axis=2, keepdims=True)
    return (mx + np.log(np.exp(logp - mx).sum(axis=2, keepdims=True)))[..., 0]


def mixture_q(pi, mu, sig, probs) -> np.ndarray:
    """Quantiles (S, K) of a 1-d mixture, by the MDN's own bisection."""
    t = lambda a: torch.as_tensor(a, dtype=torch.float64)
    return mixture_quantiles(t(np.log(pi)), t(mu[..., 0]), t(sig[..., 0]),
                             t(probs)).numpy()


# ---------------------------------------------------------------------------
# Functional labels
# ---------------------------------------------------------------------------
def stats(X: np.ndarray, functional: str):
    if functional == "moments":
        return X.mean(axis=1), X.var(axis=1)
    if functional == "quantile":
        q = np.quantile(X, [0.25, 0.75], axis=1)
        return q[1] - q[0], (X <= 0.0).mean(axis=1)
    return order_parameters(X, 1), order_parameters(X, 2)


V_FUNCTIONS = {"moments": V_target, "quantile": V_target_quantile,
               "order": V_target_order}


def label_bias(A, B, H: System, x: np.ndarray) -> dict:
    """Bias of the reference-B labels against labels from H, as a fraction
    of ||V||, for each functional.  x: (K, k) evaluation points."""
    out = {}
    for name, V in V_FUNCTIONS.items():
        y = lambda X: V(x, *(s[:, None] for s in stats(X, name)))
        yH, yHa, yHb, yA, yB = y(H.cloud), y(H.a), y(H.b), y(A), y(B)
        norm = np.sqrt((yH ** 2).mean())
        gap2 = ((yB - yH) ** 2).mean()
        noise_b2 = 0.5 * ((yA - yB) ** 2).mean()               # one replicate
        noise_h2 = 0.25 * ((yHa - yHb) ** 2).mean()            # pooled halves
        out[name] = dict(gap=float(np.sqrt(gap2) / norm),
                         label_noise=float(np.sqrt(noise_b2) / norm),
                         bias=float(np.sqrt(max(gap2 - noise_b2 - noise_h2, 0.0)) / norm))
    return out


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------
def load_run(run_dir: Path, k: int):
    meta = json.loads((run_dir / "model" / "meta.json").read_text())
    z = np.load(run_dir / "model" / "scenarios.npz")
    split = {"test": {}, "train": {}}
    for key in z.files:
        s, name = key.split("__", 1)
        split[s][name] = z[key]
    for s in split.values():
        for name, v in s.items():
            if name != "times" and k:
                s[name] = v[:k]
    mdn = MDN(**meta["mdn"])
    mdn.load_state_dict(torch.load(run_dir / "model" / "mdn.pt"))
    mdn.eval()
    return meta, split, mdn


def mdn_quantiles(mdn, feats, probs) -> np.ndarray:
    with torch.no_grad():
        logits, mu, sigma = mdn(torch.as_tensor(feats))
        return mixture_quantiles(logits, mu, sigma, torch.as_tensor(probs)).numpy()


def run_check(args):
    run_dir = Path(args.run)
    meta, split, mdn = load_run(run_dir, args.k)
    bench = get_benchmark(meta["benchmark"], **meta["benchmark_params"])
    if bench.has_closed_form:
        raise SystemExit("closed-form benchmark: use --kl-only")
    te, tr = split["test"], split["train"]
    K = len(te["m0"])
    rng = np.random.default_rng([args.seed, 5])
    probs = midpoint_probs(meta["S_quant"]).numpy()
    n_ref, N = int(meta["benchmark_params"]["n_ref"]), int(meta["N"])
    W0, times = te["W0"], te["times"]
    W0f, timesf = refine_path(W0, times, args.refine, rng)
    print(f"[error-sources] {meta['benchmark']}: K={K} scenarios, "
          f"{len(times) - 1} -> {len(timesf) - 1} steps, H = {args.n_hi} particles")

    sims, t0 = {}, time.perf_counter()
    plan = [("H", W0f, timesf, args.n_hi, 1),
            ("ref_fine", W0f, timesf, 2 * n_ref, 1),
            ("hi_coarse", W0, times, args.n_hi, 1),
            ("train_coarse", W0, times, 2 * (args.copies // 2) * N, args.copies // 2),
            ("train_fine", W0f, timesf, 2 * (args.copies // 2) * N, args.copies // 2)]
    for name, w, t, n_total, copies in plan:
        sims[name] = system(bench, te, w, t, n_total, copies, rng)
        print(f"    {name:13s} {sims[name].cloud.shape[1]:6d} particles, "
              f"{len(t) - 1:5d} steps  {sims[name].seconds:7.1f}s")
    plugins = {}
    for n in [int(s) for s in args.plugin_sizes.split(",") if s]:
        X, sec = simulate_on(bench, te, W0, times, n, 1, rng)
        plugins[n] = (X, sec, X.size * (len(times) - 1))
        print(f"    plugin_{n:<6d} {n:6d} particles, {len(times) - 1:5d} steps  {sec:7.1f}s")
    H = sims["H"]
    n2h = H.noise2()

    # reference, and the simulated systems, against H
    A, B = te["X_ref_input"], te["X_ref_target"]
    ref = System(A, B, 0.0, len(times) - 1)
    out = {"reference": bias_summary(w2_clouds(ref.cloud, H.cloud), ref.noise2(), n2h)}
    res_stated = float((w2_clouds(A, B) / 2.0).mean())
    out["reference"]["stated_resolution"] = res_stated
    out["reference"]["error_rms"] = float(np.sqrt(out["reference"]["bias_rms"] ** 2
                                                  + np.mean(ref.noise2())))
    for name in ("ref_fine", "hi_coarse", "train_coarse", "train_fine"):
        S = sims[name]
        out[name] = bias_summary(w2_clouds(S.cloud, H.cloud), S.noise2(), n2h)
    for n, (X, _, _) in plugins.items():
        out[f"plugin_{n}"] = bias_summary(w2_clouds(X, H.cloud), None, n2h)

    # representation: fit on one half of H, measure against the other, so
    # the fit cannot absorb the noise it is measured against
    out["mixture"] = {}
    for J in [int(s) for s in args.J_list.split(",") if s]:
        t = time.perf_counter()
        pi, mu, sig = em_mixture(H.a, J, iters=args.em_iters)
        w2 = w2_quantiles_vs_cloud(mixture_q(pi, mu, sig, probs), H.b, probs)
        out["mixture"][J] = bias_summary(w2, None, 2.0 * n2h)
        print(f"    mixture J={J}  W2 {w2.mean():.4f}  ({time.perf_counter() - t:.0f}s)")

    # the network: against H, and against the reference on test and train
    q_te = mdn_quantiles(mdn, te["feats"], probs)
    w2_ref_te = w2_quantiles_vs_cloud(q_te, ref.cloud, probs)
    q_tr = mdn_quantiles(mdn, tr["feats"], probs)
    ref_tr = System(tr["X_ref_input"], tr["X_ref_target"], 0.0, len(times) - 1)
    w2_ref_tr = w2_quantiles_vs_cloud(q_tr, ref_tr.cloud, probs)
    out["mdn"] = {
        "vs_H": bias_summary(w2_quantiles_vs_cloud(q_te, H.cloud, probs), None, n2h),
        "vs_reference_test": bias_summary(w2_ref_te, None, ref.noise2()),
        "vs_reference_train": bias_summary(w2_ref_tr, None, ref_tr.noise2()),
        "stored_w2_mean": float(te["w2_mdn"].mean()),
        "recomputed_w2_mean": float(w2_ref_te.mean())}

    # functional labels, at 200 points of H
    x = H.b[:, :200]
    out["labels"] = label_bias(A, B, H, x)

    # costs, for R5.6 and task 8
    cost = {name: dict(seconds=S.seconds, particle_steps=S.particle_steps,
                       us_per_particle_step=1e6 * S.seconds / S.particle_steps)
            for name, S in sims.items()}
    for n, (_, sec, ps) in plugins.items():
        cost[f"plugin_{n}"] = dict(seconds=sec, particle_steps=ps,
                                   us_per_particle_step=1e6 * sec / ps)
    out["cost"] = cost
    out["config"] = dict(benchmark=meta["benchmark"], K=K, refine=args.refine,
                         n_hi=args.n_hi, copies=2 * (args.copies // 2), N=N,
                         n_ref=n_ref, n_steps=len(times) - 1, seed=args.seed,
                         run=str(run_dir), total_seconds=time.perf_counter() - t0)
    report(out)
    return out


def report(out: dict):
    c = out["config"]
    print("\n" + "=" * 72)
    print(f"{c['benchmark']}: W2 to H (rms over {c['K']} scenarios); "
          f"bias = noise removed")
    print(f"  {'':22s} {'raw':>8s} {'noise X':>8s} {'noise H':>8s} {'bias':>8s}")
    rows = [("reference A+B", out["reference"]), ("ref, fine grid", out["ref_fine"]),
            ("H size, coarse grid", out["hi_coarse"]),
            ("train law, coarse", out["train_coarse"]),
            ("train law, fine", out["train_fine"]),
            ("MDN", out["mdn"]["vs_H"])]
    rows += [(f"plug-in N'={k[7:]}", v) for k, v in out.items() if k.startswith("plugin_")]
    rows += [(f"mixture J={J}", v) for J, v in out["mixture"].items()]
    for label, r in rows:
        star = "" if r["resolved"] else "  (not resolved)"
        print(f"  {label:22s} {r['w2_rms']:8.4f} {r['noise_x_rms']:8.4f} "
              f"{r['noise_h_rms']:8.4f} {r['bias_rms']:8.4f}{star}")
    r = out["reference"]
    flag = r["error_rms"] > 1.25 * r["stated_resolution"]
    print(f"  reference error {r['error_rms']:.4f} vs stated resolution "
          f"{r['stated_resolution']:.4f}  ->  {'TOO SMALL' if flag else 'holds'}")
    m = out["mdn"]
    print(f"  MDN vs reference: test {m['vs_reference_test']['w2_rms']:.4f}, "
          f"train {m['vs_reference_train']['w2_rms']:.4f} "
          f"(stored mean {m['stored_w2_mean']:.4f}, recomputed {m['recomputed_w2_mean']:.4f})")
    print("  labels (fraction of ||V||): " + ", ".join(
        f"{k} bias {v['bias']:.4f} / noise {v['label_noise']:.4f}"
        for k, v in out["labels"].items()))
    print("=" * 72)


def run_kl(args):
    """KL(exact law || best diagonal J-mixture) on a closed-form benchmark."""
    bench = get_benchmark(args.benchmark)
    rng = np.random.default_rng([args.seed, 5])
    K = args.k or 100
    data = bench.simulate(K, 2, rng)
    out = {}
    for J in [int(s) for s in args.J_list.split(",") if s]:
        x = bench.sample_evaluation_points(data, args.kl_samples, rng)
        pi, mu, sig = em_mixture(x, J, iters=args.em_iters)
        y = bench.sample_evaluation_points(data, args.kl_samples, rng)
        y = y[..., None] if y.ndim == 2 else y
        chol = np.linalg.cholesky(data["VT"])
        z = np.linalg.solve(chol[:, None], (y - data["mT"][:, None, :])[..., None])[..., 0]
        logp = (-0.5 * (z ** 2).sum(-1) - np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(-1)[:, None]
                - 0.5 * y.shape[-1] * np.log(2 * np.pi))
        kl = (logp - mixture_logpdf(y, pi, mu, sig)).mean(axis=1)
        out[J] = dict(kl_mean=float(kl.mean()), kl_std=float(kl.std()))
        print(f"[kl] {args.benchmark} J={J}: KL = {kl.mean():.5f} +/- {kl.std():.5f} nats")
    return {"kl": out, "config": dict(benchmark=args.benchmark, K=K,
                                      samples=args.kl_samples, seed=args.seed)}


def main():
    args = get_args()
    if args.kl_only:
        out = run_kl(args)
        path = Path(args.out) if args.out else Path("outputs") / f"{args.benchmark}_kl.json"
    else:
        out = run_check(args)
        path = Path(args.out) if args.out else Path(args.run) / "error_sources.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
