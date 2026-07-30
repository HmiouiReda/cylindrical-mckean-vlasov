"""Empirical validation and metrics.

* W2 in dimension 1 = L^2 distance between quantile functions:
      W2^2(mu, nu) = int_0^1 ( F_mu^{-1}(u) - F_nu^{-1}(u) )^2 du,
  approximated by the midpoint rule on S quantiles.
* Quantiles of the predicted mixture: DETERMINISTIC inversion of the CDF by
  vectorised bisection (dichotomy)
* "Monte-Carlo floor": W2 between the empirical measure of the N training
  particles of ONE scenario and the reference law.  A healthy MDN does
  BETTER than this floor because it pools information across the M
  scenarios.
* Relative L^2 error of the functional: ||V - V_hat||_2 / ||V||_2 with
  x ~ mu_T."""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.base import Benchmark
from mdn import MDN


# point where quantile integration is evaluated
def midpoint_probs(S: int) -> torch.Tensor:
    return (torch.arange(S, dtype=torch.float32) + 0.5) / S

#  this function is used exclusively for the 1-d benchmarks, where the quantile machinery is exact
def mixture_quantiles(logits: torch.Tensor, mu: torch.Tensor,
                      sigma: torch.Tensor, probs: torch.Tensor,
                      iters: int = 60) -> torch.Tensor:
    """Mixture CDF  F(x) = sum_j pi_j Phi((x - mu_j)/sigma_j), inverted by
    bisection.  logits/mu/sigma: (B, J), probs: (S,) -> quantiles (S, B)."""
    if mu.dim() == 3:                                         # (B, J, 1)
        mu, sigma = mu.squeeze(-1), sigma.squeeze(-1)
    pi = torch.softmax(logits, dim=-1)                        # (B, J)
    S, B = probs.shape[0], mu.shape[0]
    lo = (mu - 12.0 * sigma).min(dim=1).values.expand(S, B).clone()
    hi = (mu + 12.0 * sigma).max(dim=1).values.expand(S, B).clone()
    p = probs.view(S, 1).expand(S, B)
    comp = torch.distributions.Normal(mu.unsqueeze(0), sigma.unsqueeze(0))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        cdf = (pi.unsqueeze(0) * comp.cdf(mid.unsqueeze(-1))).sum(dim=-1)
        gt = cdf > p
        hi = torch.where(gt, mid, hi)
        lo = torch.where(gt, lo, mid)
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# W2: MDN vs reference law, and Monte-Carlo floor
# ---------------------------------------------------------------------------
def w2_mdn_vs_ref(mdn: MDN, feats_std: torch.Tensor, benchmark: Benchmark,
                  data: dict, S: int = 2048) -> np.ndarray:
    """Per-scenario W2 between the predicted mixture and the reference law."""
    probs = midpoint_probs(S)
    with torch.no_grad():
        logits, mu, sigma = mdn(feats_std)
        q_model = mixture_quantiles(logits, mu, sigma, probs).numpy()
    q_ref = benchmark.ref_quantiles(data, probs.numpy())      # (S, B)
    return np.sqrt(((q_model - q_ref) ** 2).mean(axis=0))


def w2_mc_floor(benchmark: Benchmark, data: dict) -> np.ndarray:
    """Per-scenario W2 between the empirical measure of the N training
    particles and the reference law:
    F_emp^{-1}((i - 0.5)/N) = i-th order statistic."""
    XT = data["XT"]
    N = XT.shape[1]
    xs = np.sort(XT, axis=1).T                                # (N, B)
    p = midpoint_probs(N).numpy()
    q_ref = benchmark.ref_quantiles(data, p)                  # (N, B)
    return np.sqrt(((xs - q_ref) ** 2).mean(axis=0))          # (B,)


# ---------------------------------------------------------------------------
# Sliced W2 (d = 2): deterministic half-circle directions; every projected
# law is one-dimensional, so the quantile machinery above applies per
# direction -- with the projected MDN mixture and the projected reference
# BOTH in closed form.
# ---------------------------------------------------------------------------
def slice_directions(K: int = 16) -> np.ndarray:
    ang = np.pi * np.arange(K) / K                 # half circle (W2 is even)
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)          # (K, 2)


def _project_mixture(mu: torch.Tensor, sigma: torch.Tensor,
                     theta: np.ndarray):
    """Diagonal components project to 1-d Gaussians:
    theta.Y_j ~ N(<theta, mu_j>, sum_i theta_i^2 sigma_ji^2)."""
    th = torch.as_tensor(theta, dtype=mu.dtype)
    mu_t = (mu * th).sum(-1)                                     # (B, J)
    sd_t = torch.sqrt(((sigma ** 2) * (th ** 2)).sum(-1))        # (B, J)
    return mu_t, sd_t


def sw2_mdn_vs_ref(mdn: MDN, feats_std: torch.Tensor, benchmark: Benchmark,
                   data: dict, S: int = 1024, K: int = 16) -> np.ndarray:
    """Per-scenario sliced W2 between the predicted mixture and the
    reference: SW2^2 = mean over directions of the projected 1-d W2^2."""
    probs = midpoint_probs(S)
    with torch.no_grad():
        logits, mu, sigma = mdn(feats_std)
        acc = 0.0
        for theta in slice_directions(K):
            mu_t, sd_t = _project_mixture(mu, sigma, theta)
            q_model = mixture_quantiles(logits, mu_t, sd_t, probs).numpy()
            q_ref = benchmark.ref_quantiles(data, probs.numpy(), theta=theta)
            acc = acc + ((q_model - q_ref) ** 2).mean(axis=0)
    return np.sqrt(acc / K)


def sw2_mc_floor(benchmark: Benchmark, data: dict, K: int = 16) -> np.ndarray:
    """Sliced version of the Monte-Carlo floor: project the N training
    particles direction by direction."""
    XT = data["XT"]                                              # (B, N, 2)
    N = XT.shape[1]
    p = midpoint_probs(N).numpy()
    acc = 0.0
    for theta in slice_directions(K):
        xs = np.sort(XT @ theta, axis=1).T                       # (N, B)
        q_ref = benchmark.ref_quantiles(data, p, theta=theta)
        acc = acc + ((xs - q_ref) ** 2).mean(axis=0)
    return np.sqrt(acc / K)


# ---------------------------------------------------------------------------
# Relative L2 error of the functional
# ---------------------------------------------------------------------------
def relative_l2(model: torch.nn.Module, Xin: torch.Tensor,
                y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(Xin)
        num = torch.sqrt(((pred - y) ** 2).mean())
        den = torch.sqrt((y ** 2).mean())
    return (num / den).item()


# ---------------------------------------------------------------------------
# Densities for the figures
# ---------------------------------------------------------------------------
def mixture_pdf(logits, mu, sigma, grid):
    """Mixture density for ONE scenario.  logits/mu/sigma: (J,), grid: (G,)."""
    if mu.dim() == 2:                                         # (J, 1)
        mu, sigma = mu.squeeze(-1), sigma.squeeze(-1)
    pi = torch.softmax(logits, dim=-1).numpy()
    mu, sigma = mu.numpy(), sigma.numpy()
    comp = (np.exp(-0.5 * ((grid[:, None] - mu[None, :]) / sigma[None, :]) ** 2)
            / (sigma[None, :] * np.sqrt(2 * np.pi)))                 # (G, J)
    return (comp * pi[None, :]).sum(axis=1)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(outdir, benchmark: Benchmark, data_test: dict, mdn: MDN,
                 feats_std_test, hist_mdn, hist_psi, w2_model, w2_floor,
                 idx_show=None):
    outdir = str(outdir)
    XT = data_test["XT"]
    m_ref, v_ref = benchmark.ref_moments(data_test)
    ref_lab = benchmark.ref_label()

    if getattr(benchmark, "d", 1) == 2:
        _figures_2d(outdir, benchmark, data_test, mdn, feats_std_test,
                    hist_mdn, hist_psi, w2_model, w2_floor)
        return

    def _grid(i, G=400):
        lo = m_ref[i] - 4 * np.sqrt(v_ref[i])
        hi = m_ref[i] + 4 * np.sqrt(v_ref[i])
        return np.linspace(lo, hi, G)

    # ---- Fig 1: sanity check -- particle histogram vs reference law -------
    i0 = int(np.argsort(m_ref)[len(m_ref) // 2])
    grid = _grid(i0)
    plt.figure(figsize=(6, 4))
    plt.hist(XT[i0], bins=40, density=True, alpha=0.45, color="tab:gray",
             label=f"simulated particles (N = {XT.shape[1]})")
    plt.plot(grid, benchmark.ref_density(data_test, i0, grid), "k-", lw=2,
             label=ref_lab)
    plt.title(f"{benchmark.pretty}: simulation vs reference (1 scenario)")
    plt.xlabel("x"); plt.ylabel("density"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{outdir}/fig1_benchmark_density.png", dpi=130); plt.close()

    # ---- Fig 2: MDN vs reference vs histogram, 3 test scenarios -----------
    if idx_show is None:
        order = np.argsort(m_ref)
        idx_show = [int(order[int(f * (len(order) - 1))])
                    for f in (0.1, 0.5, 0.9)]
    with torch.no_grad():
        logits, mu, sigma = mdn(feats_std_test)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=False)
    for ax, i in zip(axes, idx_show):
        g = _grid(i)
        ax.hist(XT[i], bins=35, density=True, alpha=0.35, color="tab:gray",
                label="particles")
        ax.plot(g, benchmark.ref_density(data_test, i, g), "k-", lw=2,
                label="reference")
        ax.plot(g, mixture_pdf(logits[i], mu[i], sigma[i], g),
                "r--", lw=2, label="MDN")
        ax.set_title(f"test scenario #{i}")
        ax.set_xlabel("x")
    axes[0].set_ylabel("density"); axes[0].legend()
    fig.suptitle(f"{benchmark.pretty}: conditional law -- "
                 "MDN mixture vs reference")
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig2_mdn_vs_ref.png", dpi=130); plt.close(fig)

    # ---- Fig 3: training curves -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(hist_mdn["train"], label="train")
    axes[0].plot(hist_mdn["val"], label="val")
    axes[0].set_title("MDN -- NLL"); axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].semilogy(hist_psi["train"], label="train")
    axes[1].semilogy(hist_psi["val"], label="val")
    axes[1].set_title(r"$\Psi$ -- MSE (log)"); axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig3_training_curves.png", dpi=130); plt.close(fig)

    # ---- Fig 4: per-scenario W2 -- MDN vs Monte-Carlo floor ---------------
    plt.figure(figsize=(6.5, 4))
    bins = np.linspace(0, max(w2_floor.max(), w2_model.max()) * 1.05, 45)
    plt.hist(w2_floor, bins=bins, alpha=0.55,
             label=f"MC floor (mean {w2_floor.mean():.4f})")
    plt.hist(w2_model, bins=bins, alpha=0.55,
             label=f"MDN (mean {w2_model.mean():.4f})")
    plt.axvline(w2_floor.mean(), color="tab:blue", ls="--")
    plt.axvline(w2_model.mean(), color="tab:orange", ls="--")
    plt.title(r"$\mathcal{W}_2$ to the reference law, per test scenario")
    plt.xlabel(r"$\mathcal{W}_2$"); plt.ylabel("# scenarios")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{outdir}/fig4_w2_floor.png", dpi=130); plt.close()



def _gauss_ellipse(ax, m, V, nsig, **kw):
    """1- and 2-sigma ellipse of N_2(m, V)."""
    w, U = np.linalg.eigh(V)
    t = np.linspace(0, 2 * np.pi, 120)
    circ = np.stack([np.cos(t), np.sin(t)])
    pts = m[:, None] + nsig * U @ (np.sqrt(w)[:, None] * circ)
    ax.plot(pts[0], pts[1], **kw)


def _figures_2d(outdir, benchmark, data_test, mdn, feats_std_test,
                hist_mdn, hist_psi, w2_model, w2_floor):
    """Figures for the two-dimensional benchmark: particle clouds, 
    exact Gaussian ellipses, MDN component ellipses, training curves
    and sliced-Wasserstein histograms."""
    XT, mT, VT = data_test["XT"], data_test["mT"], data_test["VT"]
    order = np.argsort(mT[:, 0])
    idx_show = [int(order[int(f * (len(order) - 1))]) for f in (0.1, 0.5, 0.9)]
    with torch.no_grad():
        logits, mu, sigma = mdn(feats_std_test)
        pi = torch.softmax(logits, dim=-1).numpy()
    mu_n, sg_n = mu.numpy(), sigma.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, i in zip(axes, idx_show):
        ax.scatter(XT[i, :, 0], XT[i, :, 1], s=6, alpha=0.3,
                   color="tab:gray", label="particles")
        for ns in (1.0, 2.0):
            _gauss_ellipse(ax, mT[i], VT[i], ns, color="k", lw=2,
                           label="exact" if ns == 1.0 else None)
        for j in range(mu_n.shape[1]):
            if pi[i, j] < 0.02:
                continue
            _gauss_ellipse(ax, mu_n[i, j], np.diag(sg_n[i, j] ** 2), 1.0,
                           color="tab:red", ls="--", lw=1.5,
                           alpha=min(1.0, 0.25 + 2.5 * pi[i, j]),
                           label="MDN components" if j == 0 else None)
        ax.set_title(f"test scenario #{i}")
        ax.set_xlabel(r"$x_1$")
        ax.set_aspect("equal")
    axes[0].set_ylabel(r"$x_2$")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"{benchmark.pretty}: exact 1-2 sigma ellipses vs "
                 "MDN mixture components (1 sigma, opacity = weight)")
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig2_mdn_vs_ref.png", dpi=130)
    plt.close(fig)

    # fig1: single-scenario cloud + ellipses (sanity)
    i0 = idx_show[1]
    plt.figure(figsize=(5.4, 5))
    plt.scatter(XT[i0, :, 0], XT[i0, :, 1], s=6, alpha=0.35,
                color="tab:gray", label=f"particles (N = {XT.shape[1]})")
    for ns in (1.0, 2.0):
        _gauss_ellipse(plt.gca(), mT[i0], VT[i0], ns, color="k", lw=2,
                       label=benchmark.ref_label() if ns == 1.0 else None)
    plt.gca().set_aspect("equal")
    plt.xlabel(r"$x_1$"); plt.ylabel(r"$x_2$"); plt.legend(fontsize=8)
    plt.title(f"{benchmark.pretty}: simulation vs reference (1 scenario)")
    plt.tight_layout()
    plt.savefig(f"{outdir}/fig1_benchmark_density.png", dpi=130)
    plt.close()

    # fig3 (training curves) and fig4 (SW2 histogram): dimension-agnostic
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(hist_mdn["train"], label="train")
    axes[0].plot(hist_mdn["val"], label="val")
    axes[0].set_title("MDN -- NLL"); axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].semilogy(hist_psi["train"], label="train")
    axes[1].semilogy(hist_psi["val"], label="val")
    axes[1].set_title(r"$\Psi$ -- MSE (log)"); axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig3_training_curves.png", dpi=130)
    plt.close(fig)

    plt.figure(figsize=(6.5, 4))
    bins = np.linspace(0, max(w2_floor.max(), w2_model.max()) * 1.05, 45)
    plt.hist(w2_floor, bins=bins, alpha=0.55,
             label=f"MC floor (mean {w2_floor.mean():.4f})")
    plt.hist(w2_model, bins=bins, alpha=0.55,
             label=f"MDN (mean {w2_model.mean():.4f})")
    plt.axvline(w2_floor.mean(), color="tab:blue", ls="--")
    plt.axvline(w2_model.mean(), color="tab:orange", ls="--")
    plt.title(r"sliced $\mathcal{W}_2$ to the reference law, per scenario")
    plt.xlabel(r"$S\mathcal{W}_2$"); plt.ylabel("# scenarios")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{outdir}/fig4_w2_floor.png", dpi=130)
    plt.close()




def make_baseline_figure(outdir, benchmark: Benchmark, rel_errors: dict):
    """Fig 5: the V-hat baseline ladder (relative L2 error, log scale).

    rel_errors: ordered dict-like {label: relative L2 error} -- typically
    empirical plug-in, MDN pipeline, oracle integrals."""
    labels = list(rel_errors.keys())
    vals = [100.0 * v for v in rel_errors.values()]
    colors = ["tab:gray", "tab:blue", "tab:red", "tab:green"][:len(labels)]
    plt.figure(figsize=(7.2, 3.4))
    ypos = np.arange(len(labels))[::-1]
    bars = plt.barh(ypos, vals, color=colors, alpha=0.85)
    plt.yticks(ypos, labels)
    plt.xscale("log")
    for b, v in zip(bars, vals):
        plt.text(v * 1.12, b.get_y() + b.get_height() / 2, f"{v:.2f}%",
                 va="center", fontsize=9)
    plt.xlim(right=max(vals) * 4.0)
    plt.xlabel(r"relative $L^2$ error of $\widehat V$ (%, log scale)")
    plt.title(f"{benchmark.pretty}: what feeds the measure channel of "
              r"$\Psi$")
    plt.tight_layout()
    plt.savefig(f"{outdir}/fig5_baselines.png", dpi=130)
    plt.close()
