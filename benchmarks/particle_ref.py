"""Shared machinery for benchmarks WITHOUT a closed-form conditional law.

The reference is a large set of N_ref extra particles per scenario, sharing
the same common-noise path.
reference_resolution estimates the reference's own W2 sampling error by a
half-sample split, so every reported W2 can be read relative to it"""

import numpy as np
from .base import Benchmark


class ParticleReferenceBenchmark(Benchmark):
    """Implements the reference interface from a `X_ref` (M, N_ref) array."""

    has_closed_form = False

    # reference conditional law (empirical, N_ref particles)
    def ref_quantiles(self, data: dict, probs: np.ndarray) -> np.ndarray:
        return np.quantile(data["X_ref"], np.asarray(probs), axis=1)  # (S, B)

    def ref_moments(self, data: dict):
        Xr = data["X_ref"]
        return Xr.mean(axis=1), Xr.var(axis=1)

    def ref_density(self, data: dict, i: int, grid: np.ndarray) -> np.ndarray:
        """Gaussian KDE of the reference particles (figures only)."""
        x = data["X_ref"][i]
        n = x.size
        h = max(0.9 * x.std() * n ** (-0.2), 1e-3)    # Scott-type bandwidth
        z = (grid[:, None] - x[None, :]) / h
        return np.exp(-0.5 * z * z).sum(axis=1) / (n * h * np.sqrt(2 * np.pi))

    def ref_label(self) -> str:
        return f"reference KDE ($N_{{\\rm ref}}$ = {self.p.n_ref})"

    def reference_resolution(self, data: dict) -> float:
        """W2 between the two half-sample empirical measures, divided by
        sqrt(2) (the halves carry independent errors of equal size)."""
        Xr = data["X_ref"]
        n2 = Xr.shape[1] // 2
        A = np.sort(Xr[:, :n2], axis=1)
        B = np.sort(Xr[:, n2:2 * n2], axis=1)
        w2 = np.sqrt(((A - B) ** 2).mean(axis=1))     # per scenario
        return float(w2.mean() / np.sqrt(2.0))

    def ref_cdf0(self, data: dict) -> np.ndarray:
        """P_mu(X <= 0) per scenario, from the reference particles."""
        return (data["X_ref"] <= 0.0).mean(axis=1)
