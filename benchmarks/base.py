"""Benchmark interface.

A benchmark provides (i) a conditional particle-system simulator that
returns the training data and (ii) a reference description of the
conditional law  mu_T | W0  used for VALIDATION ONLY:

* closed-form benchmarks (conditional OU): exact conditional moments / 
  quantiles / density;
* nonlinear benchmarks (double well): a large set of N_ref extra particles
  sharing the same common-noise path (conditional propagation of chaos),
  whose empirical law approximates mu_T at rate O(N_ref^{-1/2})
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


def order_parameters(cloud: np.ndarray, q: int = 1) -> np.ndarray:
    """Kuramoto order parameter  r_q(mu) = |int e^{i q y} mu(dy)|  of an
    empirical cloud: (M, N) -> (M,).  Bounded by 1 and q-Lipschitz in W2."""
    s = np.sin(q * cloud).mean(axis=1)
    c = np.cos(q * cloud).mean(axis=1)
    return np.hypot(s, c)


def refine_path(W0: np.ndarray, times: np.ndarray, r: int,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Refine common-noise paths r-fold by a Brownian bridge.

    Inside each coarse step the r fine increments are drawn from their exact
    law given the coarse increment, so the fine path passes through every
    coarse point.  W0: (M, n+1), times: (n+1,) -> (M, n*r+1), (n*r+1,).
    """
    M, n = W0.shape[0], W0.shape[1] - 1
    dt = np.diff(times)                                       # (n,)
    inc = np.diff(W0, axis=1)                                 # (M, n)
    z = rng.standard_normal((M, n, r)) * np.sqrt(dt / r)[None, :, None]
    # iid increments conditioned on their sum: subtract the mean, add inc/r
    z = z - z.mean(axis=2, keepdims=True) + inc[:, :, None] / r
    W0_fine = np.empty((M, n * r + 1))
    W0_fine[:, 0] = W0[:, 0]
    W0_fine[:, 1:] = W0[:, :1] + np.cumsum(z.reshape(M, n * r), axis=1)
    # put the coarse points back exactly, removing cumsum rounding
    W0_fine[:, ::r] = W0
    sub = np.arange(1, r + 1) / r
    times_fine = np.concatenate(
        [times[:1], (times[:-1, None] + dt[:, None] * sub[None, :]).ravel()])
    times_fine[::r] = times
    return W0_fine, times_fine


class Benchmark(ABC):
    name: str = "base"
    pretty: str = "benchmark"  #name for printing
    has_closed_form: bool = False
    d: int = 1                      # state dimension

    # simulation
    @abstractmethod
    def simulate(self, M: int, N: int, rng: np.random.Generator) -> dict:
        """Simulate M common-noise scenarios, each carried by N particles.

        Must return a dict with at least:
          m0, s0 : (M,)      initial-law parameters per scenario
          W0     : (M, n+1)  common-noise trajectories on the time grid
          XT     : (M, N)    terminal particle states (cond. i.i.d.)
          times  : (n+1,)    time grid
        plus benchmark-specific ground-truth entries (mT/vT or X_ref).
        """

    # Remark 15
    def sample_evaluation_points(self, data: dict, k_x: int,
                                rng: np.random.Generator,
                                n_particles: int | None = None) -> np.ndarray:
        """Fresh states conditional on the scenario (mu0, W0).

        Return (M, k_x) in 1D, or (M, k_x, d) in higher dimension.
        Use only scenario parameters and fresh randomness; do not sample
        from XT or the reference clouds. Nonlinear benchmarks simulate a
        separate system with n_particles particles and its own mean field.
        """

    # reference conditional law (validation only)
    @abstractmethod
    def ref_quantiles(self, data: dict, probs: np.ndarray) -> np.ndarray:
        """Quantiles of the reference conditional law: probs (S,) -> (S, B)."""

    @abstractmethod
    def ref_moments(self, data: dict) -> tuple[np.ndarray, np.ndarray]:
        """(mean, variance) of the reference conditional law, each (B,)."""

    @abstractmethod
    def ref_density(self, data: dict, i: int, grid: np.ndarray) -> np.ndarray:
        """Reference density of scenario i on `grid` (figures only)."""

    def ref_fourier(self, data: dict, freqs) -> "np.ndarray":
        """ORACLE integrals <phi, mu_T> against the reference law, for the
        Fourier family phi = {sin(k.), cos(k.)}: closed form when
        available, N_ref empirical otherwise.  -> (B, 2P)."""
        raise NotImplementedError

    # Regression measurement layer.  Closed-form benchmarks can use the same
    # exact law for both the measure-channel input and the regression target.
    # Empirical-reference benchmarks override these methods so that the two
    # quantities are computed from independent reference replicates.
    def oracle_fourier(self, data: dict, freqs) -> "np.ndarray":
        """Best available measure-channel integrals for the reference baseline."""
        return self.ref_fourier(data, freqs)

    def target_quantiles(self, data: dict, probs: np.ndarray) -> np.ndarray:
        """Quantiles used to form regression targets."""
        return self.ref_quantiles(data, probs)

    def target_moments(self, data: dict) -> tuple[np.ndarray, np.ndarray]:
        """Moments used to form regression targets."""
        return self.ref_moments(data)

    def target_cdf0(self, data: dict) -> np.ndarray:
        """CDF value used to form regression targets."""
        return self.ref_cdf0(data)

    def target_order(self, data: dict, qs=(1, 2)) -> tuple:
        """Order parameters used to form regression targets."""
        return self.ref_order(data, qs)

    def independent_target_pair(self, data: dict, functional: str):
        """Two independent estimates of the target statistics, when available.

        Used only to estimate the irreducible reference-noise floor.  Exact-law
        benchmarks return None.
        """
        return None

    def ref_label(self) -> str:
        return "reference law"

    def entropy_floor(self, data: dict) -> float | None:
        """Differential entropy of the true conditional law (NLL floor),
        when available in closed form."""
        return None

    def reference_resolution(self, data: dict) -> float | None:
        """Estimated W2 sampling error of the reference itself (when the
        reference is empirical); None for closed-form benchmarks."""
        return None

    def ref_cdf0(self, data: dict) -> np.ndarray:
        """P_mu(X <= 0) per scenario (generic: invert the quantile grid)."""
        S = 4096
        p = (np.arange(S) + 0.5) / S
        q = self.ref_quantiles(data, p)               # (S, B)
        return (q <= 0.0).mean(axis=0)

    def ref_order(self, data: dict, qs=(1, 2)) -> tuple:
        """Order parameters r_q of the reference law, one array per q
        (generic: invert the quantile grid, as in ref_cdf0)."""
        S = 4096
        p = (np.arange(S) + 0.5) / S
        grid = self.ref_quantiles(data, p)            # (S, B)
        return tuple(np.hypot(np.sin(q * grid).mean(axis=0),
                              np.cos(q * grid).mean(axis=0)) for q in qs)

    # utilities
    @staticmethod
    def subset(data: dict, idx: np.ndarray) -> dict:
        """Index every per-scenario array of the data dict by `idx`."""
        M = data["XT"].shape[0]
        out = {}
        for k, v in data.items():
            if (k != "times" and isinstance(v, np.ndarray)
                    and v.ndim >= 1 and v.shape[0] == M):
                out[k] = v[idx]
            else:
                out[k] = v
        return out
