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
