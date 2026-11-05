"""Shared machinery for benchmarks WITHOUT a closed-form conditional law.

The reference is a large set of N_ref extra particles per scenario, sharing
the same common-noise path.
reference_resolution estimates the reference's own W2 sampling error by a
half-sample split, so every reported W2 can be read relative to it"""

import numpy as np
from .base import Benchmark, order_parameters


class ParticleReferenceBenchmark(Benchmark):
    """Reference machinery for benchmarks without a closed form."""
    has_closed_form = False
    @staticmethod
    def _replicates(data: dict) -> tuple[np.ndarray, np.ndarray]:
        """Return the two independent reference replicates."""
        if "X_ref_input" in data and "X_ref_target" in data:
            return data["X_ref_input"], data["X_ref_target"]

        Xr = data["X_ref"]
        n1 = Xr.shape[1] // 2
        return Xr[:, :n1], Xr[:, n1:]

    @classmethod
    def _pooled_reference(cls, data: dict) -> np.ndarray:
        """Pool both references for law-level validation only."""
        A, B = cls._replicates(data)
        return np.concatenate([A, B], axis=1)


    def ref_quantiles(self,data: dict,probs: np.ndarray,) -> np.ndarray:
        Xr = self._pooled_reference(data)
        return np.quantile(Xr, np.asarray(probs), axis=1)

    def ref_moments(self,data: dict) -> tuple[np.ndarray, np.ndarray]:
        Xr = self._pooled_reference(data)
        return Xr.mean(axis=1), Xr.var(axis=1)

    def ref_density(self,data: dict,i: int,grid: np.ndarray,) -> np.ndarray:
        """Gaussian KDE of the pooled reference, for figures only."""
        x = self._pooled_reference(data)[i]
        n = x.size

        h = max(0.9 * x.std() * n ** (-0.2),1e-3)
        z = (grid[:, None] - x[None, :]) / h

        return (np.exp(-0.5 * z * z).sum(axis=1) / (n * h * np.sqrt(2.0 * np.pi)))

    def ref_cdf0(self, data: dict) -> np.ndarray:
        """CDF of the pooled reference, used for law-level descriptions."""
        Xr = self._pooled_reference(data)
        return (Xr <= 0.0).mean(axis=1)

    def ref_label(self) -> str:
        return (f"pooled reference KDE "
                f"($2N_{{\\rm ref}}$ = {2 * self.p.n_ref})")

    def reference_resolution(self, data: dict) -> float:
        """Estimated stochastic W2 error of the pooled reference."""
        A0, B0 = self._replicates(data)
        n = min(A0.shape[1], B0.shape[1])
        A = np.sort(A0[:, :n], axis=1)
        B = np.sort(B0[:, :n], axis=1)
        # Per-scenario W2(A, B).
        w2 = np.sqrt(((A - B) ** 2).mean(axis=1))

        # A and B each have error e.
        # Their distance is approximately sqrt(2) e.
        # Pooling A and B reduces e by another sqrt(2).
        # Therefore pooled error ≈ W2(A,B) / 2.
        return float(w2.mean() / 2.0)


    def oracle_fourier(self,data: dict,freqs) -> np.ndarray:
        """Fourier inputs from reference replicate A only."""
        A, _ = self._replicates(data)
        frequencies = np.asarray(freqs)
        # Convert a 1D cloud (M, N) to (M, N, 1).
        A_vector = A[..., None] if A.ndim == 2 else A

        if frequencies.ndim == 1:
            frequencies = frequencies.reshape(-1, 1)
        else:
            frequencies = frequencies.reshape(-1,A_vector.shape[-1])
        # arg[m,n,p] = <A[m,n], frequency[p]>
        arg = np.einsum("mnd,pd->mnp",A_vector,frequencies)

        sin_integrals = np.sin(arg).mean(axis=1)
        cos_integrals = np.cos(arg).mean(axis=1)

        return np.concatenate([sin_integrals, cos_integrals],axis=1)

    def target_moments(self,data: dict) -> tuple[np.ndarray, np.ndarray]:
        """Regression-label moments from replicate B only."""
        _, B = self._replicates(data)
        if B.ndim == 2:
            # One-dimensional state.
            return B.mean(axis=1), B.var(axis=1)
        # Multidimensional state:
        # mean vector and trace of covariance.
        return (B.mean(axis=1),B.var(axis=1).sum(axis=-1),)

    def target_quantiles(self,data: dict,probs: np.ndarray,) -> np.ndarray:
        """Regression-label quantiles from replicate B only."""
        _, B = self._replicates(data)
        return np.quantile(B, np.asarray(probs), axis=1)

    def target_cdf0(self, data: dict) -> np.ndarray:
        """Regression-label CDF from replicate B only."""
        _, B = self._replicates(data)
        return (B <= 0.0).mean(axis=1)

    def ref_order(self, data: dict, qs=(1, 2)) -> tuple:
        """Order parameters of the pooled reference."""
        Xr = self._pooled_reference(data)
        return tuple(order_parameters(Xr, q) for q in qs)

    def target_order(self, data: dict, qs=(1, 2)) -> tuple:
        """Regression-label order parameters from replicate B only."""
        _, B = self._replicates(data)
        return tuple(order_parameters(B, q) for q in qs)

    def independent_target_pair(self,data: dict,functional: str,):
        """Target statistics built independently from A and B."""
        A, B = self._replicates(data)
        if functional == "moments":
            if A.ndim == 2:
                target_a = (A.mean(axis=1),A.var(axis=1))
                target_b = (B.mean(axis=1),B.var(axis=1))
            else:
                target_a = (A.mean(axis=1),A.var(axis=1).sum(axis=-1),)
                target_b = (B.mean(axis=1),B.var(axis=1).sum(axis=-1))
            return target_a, target_b

        if functional == "quantile":
            if A.ndim != 2:
                raise ValueError("quantile functional is one-dimensional")
            qa = np.quantile(A, [0.25, 0.75], axis=1)
            qb = np.quantile(B, [0.25, 0.75], axis=1)
            target_a = (qa[1] - qa[0],(A <= 0.0).mean(axis=1))
            target_b = (qb[1] - qb[0],(B <= 0.0).mean(axis=1))
            return target_a, target_b

        if functional == "order":
            if A.ndim != 2:
                raise ValueError("order functional is one-dimensional")
            target_a = (order_parameters(A, 1), order_parameters(A, 2))
            target_b = (order_parameters(B, 1), order_parameters(B, 2))
            return target_a, target_b

        raise ValueError(f"unknown functional '{functional}'")