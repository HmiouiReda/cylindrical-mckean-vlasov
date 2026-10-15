"""Conditional Ornstein-Uhlenbeck in R^2
    dX_t = a ( E[X_t | F^{W0}_t] - X_t ) dt + L dB_t + sigma0_vec dW0_t,
    X_t in R^2,  W0 scalar,  X_0 ~ N(m0, diag(s0^2)),
    L @ L^T = C = [[sig1^2, rho sig1 sig2], [rho sig1 sig2, sig2^2]].

Exact conditional law:
    m_T = m0 + sigma0_vec W0_T                       in R^2,
    V_T = diag(s0^2) e^{-2aT} + C (1 - e^{-2aT}) / (2a)   (full 2x2),
    mu_T = N_2(m_T, V_T).

What this stresses that d = 1 cannot:
* the MDN must output mixtures on R^2 and, since its components are
  DIAGONAL Gaussians, it must combine several of them to capture the
  idiosyncratic correlation rho != 0 of the exact law;
* the cylindrical layer runs on 2-d Fourier moments <k, .>;
* the W2 validation uses SLICED Wasserstein over deterministic projection
  directions, each projected law being available in closed form:
  theta . X_T | W0  ~  N(theta^T m_T, theta^T V_T theta).

The common noise stays scalar, so the signature side is identical to the 
other benchmarks.
"""

from dataclasses import dataclass
import numpy as np
import torch
from .base import Benchmark


@dataclass
class OU2dParams:
    T: float = 1.0
    n_steps: int = 100
    a: float = 1.0
    sigma1: float = 0.4       # idiosyncratic vol, coordinate 1
    sigma2: float = 0.3       # idiosyncratic vol, coordinate 2
    rho: float = 0.5          # idiosyncratic correlation
    sigma0_vec: tuple = (0.3, -0.2)   # common-noise loading vector
    m0_range: tuple = (-1.0, 1.0)     # per coordinate
    s0_range: tuple = (0.2, 0.6)      # per coordinate


class OU2d(Benchmark):
    name = "ou2d"
    pretty = "Conditional OU in R^2"
    has_closed_form = True
    d = 2

    def __init__(self, **kwargs):
        self.p = OU2dParams(**kwargs)
        p = self.p
        self.C = np.array([[p.sigma1 ** 2, p.rho * p.sigma1 * p.sigma2],
                           [p.rho * p.sigma1 * p.sigma2, p.sigma2 ** 2]])
        self.L = np.linalg.cholesky(self.C)
        self.s0v = np.asarray(p.sigma0_vec, dtype=float)

    def sample_evaluation_points(self, data, k_x, rng, n_particles=None):
        """Fresh draws from N_2(mT, VT), with full covariance."""
        M = len(data["m0"])
        chol = np.linalg.cholesky(data["VT"])
        z = rng.standard_normal((M, k_x, self.d))
        return (data["mT"][:, None, :]
                + np.einsum("mij,mkj->mki", chol, z))

    def simulate(self, M: int, N: int, rng: np.random.Generator) -> dict:
        p = self.p
        dt = p.T / p.n_steps
        sqdt = np.sqrt(dt)

        m0 = rng.uniform(*p.m0_range, size=(M, 2))
        s0 = rng.uniform(*p.s0_range, size=(M, 2))
        X = m0[:, None, :] + s0[:, None, :] * rng.standard_normal((M, N, 2))

        W0 = np.zeros((M, p.n_steps + 1))
        for k in range(p.n_steps):
            dW0 = sqdt * rng.standard_normal((M, 1, 1))          # COMMON
            dB = sqdt * rng.standard_normal((M, N, 2)) @ self.L.T
            cond_mean = X.mean(axis=1, keepdims=True)            # (M, 1, 2)
            X = (X + p.a * (cond_mean - X) * dt + dB
                 + self.s0v[None, None, :] * dW0)
            W0[:, k + 1] = W0[:, k] + dW0[:, 0, 0]

        decay = np.exp(-2.0 * p.a * p.T)
        mT = m0 + self.s0v[None, :] * W0[:, -1][:, None]         # (M, 2)
        VT = (np.einsum("md,de->mde", s0 ** 2,
                        np.eye(2)) * 0.0)                        # init (M,2,2)
        VT[:, 0, 0] = s0[:, 0] ** 2 * decay
        VT[:, 1, 1] = s0[:, 1] ** 2 * decay
        VT += self.C[None, :, :] * (1.0 - decay) / (2.0 * p.a)

        times = np.linspace(0.0, p.T, p.n_steps + 1)
        return dict(m0=m0, s0=s0, W0=W0, XT=X, mT=mT, VT=VT, times=times)

    # reference conditional law N_2(m_T, V_T)
    def ref_quantiles(self, data: dict, probs: np.ndarray,
                      theta: np.ndarray | None = None) -> np.ndarray:
        """Quantiles of the PROJECTED reference law theta . X_T | W0
        (theta required for d = 2): (S,) -> (S, B)."""
        assert theta is not None, "d = 2: pass a projection direction theta"
        th = np.asarray(theta, dtype=float)
        m = data["mT"] @ th                                       # (B,)
        v = np.einsum("i,bij,j->b", th, data["VT"], th)           # (B,)
        mt = torch.as_tensor(m, dtype=torch.float64)
        st = torch.as_tensor(np.sqrt(v), dtype=torch.float64)
        pr = torch.as_tensor(np.asarray(probs),
                             dtype=torch.float64).view(-1, 1)
        return torch.distributions.Normal(mt, st).icdf(pr).numpy()

    def ref_moments(self, data: dict):
        """(mean vector (B, 2), TRACE of the covariance (B,))."""
        tr = data["VT"][:, 0, 0] + data["VT"][:, 1, 1]
        return data["mT"], tr

    def ref_density(self, data: dict, i: int, grid: np.ndarray) -> np.ndarray:
        raise NotImplementedError("d = 2 uses ellipse plots, not 1-d density")

    def ref_label(self) -> str:
        return r"exact law $\mathcal{N}_2(m_T, V_T)$"

    def entropy_floor(self, data: dict) -> float:
        detV = np.linalg.det(data["VT"])
        return float((0.5 * np.log((2 * np.pi * np.e) ** 2 * detV)).mean())

    def ref_fourier(self, data: dict, freqs: np.ndarray) -> np.ndarray:
        """<phi, N_2(m_T, V_T)> in closed form with the FULL covariance:
        E[sin(<k, Y>)] = exp(-1/2 k^T V k) sin(<k, m>).  freqs: (P, 2)."""
        fr = np.asarray(freqs)
        quad = np.einsum("pi,bij,pj->bp", fr, data["VT"], fr)     # (B, P)
        damp = np.exp(-0.5 * quad)
        arg = data["mT"] @ fr.T                                   # (B, P)
        return np.concatenate([damp * np.sin(arg), damp * np.cos(arg)],
                              axis=1)
