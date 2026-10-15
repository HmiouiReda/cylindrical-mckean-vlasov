"""Conditional Ornstein-Uhlenbeck with common noise -- the CLOSED-FORM
benchmark.
    dX_t = a ( E[X_t | F^{W0}_t] - X_t ) dt + sigma dB_t + sigma0 dW0_t,
    X_0 ~ N(m0, s0^2),   (m0, s0) drawn per scenario.

Exact conditional law:
        => m_T = m0 + sigma0 * W0_T
        => v_T = s0^2 e^{-2aT} + sigma^2 (1 - e^{-2aT}) / (2a)
    Conditional law:  mu_T = N(m_T, v_T).
this benchmark validates the pipeline against an exact target

CRITICAL broadcasting point:
  - dW0 : COMMON noise         -> shape (M, 1)  => broadcast over particles
  - dB  : IDIOSYNCRATIC noise  -> shape (M, N)  => independent per particle"""
from dataclasses import dataclass
import numpy as np
import torch
from .base import Benchmark

@dataclass
class OUParams:
    T: float = 1.0            # horizon
    n_steps: int = 100        # Euler-Maruyama steps
    a: float = 1.0            # mean-reversion strength
    sigma: float = 0.4        # idiosyncratic volatility
    sigma0: float = 0.3       # common-noise volatility
    m0_range: tuple = (-1.0, 1.0)   # initial law: m0 ~ U(m0_range)
    s0_range: tuple = (0.2, 0.8)    #              s0 ~ U(s0_range)

class OrnsteinUhlenbeck(Benchmark):
    name = "ou"
    pretty = "Conditional OU"
    has_closed_form = True

    def __init__(self, **kwargs):
        self.p = OUParams(**kwargs)

    def sample_evaluation_points(self, data, k_x, rng, n_particles=None):
        """Fresh draws from the exact conditional Gaussian law."""
        M = len(data["m0"])
        return (data["mT"][:, None]
                + np.sqrt(data["vT"])[:, None]
                * rng.standard_normal((M, k_x)))


    def simulate(self, M: int, N: int, rng: np.random.Generator) -> dict:
        p = self.p
        dt = p.T / p.n_steps
        sqdt = np.sqrt(dt)

        # scenario-dependent initial law
        m0 = rng.uniform(*p.m0_range, size=M)
        s0 = rng.uniform(*p.s0_range, size=M)
        X = m0[:, None] + s0[:, None] * rng.standard_normal((M, N))   # (M, N)

        W0 = np.zeros((M, p.n_steps + 1))

        # Time loop only; everything else is vectorised
        for k in range(p.n_steps):
            dW0 = sqdt * rng.standard_normal((M, 1))    # COMMON : broadcast
            dB = sqdt * rng.standard_normal((M, N))     # IDIOSYNCRATIC
            cond_mean = X.mean(axis=1, keepdims=True)   # empirical cond. mean field
            X = (X + p.a * (cond_mean - X) * dt
                 + p.sigma * dB + p.sigma0 * dW0)
            W0[:, k + 1] = W0[:, k] + dW0[:, 0]

        # analytic ground truth
        mT = m0 + p.sigma0 * W0[:, -1]
        decay = np.exp(-2.0 * p.a * p.T)
        vT = s0 ** 2 * decay + p.sigma ** 2 * (1.0 - decay) / (2.0 * p.a)

        times = np.linspace(0.0, p.T, p.n_steps + 1)
        return dict(m0=m0, s0=s0, W0=W0, XT=X, mT=mT, vT=vT, times=times)

    # reference conditional law
    def ref_quantiles(self, data: dict, probs: np.ndarray) -> np.ndarray:
        mT = torch.as_tensor(data["mT"], dtype=torch.float64)
        sT = torch.as_tensor(np.sqrt(data["vT"]), dtype=torch.float64)
        pr = torch.as_tensor(np.asarray(probs), dtype=torch.float64).view(-1, 1)
        return torch.distributions.Normal(mT, sT).icdf(pr).numpy()   # (S, B)

    def ref_moments(self, data: dict):
        return data["mT"], data["vT"]

    def ref_fourier(self, data: dict, freqs) -> np.ndarray:
        # closed form: E[sin kX] = e^{-k^2 v_T/2} sin(k m_T) for N(m_T, v_T)
        from features import fourier_moments
        return fourier_moments(data["mT"], np.sqrt(data["vT"]), np.asarray(freqs))

    def ref_density(self, data: dict, i: int, grid: np.ndarray) -> np.ndarray:
        m, v = data["mT"][i], data["vT"][i]
        return np.exp(-0.5 * (grid - m) ** 2 / v) / np.sqrt(2 * np.pi * v)

    def ref_label(self) -> str:
        return r"exact law $\mathcal{N}(m_T, v_T)$"


    def ref_cdf0(self, data: dict) -> np.ndarray:
        m = torch.as_tensor(data["mT"], dtype=torch.float64)
        sd = torch.as_tensor(np.sqrt(data["vT"]), dtype=torch.float64)
        return torch.distributions.Normal(m, sd).cdf(torch.zeros_like(m)).numpy()

    def entropy_floor(self, data: dict) -> float:
        # differential entropy of N(., v_T), averaged over scenarios
        return float(np.mean(0.5 * np.log(2 * np.pi * np.e * data["vT"])))
