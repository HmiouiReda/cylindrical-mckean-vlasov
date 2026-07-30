"""Bistable (double-well) McKean-Vlasov diffusion with common noise -- the
NONLINEAR benchmark, no closed form.

    dX_t = [ kappa (X_t - X_t^3) + theta ( E[X_t | F^{W0}_t] - X_t ) ] dt
           + sigma dB_t + sigma0 dW0_t,        X_0 ~ N(m0, s0^2).

Validation without a closed form
N_ref extra particles are simulated with the SAME common-noise path and
independent idiosyncratic noise.  By conditional propagation of chaos their
empirical law converges to mu_T at rate O(N_ref^{-1/2});
reference_resolution estimates this residual error by a half-sample
split, so every reported W2 can be read relative to it.

All N + N_ref particles interact through the JOINT empirical mean: both
sets then carry the same (small) propagation-of-chaos bias and are
exchangeable samples of the same conditional-law approximation. 

Numerics:
The cubic drift is superlinear, so plain Euler-Maruyama may explode; we use
the TAMED scheme of Hutzenthaler-Jentzen-Kloeden
               drift_h = drift / (1 + dt * |drift|).
"""

from dataclasses import dataclass

import numpy as np

from .particle_ref import ParticleReferenceBenchmark


@dataclass
class DoubleWellParams:
    T: float = 2.0            # horizon (long enough for bimodality)
    n_steps: int = 200        # tamed Euler steps
    kappa: float = 1.0        # double-well strength: drift kappa*(x - x^3)
    theta: float = 0.25       # mean-field attraction to E[X | W0]
    sigma: float = 0.5        # idiosyncratic volatility
    sigma0: float = 0.25      # common-noise volatility
    m0_range: tuple = (-0.3, 0.3)   # near-symmetric start => both wells filled
    s0_range: tuple = (0.3, 0.6)
    n_ref: int = 2000         # reference particles per scenario (validation)


class DoubleWell(ParticleReferenceBenchmark):
    name = "double_well"
    pretty = "Double well"

    def __init__(self, **kwargs):
        self.p = DoubleWellParams(**kwargs)

    def simulate(self, M: int, N: int, rng: np.random.Generator) -> dict:
        p = self.p
        dt = p.T / p.n_steps
        sqdt = np.sqrt(dt)
        n_tot = N + p.n_ref            # training + reference

        m0 = rng.uniform(*p.m0_range, size=M)
        s0 = rng.uniform(*p.s0_range, size=M)
        X = m0[:, None] + s0[:, None] * rng.standard_normal((M, n_tot))

        W0 = np.zeros((M, p.n_steps + 1))

        for k in range(p.n_steps):
            dW0 = sqdt * rng.standard_normal((M, 1))      # COMMON : broadcast
            dB = sqdt * rng.standard_normal((M, n_tot))   # IDIOSYNCRATIC
            cond_mean = X.mean(axis=1, keepdims=True)
            drift = p.kappa * (X - X ** 3) + p.theta * (cond_mean - X)
            drift = drift / (1.0 + dt * np.abs(drift))    # taming (HJK 2012)
            X = X + drift * dt + p.sigma * dB + p.sigma0 * dW0
            W0[:, k + 1] = W0[:, k] + dW0[:, 0]

        times = np.linspace(0.0, p.T, p.n_steps + 1)
        return dict(m0=m0, s0=s0, W0=W0,
                    XT=np.ascontiguousarray(X[:, :N]),
                    X_ref=np.ascontiguousarray(X[:, N:]),
                    times=times)
