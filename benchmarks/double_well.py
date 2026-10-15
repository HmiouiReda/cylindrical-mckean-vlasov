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
    n_ref: int = 2000         # particles in EACH independent reference replicate



class DoubleWell(ParticleReferenceBenchmark):
    name = "double_well"
    pretty = "Double well"

    def __init__(self, **kwargs):
        self.p = DoubleWellParams(**kwargs)

    
    evaluation_scheme = "tamed_euler"
    
    def sample_evaluation_points(self, data, k_x, rng, n_particles=None):
        """First k_x terminal states of a separate interacting system C."""
        if k_x <= 0 or n_particles is None or n_particles < k_x:
            raise ValueError("Require n_particles >= k_x > 0 for system C")
        p = self.p
        M = len(data["m0"])
        X = (data["m0"][:, None] + data["s0"][:, None]
             * rng.standard_normal((M, n_particles)))

        increments = np.diff(data["W0"], axis=1)
        for k, dt in enumerate(np.diff(data["times"])):
            dW0 = increments[:, k:k + 1]
            dB = np.sqrt(dt) * rng.standard_normal((M, n_particles))
            mean_C = X.mean(axis=1, keepdims=True)
            drift = p.kappa * (X - X ** 3) + p.theta * (mean_C - X)
            drift = drift / (1.0 + dt * np.abs(drift))
            X = X + drift * dt + p.sigma * dB + p.sigma0 * dW0

        return X[:, :k_x].copy()

    def simulate(self,M: int,N: int,rng: np.random.Generator) -> dict:
        p = self.p
        dt = p.T / p.n_steps
        sqdt = np.sqrt(dt)
        n_a = p.n_ref
        n_b = p.n_ref

        # Scenario parameters shared by all three systems.
        m0 = rng.uniform(*p.m0_range, size=M)
        s0 = rng.uniform(*p.s0_range, size=M)

        # Training particle system.
        X = (m0[:, None]+ s0[:, None] * rng.standard_normal((M, N)))
        # Reference A: used only for the Fourier inputs.
        XA = (m0[:, None]+ s0[:, None] * rng.standard_normal((M, n_a)))
        # Reference B: used only for the regression targets.
        XB = (m0[:, None]+ s0[:, None] * rng.standard_normal((M, n_b)))
        W0 = np.zeros((M, p.n_steps + 1))

        def step(U: np.ndarray,dB: np.ndarray,dW0: np.ndarray,) -> np.ndarray:
            # Each system computes its OWN empirical mean.
            cond_mean = U.mean(axis=1, keepdims=True)
            drift = (p.kappa * (U - U ** 3)+ p.theta * (cond_mean - U))
            # Tamed Euler drift.
            drift = drift / (1.0 + dt * np.abs(drift))
            return (U+ drift * dt+ p.sigma * dB+ p.sigma0 * dW0)

        for k in range(p.n_steps):
            # The common-noise increment is shared by all systems.
            dW0 = sqdt * rng.standard_normal((M, 1))
            # The idiosyncratic noises are independent.
            X = step(X,sqdt * rng.standard_normal((M, N)),dW0)
            XA = step(XA,sqdt * rng.standard_normal((M, n_a)),dW0)
            XB = step(XB,sqdt * rng.standard_normal((M, n_b)),dW0)
            W0[:, k + 1] = W0[:, k] + dW0[:, 0]

        times = np.linspace(0.0, p.T, p.n_steps + 1)

        return {"m0": m0,"s0": s0,"W0": W0,"XT": np.ascontiguousarray(X),
            "X_ref_input": np.ascontiguousarray(XA),"X_ref_target": np.ascontiguousarray(XB),
            "X_ref": np.ascontiguousarray(np.concatenate([XA, XB], axis=1)),"times": times}
