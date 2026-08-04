"""Mean-reverting McKean-Vlasov dynamics with MULTIPLICATIVE common noise
-- the benchmark that stresses the Ito/Stratonovich content of the paper.

    dX_t = a ( E[X_t | F^{W0}_t] - X_t ) dt + sigma dB_t
           + sigma0(X_t) dW0_t                       (Ito),
    sigma0(x) = nu (1 + beta sin x),   X_0 ~ N(m0, s0^2).

Why this coefficient:
* smooth and BOUNDED, bounded away from 0 for beta < 1 -- inside the
  regularity assumptions of Friz-Hocquet-Le (Proposition 1 of the paper);
* NON-constant, so the Ito-Stratonovich drift correction
      1/2 sigma0 sigma0'(x) = 1/2 nu^2 beta cos(x) (1 + beta sin x)
  does NOT vanish: this is exactly the regime where the geometric
  (Stratonovich) lift used for the signature features and the Ito lift used
  for the identification differ at level two, and where the additive-noise
  benchmarks say nothing;

Ito-Stratonovich numerical check:
Same Brownian increments, three schemes for the SAME target conditional
law (the Ito equation above):

  (A) Euler-Maruyama on the Ito equation                     [reference]
  (B) Heun (Stratonovich) scheme on the CORRECTED equation
      with drift  a(m - x) - 1/2 sigma0 sigma0'(x)           [must match A]
  (C) Heun (Stratonovich) scheme with the NAIVE drift
      a(m - x)                                               [wrong law]

The Heun predictor-corrector converges to the Stratonovich integral 
"""

from dataclasses import dataclass
import numpy as np
from .particle_ref import ParticleReferenceBenchmark


@dataclass
class MultiplicativeParams:
    T: float = 1.0
    n_steps: int = 200
    a: float = 1.0            # mean-reversion strength
    sigma: float = 0.3        # idiosyncratic volatility (additive)
    nu: float = 0.4           # common-noise volatility scale
    beta: float = 0.8         # relative amplitude of the state dependence
    m0_range: tuple = (-1.0, 1.0)
    s0_range: tuple = (0.2, 0.6)
    n_ref: int = 2000         # particles in EACH independent reference replicate


class MultiplicativeOU(ParticleReferenceBenchmark):
    name = "multiplicative"
    pretty = "Multiplicative common noise"

    def __init__(self, **kwargs):
        self.p = MultiplicativeParams(**kwargs)


    def sigma0(self, x: np.ndarray) -> np.ndarray:
        p = self.p
        return p.nu * (1.0 + p.beta * np.sin(x))

    def ito_strat_correction(self, x: np.ndarray) -> np.ndarray:
        """1/2 sigma0(x) sigma0'(x)."""
        p = self.p
        return 0.5 * p.nu ** 2 * p.beta * np.cos(x) * (1.0 + p.beta * np.sin(x))


    def simulate(self,M: int,N: int,rng: np.random.Generator) -> dict:
        """Simulate three conditionally independent particle systems.
        XT, X_ref_input and X_ref_target share only (mu0, W0).
        Each system has its own initial draws, idiosyncratic noises and
        empirical mean field."""
        p = self.p
        dt = p.T / p.n_steps
        sqdt = np.sqrt(dt)
        n_a = p.n_ref
        n_b = p.n_ref

        m0 = rng.uniform(*p.m0_range, size=M)
        s0 = rng.uniform(*p.s0_range, size=M)

        X = (m0[:, None]+ s0[:, None] * rng.standard_normal((M, N)))
        XA = (m0[:, None]+ s0[:, None] * rng.standard_normal((M, n_a)))
        XB = (m0[:, None]+ s0[:, None] * rng.standard_normal((M, n_b)))
        W0 = np.zeros((M, p.n_steps + 1))

        def step(U: np.ndarray,dB: np.ndarray,dW0: np.ndarray) -> np.ndarray:
            cond_mean = U.mean(axis=1, keepdims=True)
            return (U + p.a * (cond_mean - U) * dt
                + p.sigma * dB+ self.sigma0(U) * dW0)

        for k in range(p.n_steps):
            # Same common noise.
            dW0 = sqdt * rng.standard_normal((M, 1))

            X = step(X,sqdt * rng.standard_normal((M, N)),dW0)
            XA = step(XA,sqdt * rng.standard_normal((M, n_a)),dW0)
            XB = step(XB,sqdt * rng.standard_normal((M, n_b)),dW0)
            W0[:, k + 1] = W0[:, k] + dW0[:, 0]

        times = np.linspace(0.0, p.T, p.n_steps + 1)

        return {"m0": m0,"s0": s0,"W0": W0,"XT": np.ascontiguousarray(X),
            "X_ref_input": np.ascontiguousarray(XA),"X_ref_target": np.ascontiguousarray(XB),
            "X_ref": np.ascontiguousarray(np.concatenate([XA, XB], axis=1)),"times": times}


    def ito_stratonovich_check(self, M: int, N: int,
                               rng: np.random.Generator) -> dict:
        """Simulate (A) Ito EM, (B) corrected Heun, (C) naive Heun with the
        SAME Brownian increments; return the mean per-scenario W2 gaps and
        the Monte-Carlo resolution of the reference cloud A."""
        p = self.p
        dt = p.T / p.n_steps
        sqdt = np.sqrt(dt)

        m0 = rng.uniform(*p.m0_range, size=M)
        s0 = rng.uniform(*p.s0_range, size=M)
        X0 = m0[:, None] + s0[:, None] * rng.standard_normal((M, N))
        XA = X0.copy()
        XB = X0.copy()
        XC = X0.copy()

        def heun_step(X, drift_fn, dB, dW0):
            """Predictor-corrector: converges to the Stratonovich integral
            for the state-dependent common-noise term; the additive
            idiosyncratic term needs no correction."""
            mean = X.mean(axis=1, keepdims=True)          # frozen per step
            c0 = drift_fn(X, mean)
            g0 = self.sigma0(X)
            Xp = X + c0 * dt + g0 * dW0 + p.sigma * dB    # predictor
            c1 = drift_fn(Xp, mean)
            g1 = self.sigma0(Xp)
            return (X + 0.5 * (c0 + c1) * dt
                    + 0.5 * (g0 + g1) * dW0 + p.sigma * dB)

        drift_naive = lambda X, mean: p.a * (mean - X)
        drift_corr = (lambda X, mean:
                      p.a * (mean - X) - self.ito_strat_correction(X))

        for _ in range(p.n_steps):
            dW0 = sqdt * rng.standard_normal((M, 1))
            dB = sqdt * rng.standard_normal((M, N))
            meanA = XA.mean(axis=1, keepdims=True)
            XA = (XA + p.a * (meanA - XA) * dt
                  + p.sigma * dB + self.sigma0(XA) * dW0)     # (A) Ito EM
            XB = heun_step(XB, drift_corr, dB, dW0)           # (B) corrected
            XC = heun_step(XC, drift_naive, dB, dW0)          # (C) naive

        def w2(U, V):
            return np.sqrt(((np.sort(U, axis=1)
                             - np.sort(V, axis=1)) ** 2).mean(axis=1))

        #we can directly use particle_ref.reference_resolution
        n2 = N // 2
        res = np.sqrt(((np.sort(XA[:, :n2], axis=1)
                        - np.sort(XA[:, n2:2 * n2], axis=1)) ** 2
                       ).mean(axis=1)) / np.sqrt(2.0)
        gAB, gAC = w2(XA, XB), w2(XA, XC)
        return dict(gap_corrected_mean=float(gAB.mean()),
                    gap_corrected_std=float(gAB.std()),
                    gap_naive_mean=float(gAC.mean()),
                    gap_naive_std=float(gAC.std()),
                    mc_resolution=float(res.mean()),
                    M=M, N=N, n_steps=p.n_steps)
