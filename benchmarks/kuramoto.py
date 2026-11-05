"""Bi-harmonic Kuramoto oscillators with common noise -- the benchmark that
sits INSIDE the hypotheses of Proposition 1, and whose interaction is not a
conditional mean.

    dX_t = [ k1 int sin(y - X_t) mu_t(dy)
             + k2 int sin(2(y - X_t)) mu_t(dy) ] dt
           + sigma dB_t + sigma0(X_t) dW0_t          (Ito),
    sigma0(x) = nu (1 + beta sin x),   X_0 ~ N(m0, s0^2),
    mu_t = L(X_t | F^{W0}_t).

Why this model:
* the drift is BOUNDED by |k1| + |k2| and smooth with bounded x-derivatives,
  and it is Lipschitz in W1 (hence W2) because sin(qy) is q-Lipschitz --
  the regularity hypotheses of Friz-Hocquet-Le hold, unlike for the
  double-well and multiplicative benchmarks;
* the measure enters through FOUR bounded linear functionals,
      S_q = int sin(qy) mu(dy),   C_q = int cos(qy) mu(dy),  q = 1, 2,
  since  int sin(q(y-x)) mu(dy) = S_q cos qx - C_q sin qx.  So the law is
  NOT seen through its mean only: delta_0 and (delta_{-pi/2}+delta_{pi/2})/2
  share a mean but have C_1 = 1 and C_1 = 0;
* sigma0 is non-constant, so the common noise is genuinely rough and the
  Ito-Stratonovich correction does not vanish (as in multiplicative.py);
* k1 < 0 with k2 > 0 is deliberate.  Around a single cluster a displaced
  particle feels -(k1 + 2 k2) delta, so one cluster is stable exactly when
  k1 + 2 k2 > 0; we take k1 + 2 k2 = -2 to make it UNSTABLE.  Around two
  clusters at separation pi the first harmonic cancels and the restoring
  rate is 2 k2 > 0, so the two-cluster ("nematic") state is the attractor.
  The bi-harmonic kernel and that state are classical: Vlasov, Komarov and
  Pikovsky, arXiv:1411.3204; Hansel, Mato and Meunier, PRE 48, 3470 (1993).

Validation without a closed form
As for the double well: two independent N_ref reference replicates sharing
only (mu0, W0), A for the measure-channel integrals and B for the
regression labels.

Numerics:
Plain Euler-Maruyama.  The drift is bounded, so no taming is needed -- do
not copy the double well's tamed scheme.
"""

from dataclasses import dataclass

import numpy as np

from .particle_ref import ParticleReferenceBenchmark


@dataclass
class KuramotoParams:
    T: float = 1.0
    n_steps: int = 200
    k1: float = -4.0          # first harmonic: REPULSIVE, see module docstring
    k2: float = 1.0           # second harmonic: attractive, sets the clusters
    sigma: float = 0.3        # idiosyncratic volatility (additive)
    nu: float = 0.3           # common-noise volatility scale
    beta: float = 0.5         # relative amplitude of the state dependence
    m0_range: tuple = (-0.5, 0.5)
    s0_range: tuple = (0.2, 0.5)
    n_ref: int = 2000         # particles in EACH independent reference replicate


class Kuramoto(ParticleReferenceBenchmark):
    name = "kuramoto"
    pretty = "Bi-harmonic Kuramoto"

    def __init__(self, **kwargs):
        self.p = KuramotoParams(**kwargs)

    evaluation_scheme = "euler_maruyama"

    def sigma0(self, x: np.ndarray) -> np.ndarray:
        p = self.p
        return p.nu * (1.0 + p.beta * np.sin(x))

    def ito_strat_correction(self, x: np.ndarray) -> np.ndarray:
        """1/2 sigma0(x) sigma0'(x)."""
        p = self.p
        return 0.5 * p.nu ** 2 * p.beta * np.cos(x) * (1.0 + p.beta * np.sin(x))

    # the interaction, written once and shared by every scheme below
    @staticmethod
    def fourier_moments(U: np.ndarray) -> tuple:
        """(S1, C1, S2, C2) of the empirical law of U, each (M, 1)."""
        return (np.sin(U).mean(axis=1, keepdims=True),
                np.cos(U).mean(axis=1, keepdims=True),
                np.sin(2.0 * U).mean(axis=1, keepdims=True),
                np.cos(2.0 * U).mean(axis=1, keepdims=True))

    def drift(self, U: np.ndarray, moments: tuple) -> np.ndarray:
        """Bi-harmonic kernel from the four moments -- O(N), never a pair
        matrix."""
        p = self.p
        S1, C1, S2, C2 = moments
        return (p.k1 * (S1 * np.cos(U) - C1 * np.sin(U))
                + p.k2 * (S2 * np.cos(2.0 * U) - C2 * np.sin(2.0 * U)))

    def step(self, U: np.ndarray, dB: np.ndarray, dW0: np.ndarray,
             dt: float) -> np.ndarray:
        """One Euler-Maruyama step with U's own empirical mean field."""
        p = self.p
        return (U + self.drift(U, self.fourier_moments(U)) * dt
                + p.sigma * dB + self.sigma0(U) * dW0)

    def sample_evaluation_points(self, data, k_x, rng, n_particles=None):
        """First k_x terminal states of a separate interacting system C."""
        if k_x <= 0 or n_particles is None or n_particles < k_x:
            raise ValueError("Require n_particles >= k_x > 0 for system C")
        M = len(data["m0"])
        X = (data["m0"][:, None] + data["s0"][:, None]
             * rng.standard_normal((M, n_particles)))

        increments = np.diff(data["W0"], axis=1)
        for k, dt in enumerate(np.diff(data["times"])):
            dW0 = increments[:, k:k + 1]
            dB = np.sqrt(dt) * rng.standard_normal((M, n_particles))
            X = self.step(X, dB, dW0, dt)

        return X[:, :k_x].copy()

    def simulate(self, M: int, N: int, rng: np.random.Generator) -> dict:
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

        X = m0[:, None] + s0[:, None] * rng.standard_normal((M, N))
        XA = m0[:, None] + s0[:, None] * rng.standard_normal((M, n_a))
        XB = m0[:, None] + s0[:, None] * rng.standard_normal((M, n_b))
        W0 = np.zeros((M, p.n_steps + 1))

        for k in range(p.n_steps):
            # Same common noise, independent idiosyncratic noises.
            dW0 = sqdt * rng.standard_normal((M, 1))
            X = self.step(X, sqdt * rng.standard_normal((M, N)), dW0, dt)
            XA = self.step(XA, sqdt * rng.standard_normal((M, n_a)), dW0, dt)
            XB = self.step(XB, sqdt * rng.standard_normal((M, n_b)), dW0, dt)
            W0[:, k + 1] = W0[:, k] + dW0[:, 0]

        times = np.linspace(0.0, p.T, p.n_steps + 1)

        return {"m0": m0, "s0": s0, "W0": W0, "XT": np.ascontiguousarray(X),
                "X_ref_input": np.ascontiguousarray(XA),
                "X_ref_target": np.ascontiguousarray(XB),
                "X_ref": np.ascontiguousarray(np.concatenate([XA, XB], axis=1)),
                "times": times}

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

        def heun_step(X, corrected, dB, dW0):
            """Predictor-corrector: converges to the Stratonovich integral
            for the state-dependent common-noise term; the additive
            idiosyncratic term needs no correction."""
            moments = self.fourier_moments(X)          # frozen per step

            def drift_fn(U):
                c = self.drift(U, moments)
                return c - self.ito_strat_correction(U) if corrected else c

            c0 = drift_fn(X)
            g0 = self.sigma0(X)
            Xp = X + c0 * dt + g0 * dW0 + p.sigma * dB    # predictor
            c1 = drift_fn(Xp)
            g1 = self.sigma0(Xp)
            return (X + 0.5 * (c0 + c1) * dt
                    + 0.5 * (g0 + g1) * dW0 + p.sigma * dB)

        for _ in range(p.n_steps):
            dW0 = sqdt * rng.standard_normal((M, 1))
            dB = sqdt * rng.standard_normal((M, N))
            XA = self.step(XA, dB, dW0, dt)                     # (A) Ito EM
            XB = heun_step(XB, True, dB, dW0)                   # (B) corrected
            XC = heun_step(XC, False, dB, dW0)                  # (C) naive

        def w2(U, V):
            return np.sqrt(((np.sort(U, axis=1)
                             - np.sort(V, axis=1)) ** 2).mean(axis=1))

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
