"""Feature engineering: encode (initial law, common-noise path) into a vector.
Notes
Pure-NumPy fallback for the signature via Chen's identity
should be implemented for the case where the C module iisignature is not available.
"""

import numpy as np
import iisignature

FREQS = np.array([0.5, 1.0, 1.5, 2.0])
FREQS2D = np.array([[0.5, 0.0], [0.0, 0.5], [1.0, 0.0], [0.0, 1.0],
                    [0.5, 0.5], [0.5, -0.5], [1.0, 0.5], [0.5, 1.0]])

def make_freqs(n_fourier: int, d: int = 1) -> np.ndarray:
    """Construct a deterministic nested Fourier family.
    In dimension one:
        n_fourier = 4 gives [0.5, 1.0, 1.5, 2.0].
    The families are nested:
        P=2 -> [0.5, 1.0]
        P=4 -> [0.5, 1.0, 1.5, 2.0]
    Hence increasing P adds frequencies without replacing the old ones."""
    if n_fourier <= 0:
        raise ValueError("n_fourier must be positive")

    if d == 1:
        return 0.5 * np.arange(1, n_fourier + 1, dtype=float)

    if n_fourier > len(FREQS2D):
        raise ValueError(
            "2-d frequency sweep currently supports "
            f"at most {len(FREQS2D)} vectors")

    return FREQS2D[:n_fourier].copy()


"""Fourier moments of the initial law N(m0, s0^2)  (closed form)
#   E[sin(kX)] = exp(-k^2 s0^2 / 2) sin(k m0),   same for cos.
we use the closed-form characteristic function of the Gaussian
initial law N(m0, s0^2).  For non-Gaussian initial laws, replace it with
empirical moments of an initial sample against the same Fourier family."""
def fourier_moments(m0: np.ndarray, s0: np.ndarray,
                    freqs: np.ndarray = FREQS) -> np.ndarray:
    """Diagonal-Gaussian initial law on R^d.  m0, s0: (M, d); freqs: (P, d).
    E[sin(<k, X>)] = exp(-1/2 sum_i k_i^2 s0_i^2) sin(<k, m0>), same cos."""
    if m0.ndim == 1:  # d = 1
        k = freqs[None, :]                                   # (1, P)
        damp = np.exp(-0.5 * k ** 2 * s0[:, None] ** 2)      # (M, P)
        s = damp * np.sin(k * m0[:, None])
        c = damp * np.cos(k * m0[:, None])
    else:  # d >= 2
        damp = np.exp(-0.5 * ((freqs[None, :, :] ** 2)
                              * (s0[:, None, :] ** 2)).sum(-1))     # (M, P)
        arg = (freqs[None, :, :] * m0[:, None, :]).sum(-1)          # (M, P)
        s = damp * np.sin(arg)
        c = damp * np.cos(arg)
    return np.concatenate([s, c], axis=1)                # (M, 2P)


def initial_fourier_moments(data: dict,freqs: np.ndarray,) -> np.ndarray:
    """Exact Fourier encoding of every implemented initial-law family.
    Returns:
        [E sin(k_1 X0), ..., E sin(k_P X0),
         E cos(k_1 X0), ..., E cos(k_P X0)]
    for every scenario."""
    law = data.get("init_law", "gaussian")
    k = np.asarray(freqs, dtype=float)

    # Uniform initial law:
    # X0 ~ Uniform(m0-h, m0+h)
    if law == "uniform":
        kk = k[None, :]

        # np.sinc(z) = sin(pi*z)/(pi*z), so
        # sin(kh)/(kh) = np.sinc(kh/pi).
        amplitude = np.sinc(
            kk * data["half_width0"][:, None] / np.pi
        )
        argument = kk * data["m0"][:, None]

        sine = amplitude * np.sin(argument)
        cosine = amplitude * np.cos(argument)

        return np.concatenate([sine, cosine], axis=1)

    # Gaussian-mixture initial law:
    # X0 ~ w N(mu1, s²) + (1-w) N(mu2, s²)
    if law == "gaussian_mixture":
        kk = k[None, :]
        weight = data["weight"][:, None]

        damping = np.exp(
            -0.5
            * kk**2
            * data["component_std"][:, None] ** 2
        )

        sine = damping * (
            weight * np.sin(kk * data["mu1_0"][:, None])
            + (1.0 - weight)
            * np.sin(kk * data["mu2_0"][:, None])
        )

        cosine = damping * (
            weight * np.cos(kk * data["mu1_0"][:, None])
            + (1.0 - weight)
            * np.cos(kk * data["mu2_0"][:, None])
        )

        return np.concatenate([sine, cosine], axis=1)

    # Existing Gaussian initial laws.
    if data["m0"].ndim == 2:
        selected_freqs = k if k.ndim == 2 else FREQS2D
        return fourier_moments(
            data["m0"],
            data["s0"],
            selected_freqs)

    return fourier_moments(data["m0"], data["s0"], k)



def sig_dim(level: int, d: int = 2) -> int:
    "Dimension of the truncated signature (levels 1..level, path dim d)"
    return sum(d ** l for l in range(1, level + 1))


def compute_signature(paths: np.ndarray, level: int) -> np.ndarray:
    "Use `iisignature`(optimized C)"
    return np.stack([iisignature.sig(np.ascontiguousarray(p), level)
                        for p in paths])
   


def make_time_augmented_paths(times: np.ndarray, W0: np.ndarray) -> np.ndarray:
    "(n+1,), (M, n+1) -> (M, n+1, 2): paths  s |-> (s, W0_s)"
    M = W0.shape[0]
    tt = np.broadcast_to(times, (M, times.shape[0]))
    return np.stack([tt, W0], axis=2)


# Standardisation is needed for the training: level-l signature coordinates decay like
# 1/l!, a fatal scale gap for Adam otherwise
class Scaler:
    def fit(self, X: np.ndarray) -> "Scaler":
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8   # eps: handles constant features
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


def build_features(data: dict, sig_level: int,
                   freqs: np.ndarray = FREQS) -> np.ndarray:
    """Concatenate [Fourier(mu_0) | Signature(time-augmented common noise)].
    The signature side is dimension-agnostic (the common noise stays scalar,
    so the lifted path lives in R^2 in all benchmarks); only the initial-law
    encoding switches between 1-d and d-dimensional Fourier moments."""
    fm = initial_fourier_moments(data,freqs)
    paths = make_time_augmented_paths(data["times"], data["W0"])     # (M, n+1, 2)
    sg = compute_signature(paths, sig_level)                         # (M, d_sig)
    return np.concatenate([fm, sg], axis=1).astype(np.float32)