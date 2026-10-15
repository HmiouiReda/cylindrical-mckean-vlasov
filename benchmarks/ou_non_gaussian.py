"""Exact non-Gaussian conditional OU benchmarks.
The dynamics are the same as in ``benchmarks/ou.py``::
    dX_t = a (m_t - X_t) dt + sigma dB_t + sigma0 dW0_t,
    m_t = E[X_t | F_t^{W0}].

For any initial law with mean m0,
    X_T = m0 + sigma0 W0_T + exp(-aT) (X_0 - m0) + Z_T,

where Z_T is centred Gaussian with variance
    v_noise = sigma^2 (1 - exp(-2aT)) / (2a).

This identity gives exact non-Gaussian reference laws for:
  * a uniform initial law;
  * a two-component Gaussian-mixture initial law.
"""
from dataclasses import dataclass
import numpy as np
import torch
from .base import Benchmark

_STD_NORMAL = torch.distributions.Normal(
    torch.tensor(0.0, dtype=torch.float64),
    torch.tensor(1.0, dtype=torch.float64),)


def _normal_pdf(x: torch.Tensor) -> torch.Tensor:
    """Standard Gaussian density."""
    return torch.exp(-0.5 * x.square()) / np.sqrt(2.0 * np.pi)


def _uniform_plus_normal_cdf( x: torch.Tensor,mean: torch.Tensor,
    half_width: torch.Tensor,sd: torch.Tensor,) -> torch.Tensor:
    """CDF of mean + U[-half_width, half_width] + N(0, sd^2).
    All tensors must be broadcastable.
    For U ~ Uniform(-A, A) and Z ~ N(0, sd^2),
        P(mean + U + Z <= x)= (1 / (2A)) integral_{-A}^{A}
                            Phi((x - mean - u) / sd) du.
    The primitive of Phi is: G(z) = z Phi(z) + phi(z)"""
    u_plus = (x - mean + half_width) / sd
    u_minus = (x - mean - half_width) / sd
    def G(u: torch.Tensor) -> torch.Tensor:
        return u * _STD_NORMAL.cdf(u) + _normal_pdf(u)

    return (sd/ (2.0 * half_width)* (G(u_plus) - G(u_minus)))


def _bisection_quantiles(cdf_fn,lo,hi,probs,iters: int = 70,) -> np.ndarray:
    """Vectorised inversion of a CDF by bisection.
    Parameters:
    cdf_fn:Function taking an array of shape (S, B) and returning the CDF.
    lo, hi:Lower and upper bounds, shape (B,).
    probs:Quantile levels, shape (S,).
    Returns:
    Quantiles with shape (S, B)."""
    lo = torch.as_tensor(lo,dtype=torch.float64,).view(1, -1)
    hi = torch.as_tensor(hi,dtype=torch.float64,).view(1, -1)
    p = torch.as_tensor(probs,dtype=torch.float64,).view(-1, 1)
    lo = lo.expand(p.shape[0], -1).clone()
    hi = hi.expand(p.shape[0], -1).clone()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        # If CDF(mid) >= p, the quantile is to the left of mid.
        go_left = cdf_fn(mid) >= p
        hi = torch.where(go_left, mid, hi)
        lo = torch.where(go_left, lo, mid)
    return (0.5 * (lo + hi)).numpy()


# Uniform initial distribution
@dataclass
class UniformOUParams:
    T: float = 1.0
    n_steps: int = 100
    a: float = 1.0
    sigma: float = 0.4
    sigma0: float = 0.3
    m0_range: tuple = (-1.0, 1.0)
    half_width_range: tuple = (0.3, 1.0)

class UniformInitialOU(Benchmark):
    """Conditional OU with X0 ~ Uniform(m0-h, m0+h)."""
    name = "ou_uniform"
    pretty = "Conditional OU — uniform initial law"
    has_closed_form = True

    def __init__(self, **kwargs):
        self.p = UniformOUParams(**kwargs)

    def sample_evaluation_points(self, data, k_x, rng, n_particles=None):
        """Fresh draws from the exact uniform-Gaussian convolution."""
        M = len(data["m0"])
        return (data["mT"][:, None]
                + data["half_width_T"][:, None]
                * rng.uniform(-1.0, 1.0, size=(M, k_x))
                + np.sqrt(data["v_noise"])[:, None]
                * rng.standard_normal((M, k_x)))

    def simulate(self,M: int,N: int,rng: np.random.Generator,) -> dict:
        p = self.p
        dt = p.T / p.n_steps
        sqdt = np.sqrt(dt)
        # One initial distribution per common-noise scenario.
        m0 = rng.uniform(*p.m0_range,size=M)
        half_width0 = rng.uniform(*p.half_width_range,size=M)

        # X0 ~ Uniform(m0-h, m0+h).
        X = (m0[:, None]+ half_width0[:, None]* rng.uniform(-1.0,1.0,size=(M, N),))
        W0 = np.zeros((M, p.n_steps + 1))

        # Finite particle approximation of the McKean–Vlasov dynamics.
        for k in range(p.n_steps):
            # One common increment per scenario.
            dW0 = (sqdt* rng.standard_normal((M, 1)))
            # Independent idiosyncratic increments.
            dB = (sqdt* rng.standard_normal((M, N)))
            cond_mean = X.mean(axis=1,keepdims=True)

            X = (X+ p.a * (cond_mean - X) * dt
                + p.sigma * dB+ p.sigma0 * dW0)

            W0[:, k + 1] = ( W0[:, k]+ dW0[:, 0])

        # Exact limiting conditional law.
        decay = np.exp(-p.a * p.T)
        v_noise = (p.sigma ** 2* (1.0 - decay ** 2) / (2.0 * p.a))
        mT = (m0+ p.sigma0 * W0[:, -1])
        half_width_T = (decay* half_width0)

        # Variance of U[-A, A] = A^2 / 3.
        vT = (half_width_T ** 2 / 3.0+ v_noise)

        return {
            "init_law": "uniform",
            "m0": m0,
            # Kept for compatibility and diagnostics.
            # This is the true standard deviation of X0.
            "s0": half_width0 / np.sqrt(3.0),
            "half_width0": half_width0,
            "W0": W0,
            "XT": X,
            "mT": mT,
            "half_width_T": half_width_T,
            "v_noise": np.full(M,v_noise),
            "vT": vT,
            "times": np.linspace(0.0,p.T,p.n_steps + 1)}

    def _cdf(self,data: dict,x: torch.Tensor,) -> torch.Tensor:
        mean = torch.as_tensor(data["mT"], dtype=torch.float64,).view(1, -1)

        half_width = torch.as_tensor(data["half_width_T"], dtype=torch.float64,).view(1, -1)

        sd = torch.sqrt(
            torch.as_tensor(data["v_noise"],
                        dtype=torch.float64)).view(1, -1)

        return _uniform_plus_normal_cdf(x=x,mean=mean, half_width=half_width,sd=sd)

    def ref_quantiles(self,data: dict,probs: np.ndarray,) -> np.ndarray:
        sd = np.sqrt(data["v_noise"])
        # Wide enough interval to contain essentially all Gaussian mass.
        lo = (data["mT"]- data["half_width_T"]- 9.0 * sd)
        hi = (data["mT"]+ data["half_width_T"]+ 9.0 * sd)
        return _bisection_quantiles(
            cdf_fn=lambda x: self._cdf(data, x),lo=lo,hi=hi,
            probs=probs)

    def ref_moments(self,data: dict,):
        return (data["mT"],data["vT"])

    def ref_fourier(self, data: dict,freqs) -> np.ndarray:
        """Exact Fourier moments of the terminal law.
        Characteristic function:
            exp(i k mT)
            sinc(k half_width_T)
            exp(-k^2 v_noise / 2).
        """
        k = np.asarray(freqs,dtype=float)[None, :]

        gaussian_damping = np.exp(-0.5* k ** 2
            * data["v_noise"][:, None])

        uniform_cf = np.sinc(
            k* data["half_width_T"][:, None]/ np.pi)

        amplitude = (gaussian_damping* uniform_cf)

        argument = (k* data["mT"][:, None])

        sine = (amplitude* np.sin(argument))

        cosine = (amplitude* np.cos(argument))

        return np.concatenate([sine, cosine], axis=1)

    def ref_density(self,data: dict, i: int,grid: np.ndarray) -> np.ndarray:
        """Exact density of Uniform + Gaussian convolution."""
        x = torch.as_tensor(grid, dtype=torch.float64)

        mean = torch.tensor(data["mT"][i],dtype=torch.float64)
        half_width = torch.tensor(data["half_width_T"][i], dtype=torch.float64)
        sd = torch.tensor(np.sqrt(data["v_noise"][i]), dtype=torch.float64)

        density = (_STD_NORMAL.cdf((x - mean + half_width) / sd)
                - _STD_NORMAL.cdf((x - mean - half_width) / sd)
                ) / (2.0 * half_width)

        return density.numpy()

    def ref_cdf0(self,data: dict) -> np.ndarray:
        x0 = torch.zeros((1, len(data["mT"])),
            dtype=torch.float64)

        return (self._cdf(data, x0).squeeze(0).numpy())

    def ref_label(self) -> str:
        return "exact uniform–Gaussian convolution"


# Gaussian-mixture initial distribution
@dataclass
class GaussianMixtureOUParams:
    T: float = 1.0
    n_steps: int = 100
    a: float = 1.0
    sigma: float = 0.4
    sigma0: float = 0.3
    m0_range: tuple = (-1.0, 1.0)
    separation_range: tuple = (0.6, 1.8)
    component_std_range: tuple = (0.10, 0.30)
    weight_range: tuple = (0.25, 0.75)


class GaussianMixtureInitialOU(Benchmark):
    """Conditional OU with a scenario-dependent two-Gaussian initial mixture."""
    name = "ou_gmm"
    pretty = "Conditional OU — Gaussian-mixture initial law"
    has_closed_form = True

    def __init__(self, **kwargs):
        self.p = GaussianMixtureOUParams(**kwargs)

    def sample_evaluation_points(self, data, k_x, rng, n_particles=None):
        """Fresh draws from the exact terminal two-Gaussian mixture."""
        M = len(data["m0"])
        choose_first = rng.random((M, k_x)) < data["weight"][:, None]
        means = np.where(choose_first,
                         data["mu1_T"][:, None], data["mu2_T"][:, None])
        return (means + np.sqrt(data["component_var_T"])[:, None]
                * rng.standard_normal((M, k_x)))

    def simulate(self,M: int,N: int,rng: np.random.Generator,) -> dict:
        p = self.p
        dt = p.T / p.n_steps
        sqdt = np.sqrt(dt)

        # m0 is the exact mean of the mixture.
        m0 = rng.uniform(*p.m0_range, size=M)
        separation = rng.uniform(*p.separation_range, size=M)
        component_std = rng.uniform(*p.component_std_range, size=M)
        weight = rng.uniform(*p.weight_range, size=M)

        # These definitions guarantee:
        # weight * mu1_0 + (1 - weight) * mu2_0 = m0.
        mu1_0 = m0 - (1.0 - weight) * separation
        mu2_0 = m0 + weight * separation

        # Select one component for every particle.
        choose_first = rng.random((M, N)) < weight[:, None]
        initial_component_mean = np.where(
            choose_first,
            mu1_0[:, None],
            mu2_0[:, None],
        )

        X = (
            initial_component_mean
            + component_std[:, None] * rng.standard_normal((M, N))
        )

        W0 = np.zeros((M, p.n_steps + 1))

        for k in range(p.n_steps):
            dW0 = sqdt * rng.standard_normal((M, 1))
            dB = sqdt * rng.standard_normal((M, N))
            cond_mean = X.mean(axis=1, keepdims=True)

            X = (
                X
                + p.a * (cond_mean - X) * dt
                + p.sigma * dB
                + p.sigma0 * dW0
            )

            W0[:, k + 1] = W0[:, k] + dW0[:, 0]

        decay = np.exp(-p.a * p.T)
        v_noise = p.sigma**2 * (1.0 - decay**2) / (2.0 * p.a)
        mT = m0 + p.sigma0 * W0[:, -1]

        # Each component is contracted towards mT.
        mu1_T = mT + decay * (mu1_0 - m0)
        mu2_T = mT + decay * (mu2_0 - m0)

        # Both components have the same terminal variance.
        component_var_T = decay**2 * component_std**2 + v_noise

        # Total variance:
        # within-component variance + between-component variance.
        vT = (
            component_var_T
            + weight * (1.0 - weight) * (decay * separation) ** 2
        )

        initial_var = (
            component_std**2
            + weight * (1.0 - weight) * separation**2
        )

        return {
            "init_law": "gaussian_mixture",
            "m0": m0,
            "s0": np.sqrt(initial_var),
            "weight": weight,
            "separation": separation,
            "component_std": component_std,
            "mu1_0": mu1_0,
            "mu2_0": mu2_0,
            "W0": W0,
            "XT": X,
            "mT": mT,
            "mu1_T": mu1_T,
            "mu2_T": mu2_T,
            "component_var_T": component_var_T,
            "vT": vT,
            "times": np.linspace(0.0, p.T, p.n_steps + 1),
        }

    def _cdf(
        self,
        data: dict,
        x: torch.Tensor,
    ) -> torch.Tensor:
        weight = torch.as_tensor(
            data["weight"],
            dtype=torch.float64,
        ).view(1, -1)

        mu1 = torch.as_tensor(
            data["mu1_T"],
            dtype=torch.float64,
        ).view(1, -1)

        mu2 = torch.as_tensor(
            data["mu2_T"],
            dtype=torch.float64,
        ).view(1, -1)

        sd = torch.sqrt(
            torch.as_tensor(
                data["component_var_T"],
                dtype=torch.float64,
            )
        ).view(1, -1)

        cdf1 = _STD_NORMAL.cdf((x - mu1) / sd)
        cdf2 = _STD_NORMAL.cdf((x - mu2) / sd)

        return weight * cdf1 + (1.0 - weight) * cdf2

    def ref_quantiles(
        self,
        data: dict,
        probs: np.ndarray,
    ) -> np.ndarray:
        sd = np.sqrt(data["component_var_T"])
        lo = np.minimum(data["mu1_T"], data["mu2_T"]) - 9.0 * sd
        hi = np.maximum(data["mu1_T"], data["mu2_T"]) + 9.0 * sd

        return _bisection_quantiles(
            cdf_fn=lambda x: self._cdf(data, x),
            lo=lo,
            hi=hi,
            probs=probs,
        )

    def ref_moments(self, data: dict):
        return data["mT"], data["vT"]

    def ref_fourier(
        self,
        data: dict,
        freqs,
    ) -> np.ndarray:
        """Exact terminal Fourier moments of the Gaussian mixture."""
        k = np.asarray(freqs, dtype=float)[None, :]
        weight = data["weight"][:, None]

        damping = np.exp(
            -0.5 * k**2 * data["component_var_T"][:, None]
        )

        sine = damping * (
            weight * np.sin(k * data["mu1_T"][:, None])
            + (1.0 - weight) * np.sin(k * data["mu2_T"][:, None])
        )

        cosine = damping * (
            weight * np.cos(k * data["mu1_T"][:, None])
            + (1.0 - weight) * np.cos(k * data["mu2_T"][:, None])
        )

        return np.concatenate([sine, cosine], axis=1)

    def ref_density(
        self,
        data: dict,
        i: int,
        grid: np.ndarray,
    ) -> np.ndarray:
        weight = data["weight"][i]
        sd = np.sqrt(data["component_var_T"][i])

        z1 = (grid - data["mu1_T"][i]) / sd
        z2 = (grid - data["mu2_T"][i]) / sd
        normalisation = sd * np.sqrt(2.0 * np.pi)

        density1 = np.exp(-0.5 * z1**2) / normalisation
        density2 = np.exp(-0.5 * z2**2) / normalisation

        return weight * density1 + (1.0 - weight) * density2

    def ref_cdf0(self, data: dict) -> np.ndarray:
        x0 = torch.zeros(
            (1, len(data["mT"])),
            dtype=torch.float64,
        )
        return self._cdf(data, x0).squeeze(0).numpy()

    def ref_label(self) -> str:
        return "exact two-component Gaussian mixture"