"""Closed-form mixture math: Fourier integrals and quantile inversion."""
import numpy as np
import torch
import torch.distributions as D

from cylindrical import mixture_fourier
from evaluate import  mixture_quantiles


def _random_mixture(B=3, J=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, J, generator=g)
    mu = 2.0 * torch.randn(B, J, generator=g)
    sigma = 0.2 + 0.8 * torch.rand(B, J, generator=g)
    return logits, mu, sigma


def test_mixture_fourier_matches_monte_carlo():
    torch.manual_seed(0)
    logits, mu, sigma = _random_mixture()
    freqs = torch.tensor([0.5, 1.0, 1.5, 2.0])
    closed = mixture_fourier(logits, mu, sigma, freqs)        # (B, 8)

    mix = D.MixtureSameFamily(D.Categorical(logits=logits),
                              D.Normal(mu, sigma))
    y = mix.sample((400_000,))                                # (n, B)
    mc = torch.cat(
        [torch.stack([torch.sin(k * y).mean(0) for k in freqs], dim=1),
         torch.stack([torch.cos(k * y).mean(0) for k in freqs], dim=1)],
        dim=1)
    assert (closed - mc).abs().max().item() < 1e-2


def test_mixture_quantiles_single_component_is_gaussian():
    # J identical components => the mixture IS N(0.7, 1.3^2)
    B, J = 2, 3
    logits = torch.zeros(B, J)
    mu = torch.full((B, J), 0.7)
    sigma = torch.full((B, J), 1.3)
    probs = torch.linspace(0.02, 0.98, 25)
    q = mixture_quantiles(logits, mu, sigma, probs)
    q_exact = D.Normal(0.7, 1.3).icdf(probs).view(-1, 1).expand(-1, B)
    assert (q - q_exact).abs().max().item() < 1e-3
