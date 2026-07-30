"""Closed-form Fourier moments vs Monte Carlo; standardisation; shapes."""
import numpy as np

from features import FREQS, Scaler, build_features, fourier_moments, sig_dim


def test_fourier_moments_match_monte_carlo():
    rng = np.random.default_rng(0)
    m0 = np.array([0.3, -0.7])
    s0 = np.array([0.5, 0.9])
    closed = fourier_moments(m0, s0)                     # (2, 2P)
    X = m0[:, None] + s0[:, None] * rng.standard_normal((2, 400_000))
    mc = np.concatenate(
        [np.stack([np.sin(k * X).mean(axis=1) for k in FREQS], axis=1),
         np.stack([np.cos(k * X).mean(axis=1) for k in FREQS], axis=1)],
        axis=1)
    assert np.abs(closed - mc).max() < 5e-3


def test_scaler_train_only_statistics():
    rng = np.random.default_rng(1)
    X = rng.normal(3.0, 2.0, size=(500, 6))
    Z = Scaler().fit(X).transform(X)
    assert np.abs(Z.mean(0)).max() < 1e-8
    assert np.abs(Z.std(0) - 1.0).max() < 1e-6


def test_build_features_shape():
    rng = np.random.default_rng(2)
    n = 10
    data = dict(m0=rng.uniform(-1, 1, 5), s0=rng.uniform(0.2, 0.8, 5),
                times=np.linspace(0, 1, n + 1),
                W0=rng.standard_normal((5, n + 1)).cumsum(axis=1) * 0.1)
    for lvl in (1, 2, 3):
        feats = build_features(data, sig_level=lvl)
        assert feats.shape == (5, 2 * len(FREQS) + sig_dim(lvl))
        assert feats.dtype == np.float32
        assert np.isfinite(feats).all()
