"""Tests for the referee-driven extensions: multiplicative common noise
(Ito-Stratonovich correction), the 2-d benchmark"""
import numpy as np

from benchmarks import get_benchmark
from cylindrical import empirical_fourier
                    


# Multiplicative common noise
def test_ito_stratonovich_correction():
    b = get_benchmark("multiplicative", n_steps=120)
    r = b.ito_stratonovich_check(M=40, N=1500,
                                 rng=np.random.default_rng(0))
    # corrected Heun reproduces the Ito law within (twice) the MC noise
    assert r["gap_corrected_mean"] < 2.0 * r["mc_resolution"]
    # the NAIVE Stratonovich reading is measurably wrong
    assert r["gap_naive_mean"] > 2.0 * max(r["gap_corrected_mean"],
                                           r["mc_resolution"])


def test_multiplicative_simulation_sane():
    b = get_benchmark("multiplicative", n_steps=60, n_ref=200)
    d = b.simulate(30, 25, np.random.default_rng(1))
    assert d["XT"].shape == (30, 25) and d["X_ref"].shape == (30, 200)
    assert np.isfinite(d["XT"]).all() and np.isfinite(d["X_ref"]).all()
    # bounded sigma0: nu (1 - beta) <= sigma0 <= nu (1 + beta)
    s0 = b.sigma0(np.linspace(-8, 8, 400))
    assert s0.min() > 0.0
    assert np.isclose(s0.max(), b.p.nu * (1 + b.p.beta), atol=1e-2)



# 2-d conditional OU: exact law vs particle system, and the full-covariance
# oracle integrals.
def test_ou2d_closed_form_projections():
    b = get_benchmark("ou2d", n_steps=50)
    d = b.simulate(300, 300, np.random.default_rng(0))
    for th in (np.array([1.0, 0.0]), np.array([0.0, 1.0]),
               np.array([0.6, 0.8])):
        P = d["XT"] @ th
        m_th = d["mT"] @ th
        assert abs((P.mean(1) - m_th).mean()) < 0.02



def test_ou2d_oracle_fourier_vs_sampling():
    from features import FREQS2D
    b = get_benchmark("ou2d", n_steps=40)
    d = b.simulate(3, 10, np.random.default_rng(2))
    ints = b.ref_fourier(d, FREQS2D)                      # (3, 2P)
    rng = np.random.default_rng(3)
    for i in range(3):
        L = np.linalg.cholesky(d["VT"][i])
        Y = d["mT"][i] + rng.standard_normal((40000, 2)) @ L.T
        emp = empirical_fourier(Y[None, :, :], FREQS2D).numpy()[0]
        assert np.abs(emp - ints[i]).max() < 0.02




