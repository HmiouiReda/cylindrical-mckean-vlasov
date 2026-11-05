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
    assert d["XT"].shape == (30, 25) and d["X_ref"].shape == (30, 400)
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





# Bi-harmonic Kuramoto: the common noise is state-dependent here too, so the
# Ito-Stratonovich correction is non-trivial.  Unlike `multiplicative` the
# correction is small against the resolution of this benchmark's wide
# two-cluster law, so the comparison is pathwise: same increments, and the
# naive Stratonovich reading must drift measurably further from the Ito law
# than the corrected one does.
def test_kuramoto_ito_stratonovich_correction():
    b = get_benchmark("kuramoto", n_steps=200)
    r = b.ito_stratonovich_check(M=40, N=1500,
                                 rng=np.random.default_rng(0))
    assert r["gap_naive_mean"] > 2.0 * r["gap_corrected_mean"]


# Order-parameter functional (task 7)
def test_order_parameters_of_a_two_cluster_cloud():
    """r_1 vanishes and r_2 is one for two equal clusters at separation pi:
    the odd order parameters of the nematic state are zero."""
    from benchmarks.base import order_parameters
    cloud = np.tile(np.array([-np.pi / 2, np.pi / 2]), (3, 500))
    assert abs(order_parameters(cloud, 1)).max() < 1e-12
    assert np.abs(order_parameters(cloud, 2) - 1.0).max() < 1e-12


def test_order_parameters_match_the_gaussian_closed_form():
    """For a Gaussian law r_q = exp(-q^2 v / 2) exactly.  Checks the generic
    quantile-grid implementation against ou's exact conditional variance."""
    b = get_benchmark("ou", n_steps=50)
    d = b.simulate(50, 10, np.random.default_rng(0))
    r1, r2 = b.ref_order(d)
    v = d["vT"]
    assert np.abs(r1 - np.exp(-0.5 * v)).max() < 1e-3
    assert np.abs(r2 - np.exp(-2.0 * v)).max() < 1e-3


def test_order_target_pair_uses_independent_replicates():
    """A and B must give different numbers: the labels and the measure
    channel are built from separate reference systems."""
    b = get_benchmark("kuramoto", n_steps=60, n_ref=400)
    d = b.simulate(20, 25, np.random.default_rng(1))
    (a1, a2), (b1, b2) = b.independent_target_pair(d, "order")
    for r in (a1, a2, b1, b2):
        assert r.shape == (20,)
        assert ((r >= 0.0) & (r <= 1.0)).all()
    assert not np.allclose(a1, b1)
    # The labels come from replicate B.
    t1, t2 = b.target_order(d)
    assert np.allclose(t1, b1) and np.allclose(t2, b2)
