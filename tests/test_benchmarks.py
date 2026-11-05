"""Benchmark simulators: OU against its closed form; double-well sanity."""
import numpy as np
import pytest

from benchmarks import get_benchmark


# ---------------------------------------------------------------------------
# Conditional OU: the particle system must match the exact conditional law
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ou_data():
    bench = get_benchmark("ou", n_steps=50)
    rng = np.random.default_rng(0)
    return bench, bench.simulate(M=300, N=500, rng=rng)


def test_ou_shapes(ou_data):
    _, d = ou_data
    assert d["XT"].shape == (300, 500)
    assert d["W0"].shape == (300, 51)
    for k in ("m0", "s0", "mT", "vT"):
        assert d[k].shape == (300,)


def test_ou_conditional_mean(ou_data):
    _, d = ou_data
    diff = d["XT"].mean(axis=1) - d["mT"]
    # E[m_hat - m_T] = 0 ; per-scenario std ~ sqrt((s0^2 + sigma^2 T + v_T)/N)
    assert abs(diff.mean()) < 0.02
    assert diff.std() < 0.12


def test_ou_conditional_variance(ou_data):
    _, d = ou_data
    ratio = d["XT"].var(axis=1).mean() / d["vT"].mean()
    assert 0.9 < ratio < 1.1



# Double well: no explosion, reference machinery, bimodality
@pytest.fixture(scope="module")
def dw_data():
    bench = get_benchmark("double_well", n_steps=80, n_ref=400)
    rng = np.random.default_rng(0)
    return bench, bench.simulate(M=120, N=40, rng=rng)


def test_dw_shapes_and_stability(dw_data):
    _, d = dw_data
    assert d["XT"].shape == (120, 40)
    assert d["X_ref"].shape == (120, 800)
    assert np.isfinite(d["XT"]).all() and np.isfinite(d["X_ref"]).all()



def test_dw_reference_resolution_positive(dw_data):
    bench, d = dw_data
    res = bench.reference_resolution(d)
    assert 0.0 < res < 1.0







# Bi-harmonic Kuramoto: two-cluster terminal law, reference machinery, and
# the separate evaluation-point system
@pytest.fixture(scope="module")
def kur_data():
    bench = get_benchmark("kuramoto", n_steps=100, n_ref=400)
    rng = np.random.default_rng(0)
    return bench, bench.simulate(M=120, N=40, rng=rng)


def test_kur_shapes_and_stability(kur_data):
    _, d = kur_data
    assert d["XT"].shape == (120, 40)
    assert d["X_ref"].shape == (120, 800)
    assert np.isfinite(d["XT"]).all() and np.isfinite(d["X_ref"]).all()


def test_kur_reference_resolution_positive(kur_data):
    bench, d = kur_data
    res = bench.reference_resolution(d)
    assert 0.0 < res < 1.0


def test_kur_clusters_form(kur_data):
    """Two clusters locked at separation pi have variance near (pi/2)^2.
    Far below => no clusters; check the sign of k1."""
    _, d = kur_data
    assert 1.5 < d["X_ref"].var(axis=1).mean() < 3.5


def test_kur_sigma0_bounded(kur_data):
    bench, _ = kur_data
    s0 = bench.sigma0(np.linspace(-8.0, 8.0, 400))
    assert s0.min() > 0.0
    assert np.isclose(s0.max(), bench.p.nu * (1 + bench.p.beta), atol=1e-2)


def test_kur_evaluation_points_match_the_simulated_law(kur_data):
    """System C and reference A must be the same law: their discrepancy is
    the A/B discrepancy.  Guards the two copies of the step against drift."""
    bench, d = kur_data
    rng = np.random.default_rng(7)
    C = bench.sample_evaluation_points(d, bench.p.n_ref, rng,
                                       n_particles=bench.p.n_ref)
    assert C.shape == (120, bench.p.n_ref)
    assert np.isfinite(C).all()

    def w2(U, V):
        return np.sqrt(((np.sort(U, axis=1)
                         - np.sort(V, axis=1)) ** 2).mean(axis=1)).mean()

    ab = w2(d["X_ref_input"], d["X_ref_target"])
    ca = w2(C, d["X_ref_input"])
    assert ca < 1.5 * ab


def test_kur_not_mean_determined():
    """Two initial laws with the same mean but different C1 reach different
    terminal laws: the interaction is not a function of the conditional
    mean.  This is the claim R2.3 asked for."""
    bench = get_benchmark("kuramoto", n_steps=100)
    rng = np.random.default_rng(0)
    M, N, spread = 40, 2000, 0.05

    # Smeared delta_0 (C1 = 1) and the smeared two-point law at +-pi/2
    # (C1 = 0); both have mean 0.
    near_zero = spread * rng.standard_normal((M, N))
    signs = np.where(np.arange(N) % 2 == 0, -1.0, 1.0)
    two_point = (np.pi / 2) * signs + spread * rng.standard_normal((M, N))

    assert abs(near_zero.mean(axis=1).mean()) < 0.01
    assert abs(two_point.mean(axis=1).mean()) < 0.01
    assert np.cos(near_zero).mean(axis=1).mean() > 0.9
    assert abs(np.cos(two_point).mean(axis=1).mean()) < 0.1

    # Same noise for both, so any difference comes from the initial law.
    dt = bench.p.T / bench.p.n_steps
    sqdt = np.sqrt(dt)
    U, V = near_zero.copy(), two_point.copy()
    for _ in range(bench.p.n_steps):
        dW0 = sqdt * rng.standard_normal((M, 1))
        dB = sqdt * rng.standard_normal((M, N))
        U = bench.step(U, dB, dW0, dt)
        V = bench.step(V, dB, dW0, dt)

    w2 = np.sqrt(((np.sort(U, axis=1)
                   - np.sort(V, axis=1)) ** 2).mean(axis=1)).mean()
    assert w2 > 0.5
