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





