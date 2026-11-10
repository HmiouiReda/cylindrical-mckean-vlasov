import numpy as np

from benchmarks import get_benchmark
from sim_cache import SimCache, cache_key


def test_key_depends_on_simulation_inputs_only():
    base = dict(benchmark="ou", bench_params={"a": 1.0}, M=10, N=5, seed=0,
                k_x=16, k_x_test=200, n_eval_particles=2000)
    k0 = cache_key(**base)
    assert cache_key(**base) == k0
    for field, value in [("seed", 1), ("M", 11), ("N", 6), ("k_x", 8),
                         ("bench_params", {"a": 2.0}), ("benchmark", "kuramoto")]:
        assert cache_key(**{**base, field: value}) != k0


def test_round_trip_restores_data_and_rng_state(tmp_path):
    bench = get_benchmark("double_well", n_steps=10, n_ref=20)
    rng = np.random.default_rng(3)
    data = bench.simulate(4, 6, rng)
    state = rng.bit_generator.state
    expected_next = np.random.default_rng(3)
    bench.simulate(4, 6, expected_next)
    expected = expected_next.permutation(4)

    cache = SimCache(tmp_path, "k", label="dw")
    cache.save(dict(data=data, rng_after_sim=state, x_eval={}))
    got = cache.load()

    for key, value in data.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(got["data"][key], value)
    fresh = np.random.default_rng(99)
    fresh.bit_generator.state = got["rng_after_sim"]
    np.testing.assert_array_equal(fresh.permutation(4), expected)
    assert not list(tmp_path.glob("*.tmp*"))
