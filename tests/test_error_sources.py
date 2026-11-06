"""Task 5: path refinement, the EM fit, cloud W2, and the saved-model check."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from benchmarks.base import refine_path
from error_sources import em_mixture, w2_clouds

REPO = Path(__file__).resolve().parents[1]


def test_refine_path_keeps_coarse_points_and_bridge_variance():
    rng = np.random.default_rng(0)
    M, n, r, T = 4000, 10, 8, 2.0
    times = np.linspace(0.0, T, n + 1)
    W0 = np.concatenate([np.zeros((M, 1)),
                         np.cumsum(np.sqrt(T / n) * rng.standard_normal((M, n)), axis=1)],
                        axis=1)
    Wf, tf = refine_path(W0, times, r, rng)
    assert Wf.shape == (M, n * r + 1) and tf.shape == (n * r + 1,)
    np.testing.assert_array_equal(Wf[:, ::r], W0)
    np.testing.assert_array_equal(tf[::r], times)
    np.testing.assert_allclose(np.diff(tf), T / (n * r))
    # unconditionally the fine increments are N(0, dt_fine)
    inc = np.diff(Wf, axis=1)
    assert abs(inc.var() / (T / (n * r)) - 1.0) < 0.02
    # given the coarse path, the midpoint of a step has bridge variance dt/4
    mid = Wf[:, r // 2] - 0.5 * (W0[:, 0] + W0[:, 1])
    assert abs(mid.var() / (T / n / 4) - 1.0) < 0.06


def test_em_recovers_a_known_mixture():
    rng = np.random.default_rng(1)
    pi, mu, sd = np.array([0.2, 0.5, 0.3]), np.array([-2.0, 0.0, 3.0]), np.array([0.3, 0.5, 0.4])
    comp = rng.choice(3, size=(2, 20000), p=pi)
    X = mu[comp] + sd[comp] * rng.standard_normal((2, 20000))
    w, m, s = em_mixture(X, 3, iters=500)
    o = np.argsort(m[0, :, 0])
    np.testing.assert_allclose(w[0, o], pi, atol=0.02)
    np.testing.assert_allclose(m[0, o, 0], mu, atol=0.05)
    np.testing.assert_allclose(s[0, o, 0], sd, atol=0.05)


def test_w2_clouds_is_exact():
    rng = np.random.default_rng(2)
    X, Y = rng.standard_normal((3, 500)), rng.standard_normal((3, 500))
    direct = np.sqrt(((np.sort(X, 1) - np.sort(Y, 1)) ** 2).mean(1))
    np.testing.assert_allclose(w2_clouds(X, Y), direct, rtol=1e-12)
    # a cloud and its duplicate are the same measure
    np.testing.assert_allclose(w2_clouds(X, np.concatenate([X, X], 1)), 0.0, atol=1e-12)
    # a shift by c moves W2 by exactly c, whatever the sizes
    Z = np.concatenate([X, X, X], 1) + 0.7
    np.testing.assert_allclose(w2_clouds(X, Z), 0.7, rtol=1e-10)


def test_saved_model_reproduces_and_check_runs(tmp_path):
    cfg = {"benchmark": {"n_steps": 40, "n_ref": 300},
           "run": {"M": 240, "N": 50, "sig_level": 2, "J": 4, "k_keep": 16,
                   "mdn_epochs": 12, "psi_epochs": 12, "hidden": 48,
                   "S_quant": 256, "batch": 128, "lr": 1e-3, "k_x": 8,
                   "n_eval_particles": 300}}
    cfg_path = tmp_path / "dw.json"
    cfg_path.write_text(json.dumps(cfg))
    run = lambda cmd: subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                                     timeout=600, check=True)
    run([sys.executable, "main.py", "--benchmark", "double_well", "--config",
         str(cfg_path), "--save-model", "10", "--outdir", str(tmp_path)])
    run([sys.executable, "error_sources.py", "--run", str(tmp_path), "--refine", "2",
         "--n-hi", "2000", "--copies", "4", "--plugin-sizes", "50,200",
         "--J-list", "1,3", "--em-iters", "50"])
    out = json.loads((tmp_path / "error_sources.json").read_text())
    m = out["mdn"]
    assert abs(m["recomputed_w2_mean"] - m["stored_w2_mean"]) < 1e-6
    assert out["config"]["K"] == 10
    for key in ("reference", "ref_fine", "hi_coarse", "train_coarse", "train_fine",
                "plugin_50", "plugin_200"):
        assert np.isfinite(out[key]["bias_rms"]) and out[key]["w2_rms"] > 0
    assert set(out["labels"]) == {"moments", "quantile", "order"}
    assert out["cost"]["H"]["us_per_particle_step"] > 0
