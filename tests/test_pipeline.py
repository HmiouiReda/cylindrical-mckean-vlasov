"""End-to-end smoke tests: main.py must run for representative benchmarks
and produce finite metrics and all expected figures. Sizes are tiny; these
tests check plumbing, not numerical accuracy."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
FIGS = ["fig1_benchmark_density.png", "fig2_mdn_vs_ref.png",
        "fig3_training_curves.png", "fig4_w2_floor.png", "fig5_baselines.png"]


def _run(cmd):
    res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                         timeout=600)
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    return res


def _check_outputs(outdir: Path):
    metrics = json.loads((outdir / "metrics.json").read_text())
    for k in ("w2_mdn_mean", "w2_floor_mean", "rel_l2_full",
              "rel_l2_empirical","rel_l2_oracle","mdn_val_nll"):
        assert isinstance(metrics[k], float)
    for f in FIGS:
        assert (outdir / f).exists(), f"missing figure {f}"
    _check_timings(metrics)
    return metrics


def _leaves(node):
    for v in node.values():
        if isinstance(v, dict):
            yield from _leaves(v)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            yield v


def _check_timings(metrics: dict):
    t = metrics["timings"]
    timed = {k: v for k, v in t.items() if k != "environment"}
    assert all(v >= 0 for v in _leaves(timed))
    # the disjoint stages cover the run, up to a small untimed rest
    assert 0 <= t["other"] <= 0.05 * t["total"]
    assert abs(metrics["runtime_sec"] - round(t["total"], 1)) < 1e-9
    sim = t["simulate"]
    parts = sum(sim.get(k, 0.0) for k in ("train_cloud", "ref_A", "ref_B"))
    assert parts <= sim["total"] + 1e-9
    for key in ("batched_per_scenario", "single_scenario_latency",
                "features_per_scenario"):
        assert t["inference"][key] > 0


def test_pipeline_ou_smoke(tmp_path):
    _run([sys.executable,"main.py","--benchmark", "ou","--quick",
        "--M", "300","--N", "60","--mdn-epochs", "12","--psi-epochs",
        "12", "--baselines", "full","--outdir", str(tmp_path)])

    metrics = _check_outputs(tmp_path)

    assert metrics["benchmark"] == "ou"
    assert metrics["entropy_floor"] is not None
    assert np.isfinite(metrics["entropy_floor"])


def test_pipeline_double_well_smoke(tmp_path):
    cfg = {
        "benchmark": {"n_steps": 40, "n_ref": 300},
        "run": {"M": 240, "N": 50, "sig_level": 2, "J": 4, "k_keep": 16,
                "mdn_epochs": 12, "psi_epochs": 12, "hidden": 48,
                "S_quant": 256, "batch": 128, "lr": 1e-3, "k_x": 8},
    }
    cfg_path = tmp_path / "dw_smoke.json"
    cfg_path.write_text(json.dumps(cfg))
    _run([sys.executable, "main.py", "--benchmark", "double_well",
          "--config", str(cfg_path), "--outdir", str(tmp_path)])
    metrics = _check_outputs(tmp_path)
    assert metrics["benchmark"] == "double_well"
    assert metrics["reference_resolution"] > 0
