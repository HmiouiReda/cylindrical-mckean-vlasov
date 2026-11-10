import json

import sweep_table as st

DEFAULTS = {"ou": dict(J=5, n_fourier=4, sig_level=2, mdn_depth=2, psi_depth=2,
                       hidden=128, lr=0.001),
            "kuramoto": dict(J=5, n_fourier=4, sig_level=3, mdn_depth=2,
                             psi_depth=2, hidden=128, lr=0.001)}


def write(out, name, b, val, nll=1.0, train_seed=0, functional="moments"):
    d = out / "runs" / name
    d.mkdir(parents=True)
    m = dict(config=dict(DEFAULTS[b], train_seed=train_seed, functional=functional),
             rel_l2_val=val, rel_l2_full=val + 0.001, mdn_val_nll=nll,
             w2_mdn_val_mean=0.03, w2_mdn_mean=0.03, runtime_sec=1.0,
             timings=dict(environment=dict(start=dict(torch_threads=4),
                                           simulate_cached=True)))
    (d / "metrics.json").write_text(json.dumps(m))


def one_factor_stage(out):
    """J=2 wins on both; sig_level=4 wins on kuramoto; mdn_depth=3 wins on
    the functional but the law gets worse, so it is vetoed."""
    base = {0: 0.020, 1: 0.022}
    for b in st.BENCHMARKS:
        for s in st.SEEDS:
            write(out, f"{b}__s{s}__default", b, base[s])
            write(out, f"{b}__s{s}__J__2", b, base[s] - 0.002)
            write(out, f"{b}__s{s}__lr__0.003", b, base[s] + 0.001)
            write(out, f"{b}__s{s}__mdn_depth__3", b, base[s] - 0.002, nll=1.5)
        sig = "4" if b == "kuramoto" else "3"
        write(out, f"{b}__s0__sig_level__{sig}", b,
              base[0] - (0.002 if b == "kuramoto" else 0.0))
        write(out, f"{b}__s1__sig_level__{sig}", b,
              base[1] - (0.002 if b == "kuramoto" else 0.0))
        for k, dv in zip(range(1, 5), [1e-4, -1e-4, 2e-4, -2e-4]):
            write(out, f"{b}__s0__ts__{k}", b, base[0] + dv,
                  nll=1.0 + dv, train_seed=k)


def test_select_accepts_real_changes_and_vetoes(tmp_path):
    one_factor_stage(tmp_path)
    st.stage_select(tmp_path)
    sel = json.loads((tmp_path / "selection.json").read_text())
    got = {(c["param"], c["new"], c["scope"].split()[0]) for c in sel["changes"]}
    assert got == {("J", "2", "all"), ("sig_level", "4", "particle")}
    assert any("mdn_depth=3" in f and "vetoed" in f for f in sel["flags"])
    jobs = (tmp_path / "stage_c_jobs.txt").read_text().split("\n")
    names = [j.split()[0] for j in jobs if j]
    # ou has a single change, whose run exists; kuramoto has two
    assert sorted(n for n in names if "combined" in n) == [
        "kuramoto__s0__combined", "kuramoto__s1__combined"]
    assert "kuramoto__s0__order__final" in names
    combined = [j for j in jobs if j.startswith("kuramoto__s0__combined")][0]
    assert "--J 2" in combined and "--sig-level 4" in combined


def test_verdict_takes_combined_or_falls_back(tmp_path):
    for good, expect in [(True, 2), (False, 1)]:
        out = tmp_path / str(good)
        one_factor_stage(out)
        for s, v in zip(st.SEEDS, [0.020, 0.022]):
            write(out, f"kuramoto__s{s}__combined", "kuramoto",
                  v - (0.0035 if good else -0.001))
        st.stage_final(out)
        prop = json.loads((out / "freeze_proposal.json").read_text())
        assert len(prop["changes"]) == expect
        assert (out / "sweep_table.tex").exists()
