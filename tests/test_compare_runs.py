"""The comparison must refuse a confounded pair of sweeps.

Run 2002135 was used as the baseline for a set of runs that also differed from it in
`seasonal_bands`. The result attributed to the treatment was partly that second variable, and
nothing caught it because comparing two runs' configurations was something nobody did. The
guard tested here is the whole reason analysis/compare_runs.py exists; the table it prints is
the easy part.
"""
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import pytest  # noqa: E402
from compare_runs import (check_cohort, check_configs, collect,  # noqa: E402
                          metrics, scan, seed_of)


def _variant(run, seed, cfg, rq=None, test_pids=None):
    d = run / f"tcn_none_seed{seed}_nw0.3"
    d.mkdir(parents=True, exist_ok=True)
    m = {"config": cfg}
    if test_pids is not None:
        m["test_pids"] = sorted(test_pids)
    (d / "metrics.json").write_text(json.dumps(m), encoding="utf-8")
    for name, payload in (rq or {}).items():
        (d / name).mkdir(exist_ok=True)
        (d / name / f"{name.lower()}.json").write_text(json.dumps(payload), encoding="utf-8")
    return d


BASE = {"backbone": "tcn", "pe": "none", "seasonal_bands": "harmonics", "epochs": 40}


def test_seed_is_read_from_the_directory_name():
    assert seed_of("results_hrd/2503584/tcn_none_seed42_nw0.3") == 42
    assert seed_of("results_hrd/2503584/paper_cosinor_topk2_cache.npz") is None


def test_arms_differing_only_in_the_tested_key_are_accepted(tmp_path):
    t, c = tmp_path / "t", tmp_path / "c"
    for s in (7, 13):
        _variant(t, s, {**BASE, "noise_weight": 0.3})
        _variant(c, s, {**BASE, "noise_weight": 0.0})
    assert check_configs(collect([t]), collect([c]), ["noise_weight"]) == {}


def test_a_second_moving_variable_is_reported(tmp_path):
    """The 2002135 failure, reproduced: the treatment key is declared, and a second one
    moves alongside it."""
    t, c = tmp_path / "t", tmp_path / "c"
    for s in (7, 13):
        _variant(t, s, {**BASE, "noise_weight": 0.3})
        _variant(c, s, {**BASE, "noise_weight": 0.0, "seasonal_bands": "single"})
    off = check_configs(collect([t]), collect([c]), ["noise_weight"])
    assert set(off) == {"seasonal_bands"}


def test_an_undeclared_treatment_key_is_itself_a_confound(tmp_path):
    """Forgetting --expect must not pass quietly: the table would then be attributed to a
    variable the caller never said was moving."""
    t, c = tmp_path / "t", tmp_path / "c"
    _variant(t, 7, {**BASE, "noise_weight": 0.3})
    _variant(c, 7, {**BASE, "noise_weight": 0.0})
    assert set(check_configs(collect([t]), collect([c]), [])) == {"noise_weight"}


def test_incidental_keys_never_count_as_confounds(tmp_path):
    """Two sweeps always land in different folders on different cards. If that counted, the
    guard would refuse every real comparison and be turned off."""
    t, c = tmp_path / "t", tmp_path / "c"
    _variant(t, 7, {**BASE, "run_id": "2503584", "gpu": 0, "output_dir": "results_hrd"})
    _variant(c, 7, {**BASE, "run_id": "2438763", "gpu": 3, "output_dir": "results_hrd"})
    assert check_configs(collect([t]), collect([c]), []) == {}


def test_a_control_split_over_two_sweeps_is_one_arm(tmp_path):
    """A stage-0 array and its self-healed tail are the same run in two folders."""
    a, b = tmp_path / "a", tmp_path / "b"
    _variant(a, 7, BASE)
    _variant(b, 13, BASE)
    assert sorted(collect([a, b])) == [7, 13]


def test_metrics_are_pulled_from_all_three_levels(tmp_path):
    d = _variant(tmp_path / "t", 7, BASE, rq={
        "RQ1": {"decomposition": {"rec_full_trend": 0.68, "rec_full_rhythm": 0.93,
                                  "DIS": 0.11}},
        "RQ2": {"concordance": {"DSSL": {"C": 0.88}, "V^S amp": {"C": 0.56}}},
        "RQ3": {"utility": {"DSSL (frozen)": {"auc": 0.72}, "Cosinor": {"auc": 0.66}}}})
    m = metrics(d)
    assert m[("RQ1", "DIS")] == 0.11
    assert m[("RQ2", "DSSL")] == 0.88
    assert m[("RQ3", "Cosinor")] == 0.66
    assert len(m) == 7


def test_a_missing_rq_directory_is_absent_not_an_error(tmp_path):
    """Not every variant runs every level, and a half-finished sweep must still compare on
    what it does have rather than crash."""
    assert metrics(_variant(tmp_path / "t", 7, BASE)) == {}


def test_a_null_auc_is_dropped_rather_than_read_as_zero(tmp_path):
    """rq3 writes null for a rung that could not be scored. Reading that as 0.0 would show
    the arm losing by 0.7 on a rung it never ran."""
    d = _variant(tmp_path / "t", 7, BASE,
                 rq={"RQ3": {"utility": {"A": {"auc": None}, "B": {"auc": 0.6}}}})
    assert metrics(d) == {("RQ3", "B"): 0.6}


def test_scan_ranks_the_matching_control_first(tmp_path, capsys):
    """`2438763` was assumed to be the V^N sweep's control because it carried the same
    `_nw0.3` tag, and it moves three other variables. The tag is not the configuration."""
    t = tmp_path / "treat"
    _variant(t, 7, {**BASE, "noise_weight": 0.3})
    _variant(t, 13, {**BASE, "noise_weight": 0.3})
    runs = tmp_path / "runs"
    for s in (7, 13):
        _variant(runs / "match", s, {**BASE, "noise_weight": 0.0})
        _variant(runs / "tagged", s, {**BASE, "noise_weight": 0.0,
                                      "smooth_bins": 5, "decomp_aug": True})
    scan(collect([t]), str(runs / "*"), ["noise_weight"])
    lines = [l for l in capsys.readouterr().out.splitlines()
             if "match" in l or "tagged" in l]
    assert "match" in lines[0] and "matches on everything" in lines[0]
    assert "tagged" in lines[1] and "smooth_bins" in lines[1] and "decomp_aug" in lines[1]


def test_scan_ignores_runs_that_share_no_seed(tmp_path):
    """A sweep over different seeds cannot be paired, so it is not a candidate at all."""
    t = tmp_path / "treat"
    _variant(t, 7, BASE)
    runs = tmp_path / "runs"
    _variant(runs / "elsewhere", 999, BASE)
    with pytest.raises(SystemExit):
        scan(collect([t]), str(runs / "*"), [])


def test_the_same_seed_on_a_changed_dataset_is_refused(tmp_path, capsys):
    """The confound no configuration comparison can see. The split is a deterministic
    function of --seed and the participant pool, so if a shared seed puts different people in
    test, the dataset moved underneath the two runs while every argparse value stayed equal --
    which is exactly what a schema migration does."""
    t, c = tmp_path / "t", tmp_path / "c"
    _variant(t, 7, BASE, test_pids=["p1", "p2", "p3"])
    _variant(c, 7, BASE, test_pids=["p1", "p2", "p9"])
    diff = check_cohort(collect([t]), collect([c]))
    assert [d[0] for d in diff] == [7]
    assert "1 differing" in capsys.readouterr().out


def test_an_identical_split_passes(tmp_path):
    t, c = tmp_path / "t", tmp_path / "c"
    for s in (7, 13):
        _variant(t, s, BASE, test_pids=["p1", "p2"])
        _variant(c, s, BASE, test_pids=["p2", "p1"])   # order must not matter
    assert check_cohort(collect([t]), collect([c])) == []


def test_a_run_predating_the_pid_record_is_unrecorded_not_a_mismatch(tmp_path, capsys):
    """Older runs wrote no test_pids. Counting that as a differing cohort would refuse every
    comparison against them; counting it as a match would claim a check that did not run."""
    t, c = tmp_path / "t", tmp_path / "c"
    _variant(t, 7, BASE, test_pids=["p1"])
    _variant(c, 7, BASE)
    assert check_cohort(collect([t]), collect([c])) == []
    assert "1 unrecorded" in capsys.readouterr().out
