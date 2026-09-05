"""The arms are assembled right, and the verdict follows the numbers rather than the wish."""
import json
from pathlib import Path

import numpy as np
import pytest

from analysis.readout_interaction import aggregate, arms
from analysis.readout_sweep import SEGS


def _parts(n=6, d=3, T=112):
    rng = np.random.default_rng(0)
    p = {"season spec": rng.normal(size=(n, 5)).astype(np.float32)}
    for s in SEGS:
        p[f"trend seg {s:2d}"] = rng.normal(size=(n, s * d)).astype(np.float32)
        p[f"season seg {s:2d}"] = rng.normal(size=(n, s * d)).astype(np.float32)
    return p, n, d


def test_arm_set_and_widths():
    p, n, d = _parts()
    a = arms(p)
    assert set(a) == ({"PRODUCTION"} | {f"both seg {s:2d}" for s in SEGS}
                      | {f"trend seg {s:2d} + spec" for s in SEGS})
    assert a["PRODUCTION"].shape == (n, d + 5)
    assert np.allclose(a["PRODUCTION"],
                       np.concatenate([p["trend seg  1"], p["season spec"]], 1))
    # seg 1 with the spectral seasonal half IS the production readout, which is the
    # internal check that the family is wired to the right reference
    assert np.allclose(a["trend seg  1 + spec"], a["PRODUCTION"])
    for s in SEGS:
        assert a[f"both seg {s:2d}"].shape == (n, 2 * s * d)
        assert a[f"trend seg {s:2d} + spec"].shape == (n, s * d + 5)


def _write(tmp, diffs, n_variants=12):
    """One JSON per variant, with DSSL sitting `diffs[readout]` above its control."""
    names = ["PRODUCTION"] + [f"both seg {s:2d}" for s in SEGS]
    rng = np.random.default_rng(1)
    for i in range(n_variants):
        d = tmp / f"tcn_none_seed{i}" / "RQ3"
        d.mkdir(parents=True)
        base = {n: 0.55 + rng.normal(0, 0.01) for n in names}
        (d / "readout_interaction.json").write_text(json.dumps({
            "variant": "tcn/none", "seed": i,
            "Random-init": base,
            "DSSL": {n: base[n] + diffs.get(n, 0.0) for n in names}}))
    return names


def test_verdict_rejects_when_both_arms_rise_together(tmp_path, capsys):
    _write(tmp_path, {})                      # DSSL never above its control
    out = aggregate(tmp_path)
    assert out["verdict"].startswith("REJECT")
    assert abs(out["diff"]["PRODUCTION"]) < 1e-9


def test_verdict_builds_when_the_gap_opens_with_resolution(tmp_path):
    _write(tmp_path, {"both seg 28": 0.05})   # only the widest readout separates them
    out = aggregate(tmp_path)
    assert out["best_readout"] == "both seg 28"
    assert out["verdict"].startswith("OPEN THE READOUT")


def test_aggregate_refuses_an_empty_run(tmp_path):
    with pytest.raises(SystemExit):
        aggregate(tmp_path)


def test_vs_production_table_is_paired_not_averaged(tmp_path, capsys):
    """A readout that beats PRODUCTION in every variant must read 24/24, and one that beats
    it only on average -- big wins on a few, losses on the rest -- must not."""
    names = ["PRODUCTION", "both seg  4", "both seg 28"]
    rng = np.random.default_rng(2)
    n = 24
    for i in range(n):
        d = tmp_path / f"tcn_none_seed{i}" / "RQ3"
        d.mkdir(parents=True)
        prod = 0.52 + rng.normal(0, 0.01)
        arm = {"PRODUCTION": prod,
               # consistently better by a hair
               "both seg  4": prod + 0.004,
               # better on average, worse in most variants: two big wins carry it
               "both seg 28": prod + (0.20 if i < 2 else -0.005)}
        (d / "readout_interaction.json").write_text(json.dumps({
            "variant": "tcn/none", "seed": i, "DSSL": arm, "Random-init": dict(arm)}))
    aggregate(tmp_path)
    # the SECOND table -- the first one contrasts the two arms, not the two readouts
    out = capsys.readouterr().out.split("paired per variant")[1]
    line4 = next(l for l in out.splitlines() if l.strip().startswith("both seg  4"))
    line28 = next(l for l in out.splitlines() if l.strip().startswith("both seg 28"))
    assert f"{n}/{n}" in line4, line4
    assert f" 2/{n}" in line28, line28
    assert "+0.0040" in line4


def test_an_arm_identical_to_production_reads_as_no_evidence(tmp_path, capsys):
    """`trend seg 1 + spec` IS the production readout, so it ties in every variant. Counting
    ties as losses made that the most significant row in the table."""
    for i in range(24):
        d = tmp_path / f"tcn_none_seed{i}" / "RQ3"
        d.mkdir(parents=True)
        arm = {"PRODUCTION": 0.52 + i * 1e-4, "trend seg  1 + spec": 0.52 + i * 1e-4}
        (d / "readout_interaction.json").write_text(json.dumps({
            "variant": "tcn/none", "seed": i, "DSSL": arm, "Random-init": dict(arm)}))
    aggregate(tmp_path)
    out = capsys.readouterr().out.split("paired per variant")[1]
    line = next(l for l in out.splitlines() if l.strip().startswith("trend seg  1 + spec"))
    assert "1.0000" in line and "0/0" in line, line
