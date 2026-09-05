"""The paired readout comparison: right pairing, right counts, ties dropped."""
import subprocess
import sys
from pathlib import Path

HEADER = "role,representation,auc,ci_lo,ci_hi,balanced_acc\n"


def _table(rows):
    return HEADER + "".join(
        f"ladder,{n},{auc},,,{ba}\n" for n, auc, ba in rows)


def _tree(tmp, per_variant):
    """per_variant: list of (before_rows, after_rows), one entry per variant directory."""
    for i, (before, after) in enumerate(per_variant):
        d = tmp / "results_globem" / "2240054" / f"tcn_none_seed{i}" / "RQ3"
        d.mkdir(parents=True)
        (d / "rq3_utility_meanpool.csv").write_text(_table(before))
        (d / "rq3_utility.csv").write_text(_table(after))
    return str(tmp / "results_globem" / "*" / "*" / "RQ3" / "rq3_utility.csv")


def run(glob_pat):
    out = subprocess.run(
        [sys.executable, "scripts/compare_readouts.py", "rq3_utility_meanpool.csv",
         "--glob", glob_pat],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_consistent_gain_reads_as_unanimous(tmp_path):
    g = _tree(tmp_path, [([("DSSL (frozen)", 0.55, 0.53)],
                          [("DSSL (frozen)", 0.57, 0.55)]) for _ in range(12)])
    line = next(l for l in run(g).splitlines() if "DSSL (frozen)" in l)
    assert "+0.0200" in line and "12/12" in line


def test_an_unchanged_arm_is_not_significant(tmp_path):
    """Ties must be dropped: an arm that did not move reads 0/0 and p=1, not 0/12."""
    g = _tree(tmp_path, [([("Cosinor (paper)", 0.56, 0.54)],
                          [("Cosinor (paper)", 0.56, 0.54)]) for _ in range(12)])
    line = next(l for l in run(g).splitlines() if "Cosinor (paper)" in l)
    assert "0/0" in line and "1.0000" in line
    assert "+0.0000" in line


def test_arms_in_only_one_file_are_named_not_dropped_silently(tmp_path):
    """The real case: the rerun dropped a rung and added one, and both must be visible
    rather than quietly missing from a table that otherwise looks complete."""
    g = _tree(tmp_path, [([("Shared", 0.55, 0.53), ("Old arm", 0.55, 0.53)],
                          [("Shared", 0.57, 0.55), ("New arm", 0.57, 0.55)])
                         for _ in range(4)])
    out = run(g)
    assert "present in only one" in out
    assert "Old arm" in out and "New arm" in out
    assert next(l for l in out.splitlines() if l.strip().startswith("Shared"))


def test_refuses_when_nothing_overlaps(tmp_path):
    """No shared arm means no paired comparison exists, and printing an empty table would
    look like a result."""
    g = _tree(tmp_path, [([("Old arm", 0.55, 0.53)], [("New arm", 0.57, 0.55)])
                         for _ in range(4)])
    out = subprocess.run(
        [sys.executable, "scripts/compare_readouts.py", "rq3_utility_meanpool.csv",
         "--glob", g], capture_output=True, text=True,
        cwd=Path(__file__).resolve().parent.parent)
    assert out.returncode != 0 and "no arm appears in both" in out.stderr


def test_a_gain_carried_by_a_few_variants_does_not_read_as_unanimous(tmp_path):
    """Mean up, most variants down -- exactly what reading two tables of means would hide."""
    pv = [([("DSSL (frozen)", 0.55, 0.53)],
           [("DSSL (frozen)", 0.55, 0.53 + (0.30 if i < 2 else -0.01))])
          for i in range(12)]
    line = next(l for l in run(_tree(tmp_path, pv)).splitlines() if "DSSL (frozen)" in l)
    assert " 2/12" in line, line
