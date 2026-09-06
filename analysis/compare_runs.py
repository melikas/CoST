"""Two sweeps, paired on seed, across RQ1, RQ2 and RQ3 at once.

Every comparison this project has made between two sweeps has been assembled by hand, and
one of them was wrong in a way that took weeks to find: run 2002135 was used as the baseline
for a set of runs that differed from it in `seasonal_bands` as well as in the thing under
test, so a result attributed to the treatment was partly the confound. Nothing in the
pipeline noticed, because a run directory records its configuration and nobody compared two.

So the configuration check is not a convenience here, it is the point. Both runs'
metrics.json holds vars(args); for each shared seed this compares them key by key and
refuses to print a table when anything outside --expect differs. A confounded comparison is
worse than no comparison, because it looks like a measurement.

Pairing is on seed, so the same split and the same initialisation stand on both sides and
the sign test is over matched pairs rather than two independent samples.

    python analysis/compare_runs.py --treat results_hrd/2503584 \
      --control results_hrd/2438763 results_hrd/2438765 --expect noise_weight noise_branch
"""
import sys
from pathlib import Path

# Run as `python analysis/<name>.py` from the repository root: the interpreter puts
# this file's own directory on sys.path, not the project root, so the shared modules
# would not import. scripts/ already does this; the pattern is the same.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import glob
import json

import numpy as np

from tasks.sign_test import sign_summary

# Differ between any two sweeps and say nothing about the model: where the run was written,
# which card it landed on, and what the array called it.
INCIDENTAL = {"run_id", "output_dir", "variant_dir", "cache_dir", "results_dir", "gpu",
              "job_id", "tag"}


def seed_of(variant_dir):
    name = Path(variant_dir).name
    return int(name.split("seed")[1].split("_")[0]) if "seed" in name else None


def _find(run_dirs, only=None):
    """{seed: [variant_dir, ...]} -- every directory matching, ambiguity included."""
    found = {}
    for run in run_dirs:
        for v in sorted(glob.glob(str(Path(run) / "*"))):
            s = seed_of(v)
            if s is None or not Path(v).is_dir():
                continue
            if only and only not in Path(v).name:
                continue
            found.setdefault(s, []).append(v)
    return found


def collect(run_dirs, only=None, quiet=False):
    """{seed: variant_dir} over one or more sweep directories.

    Several sweeps are accepted as one arm because a stage-0 array and its self-healed tail
    land in different folders while being the same run.

    A sweep usually holds several VARIANTS per seed -- tcn_none, tcn_none_plain, a clock
    ablation -- so a seed does not identify a run on its own. The first version of this
    keyed on seed alone and silently kept whichever directory sorted last, which made an arm
    a mixture of variants that merely shared a random seed. An ambiguous seed is refused
    here, with the candidates printed, because which variant is the comparison is a question
    only the caller can answer.
    """
    found = _find(run_dirs, only)
    bad = {s: d for s, d in found.items() if len(d) > 1}
    if bad and not quiet:
        s, dirs = sorted(bad.items())[0]
        raise SystemExit(
            f"\n  AMBIGUOUS: {len(bad)} seed(s) match more than one directory, so this arm\n"
            f"  would be a mixture of variants. Seed {s} matches:\n"
            + "".join(f"    {d}\n" for d in dirs)
            + "  Narrow it with --treat-only / --control-only (a substring of the directory\n"
              "  name, e.g. the variant tag).")
    return {s: d[0] for s, d in found.items() if len(d) == 1}


def _metrics_json(variant_dir):
    p = Path(variant_dir) / "metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def config(variant_dir):
    m = _metrics_json(variant_dir)
    return None if m is None else m.get("config", {})


def check_cohort(treat, control):
    """Both arms must have seen the same data, which the configuration cannot tell you.

    The split is a deterministic function of --seed and the post-windowing participant pool,
    and metrics.json records WHICH participants landed in test. So for a shared seed the two
    runs agree exactly, or the dataset changed underneath them -- a new CSV schema, a
    different preprocessing decision, a channel added -- while every argparse value stayed
    identical. Pairing on seed is then comparing two different cohorts and calling it a
    matched pair, and nothing else in this file would notice.
    """
    same, diff, missing = 0, [], 0
    for s in sorted(set(treat) & set(control)):
        a, b = _metrics_json(treat[s]), _metrics_json(control[s])
        if a is None or b is None or "test_pids" not in a or "test_pids" not in b:
            missing += 1
            continue
        if a["test_pids"] == b["test_pids"]:
            same += 1
        else:
            diff.append((s, len(set(a["test_pids"]) & set(b["test_pids"])),
                         len(a["test_pids"]), len(b["test_pids"])))
    print(f"  cohort: {same} seed(s) with an identical test split, {len(diff)} differing"
          + (f", {missing} unrecorded" if missing else ""))
    for s, shared, na, nb in diff[:3]:
        print(f"    seed {s}: {shared} participants in common, {na} vs {nb} in test")
    return diff


def _differences(treat, control, expected):
    """{key: {(treatment value, control value)}} over every shared seed, minus `expected`."""
    expected = set(expected) | INCIDENTAL
    off = {}
    for s in sorted(set(treat) & set(control)):
        a, b = config(treat[s]), config(control[s])
        if a is None or b is None:
            continue
        for k in set(a) | set(b):
            if k in expected or a.get(k) == b.get(k):
                continue
            off.setdefault(k, set()).add((json.dumps(a.get(k)), json.dumps(b.get(k))))
    return off


def check_configs(treat, control, expected):
    """Refuse the comparison unless the two arms differ only in `expected`."""
    off = _differences(treat, control, expected)
    n = len(set(treat) & set(control))
    print(f"\n  configuration: {n} seed pairs compared, {len(off)} unexpected difference(s)")
    for k, vals in sorted(off.items()):
        for x, y in sorted(vals)[:3]:
            print(f"    {k}: treatment {x}  vs  control {y}")
    return off


def scan(treat, pattern, expected):
    """Rank candidate controls for `treat` by how many variables they move.

    A control is not something to remember, it is something to look up. `2438763` was assumed
    to be the matched control for the V^N sweep because it carried the same `_nw0.3` tag, and
    it moves three other variables -- decomp_aug, drop_channels and smooth_bins -- every one
    of which this project has separately measured as mattering. The tag is not the
    configuration, and a ranked list is the difference between a measurement and a confound.
    """
    cand = []
    for run in sorted(glob.glob(pattern)):
        if not Path(run).is_dir():
            continue
        found = _find([run])
        other = {s: d[0] for s, d in found.items() if len(d) == 1}
        ambig = sum(1 for s, d in found.items() if len(d) > 1 and s in treat)
        shared = set(treat) & set(other)
        if not shared and not ambig:
            continue
        off = _differences(treat, other, expected)
        cand.append((len(off), -len(shared), run, len(shared), ambig, off))
    if not cand:
        raise SystemExit(f"  nothing under {pattern} shares a seed with the treatment")
    print(f"\n  {len(cand)} candidate control(s), fewest moved variables first")
    print("  'ambig' counts seeds holding several variants -- those need --control-only\n")
    print(f"  {'run':32s} {'seeds':>6s} {'ambig':>6s} {'moved':>6s}  differing keys")
    for n_off, _, run, n_sh, ambig, off in sorted(cand):
        keys = ", ".join(sorted(off)) if off else "-- matches on everything --"
        # Trimmed from the LEFT: the run id is the tail of the path, so cutting the head
        # keeps the one part of it that identifies the run.
        shown = run if len(run) <= 32 else "..." + run[-29:]
        print(f"  {shown:32s} {n_sh:6d} {ambig:6d} {n_off:6d}  {keys[:80]}")
    print("\n  Only a control moving 0 keys makes the difference below the treatment.")


def read(variant_dir, rq, key):
    p = Path(variant_dir) / rq / f"{rq.lower()}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))[key]
    except (KeyError, json.JSONDecodeError):
        return None


def metrics(variant_dir):
    """{(level, name): value} -- the numbers each RQ reports as its headline."""
    out = {}
    dec = read(variant_dir, "RQ1", "decomposition") or {}
    for k in ("rec_full_trend", "rec_full_rhythm", "DIS"):
        if k in dec:
            out[("RQ1", k)] = float(dec[k])
    for name, row in (read(variant_dir, "RQ2", "concordance") or {}).items():
        if isinstance(row, dict) and "C" in row:
            out[("RQ2", name)] = float(row["C"])
    for name, row in (read(variant_dir, "RQ3", "utility") or {}).items():
        if isinstance(row, dict) and row.get("auc") is not None:
            out[("RQ3", name)] = float(row["auc"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--treat", nargs="+", required=True)
    ap.add_argument("--control", nargs="+", help="required unless --scan is given")
    ap.add_argument("--expect", nargs="*", default=[],
                    help="configuration keys allowed to differ -- i.e. the thing under test")
    ap.add_argument("--force", action="store_true",
                    help="print the table even when the arms are confounded")
    ap.add_argument("--treat-only", metavar="SUBSTRING",
                    help="keep only variant directories whose name contains this")
    ap.add_argument("--control-only", metavar="SUBSTRING")
    ap.add_argument("--scan", metavar="GLOB",
                    help="instead of comparing, rank every run under GLOB by how many "
                         "configuration keys it moves against --treat")
    a = ap.parse_args()

    if a.scan:
        scan(collect(a.treat, a.treat_only), a.scan, a.expect)
        return
    if not a.control:
        ap.error("--control is required unless --scan is given")

    treat = collect(a.treat, a.treat_only)
    control = collect(a.control, a.control_only)
    shared = sorted(set(treat) & set(control))
    print(f"\n  treatment {len(treat)} seeds | control {len(control)} seeds "
          f"| {len(shared)} paired")
    if not shared:
        raise SystemExit("  no shared seeds -- these two sweeps cannot be paired")

    bad_cfg = check_configs(treat, control, a.expect)
    bad_cohort = check_cohort(treat, control)
    if (bad_cfg or bad_cohort) and not a.force:
        raise SystemExit(
            "\n  REFUSED: " + ("the arms differ outside --expect, so any difference below\n"
                               "  would be part treatment and part confound. Add the key to "
                               "--expect if it IS\n  the thing under test, or pick a control "
                               "that matches." if bad_cfg else
                               "a shared seed put different participants in test, so the two\n"
                               "  arms did not see the same data and pairing on seed compares "
                               "two cohorts.")
            + " --force prints it anyway.")

    rows = {}
    for s in shared:
        mt, mc = metrics(treat[s]), metrics(control[s])
        for k in set(mt) & set(mc):
            rows.setdefault(k, []).append((mt[k], mc[k]))

    if not rows:
        # Which arm is empty is the whole diagnosis: a control predating the current RQ
        # schema writes no rq1/rq2/rq3.json at all, and reporting only "no shared metric"
        # sends you looking for a mismatch in names that is really a missing file.
        s = sorted(shared)[0]
        for label, d in (("treatment", treat[s]), ("control", control[s])):
            have = sorted({k[0] for k in metrics(d)}) or ["nothing"]
            print(f"  {label} seed {s} has: {', '.join(have)}")
        raise SystemExit(
            "  no metric is present in both arms. An arm with nothing predates the current\n"
            "  RQ output schema and cannot be compared without re-running it.")

    for level in ("RQ1", "RQ2", "RQ3"):
        keys = [k for k in rows if k[0] == level]
        if not keys:
            continue
        print(f"\n  --- {level} ---")
        print(f"  {'metric':38s} {'treat':>8s} {'control':>9s} {'diff':>9s}"
              f" {'wins':>8s} {'p':>8s}")
        for k in sorted(keys, key=lambda k: -np.mean([x - y for x, y in rows[k]])):
            pairs = np.array(rows[k], dtype=float)
            d = pairs[:, 0] - pairs[:, 1]
            w, n, p = sign_summary(d)
            print(f"  {k[1][:38]:38s} {pairs[:, 0].mean():8.4f} {pairs[:, 1].mean():9.4f}"
                  f" {d.mean():+9.4f} {w:4d}/{n:<3d} {p:8.4f}")
    print("\n  Paired on seed: same split, same initialisation, one variable moved.")


if __name__ == "__main__":
    main()
