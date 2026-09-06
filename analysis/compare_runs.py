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


def collect(run_dirs, quiet=False):
    """{seed: variant_dir} over one or more sweep directories.

    Several sweeps are accepted as one control because a stage-0 array and its self-healed
    tail land in different folders while being the same run; a seed appearing twice with
    different configurations is a mistake, and is reported rather than silently resolved.
    `quiet` is for --scan, which reads dozens of runs and would bury its own table.
    """
    out, seen = {}, {}
    for run in run_dirs:
        for v in sorted(glob.glob(str(Path(run) / "*"))):
            if not Path(v).is_dir():
                continue
            s = seed_of(v)
            if s is None:
                continue
            if s in out:
                seen.setdefault(s, [out[s]]).append(v)
            out[s] = v
    for s, dirs in sorted(seen.items()):
        if not quiet:
            print(f"  NOTE seed {s} appears in {len(dirs)} directories; using {out[s]}")
    return out


def config(variant_dir):
    p = Path(variant_dir) / "metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("config", {})


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
        other = collect([run], quiet=True)
        shared = set(treat) & set(other)
        if not shared:
            continue
        off = _differences(treat, other, expected)
        cand.append((len(off), -len(shared), run, len(shared), off))
    if not cand:
        raise SystemExit(f"  nothing under {pattern} shares a seed with the treatment")
    print(f"\n  {len(cand)} candidate control(s), fewest moved variables first\n")
    print(f"  {'run':32s} {'seeds':>6s} {'moved':>6s}  differing keys")
    for n_off, _, run, n_sh, off in sorted(cand):
        keys = ", ".join(sorted(off)) if off else "-- matches on everything --"
        # Trimmed from the LEFT: the run id is the tail of the path, so cutting the head
        # keeps the one part of it that identifies the run.
        shown = run if len(run) <= 32 else "..." + run[-29:]
        print(f"  {shown:32s} {n_sh:6d} {n_off:6d}  {keys[:88]}")
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
    ap.add_argument("--scan", metavar="GLOB",
                    help="instead of comparing, rank every run under GLOB by how many "
                         "configuration keys it moves against --treat")
    a = ap.parse_args()

    if a.scan:
        scan(collect(a.treat), a.scan, a.expect)
        return
    if not a.control:
        ap.error("--control is required unless --scan is given")

    treat, control = collect(a.treat), collect(a.control)
    shared = sorted(set(treat) & set(control))
    print(f"\n  treatment {len(treat)} seeds | control {len(control)} seeds "
          f"| {len(shared)} paired")
    if not shared:
        raise SystemExit("  no shared seeds -- these two sweeps cannot be paired")

    if check_configs(treat, control, a.expect) and not a.force:
        raise SystemExit(
            "\n  REFUSED: the arms differ outside --expect, so any difference below would be\n"
            "  part treatment and part confound. Add the key to --expect if it IS the thing\n"
            "  under test, or pick a control that matches. --force prints it anyway.")

    rows = {}
    for s in shared:
        mt, mc = metrics(treat[s]), metrics(control[s])
        for k in set(mt) & set(mc):
            rows.setdefault(k, []).append((mt[k], mc[k]))

    if not rows:
        raise SystemExit("  no metric is present in both arms -- check the RQ json files")

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
