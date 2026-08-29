"""Shared reader for a training results tree (``results_hrd`` / ``results_globem``).

``train_globem.py`` is a thin wrapper over ``train_hrd.main()``, so both datasets write
the SAME layout and this one module serves both:

    <results-dir>/<run>/<variant>/metrics.json                 required
                                  decomposition_recovery.json  DIS / recovery / leak

``collect_results.py`` reads that tree through this module for both its tables and its
panels. Keeping the walk, the partial-file guard and the variant identity here is what
stops any two readers from disagreeing about what a run is.

A variant is identified by ``Variant(run, backbone, pe, holdout, label, seed)``. ``holdout``
and ``label`` belong in the key on purpose: ``--holdout`` folds test DIFFERENT populations
and ``--globem-label`` weekly/endpoint are DIFFERENT prediction targets, so pooling across
either is meaningless. HRD sweeps leave both at ``"-"``; GLOBEM sweeps set them -- which is
exactly where a key that ignored them would silently average unrelated cells together.
"""
import json
from collections import namedtuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tasks.rq_paths import rq_path

Variant = namedtuple("Variant", "run backbone pe holdout label seed")

# `probe_scores.json` used to be a second sidecar. Nothing has written it since the probe
# was rewritten, so every reader of it was dead code; it is gone rather than left to look
# like an optional input.
SIDECARS = {"d": "decomposition_recovery.json"}


def read_json(fp):
    """Parse ``fp``, or return None and say so -- a sibling array task may be mid-write."""
    try:
        return json.loads(Path(fp).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        print(f"  [skip] unreadable/partial: {fp}")
        return None


def variant_key(d, fp) -> Variant:
    """Identity of the variant that wrote ``d`` (a parsed metrics.json living at ``fp``)."""
    cfg = d.get("config", {})
    pe = d.get("pe", "?")
    if cfg.get("disentangle") is False:          # plain-SSL (no trend/season split) baseline
        pe += "_plain"
    return Variant(run=Path(fp).parent.parent.name,
                   backbone=d.get("backbone", "?"),
                   pe=pe,
                   holdout=cfg.get("holdout") or "-",
                   label=cfg.get("globem_label", "-"),
                   seed=cfg.get("seed", "?"))


def iter_metrics(results_dir, runs=None):
    """Yield ``(path, parsed metrics.json)`` for every variant, optionally limited to ``runs``."""
    root = Path(results_dir)
    paths = (sorted(root.rglob("metrics*.json")) if runs is None else
             sorted(p for r in runs for p in (root / r).glob("*/metrics*.json")))
    for fp in paths:
        d = read_json(fp)
        if d is not None:
            yield fp, d


def load_variants(results_dir, runs=None, exclude=(), drop_seeds=()):
    """-> ``{Variant: {"m": metrics, "d": decomposition, "views": probe views}}``

    ``exclude`` drops backbones; ``drop_seeds`` drops ``(run, seed)`` cells, e.g. a seed
    shared between two pooled runs. Missing sidecars leave their entry as an empty dict.
    """
    out = {}
    for fp, m in iter_metrics(results_dir, runs):
        k = variant_key(m, fp)
        if k.backbone in exclude or (k.run, k.seed) in drop_seeds:
            continue
        rec = {"m": m, "d": {}}
        for key, name in SIDECARS.items():
            side = rq_path(fp.parent, name, create=False)
            j = read_json(side) if side.exists() else None
            if j is not None:
                rec[key] = j
        out[k] = rec
    return out


def seed_cells(variants):
    """Independent observation identity: the key minus the model (backbone, pe).

    Two variants share a cell only when they were trained on the same fold, for the same
    target, from the same seed -- so paired contrasts stay within-cell.
    """
    return sorted({(k.run, k.holdout, k.label, k.seed) for k in variants})


def in_cell(k, cell):
    """True when variant ``k`` belongs to the ``seed_cells`` entry ``cell``."""
    return (k.run, k.holdout, k.label, k.seed) == cell


def census(variants, results_dir=""):
    """Print a one-line census and warn loudly when a tree mixes folds or targets."""
    cells = seed_cells(variants)
    models = {(k.backbone, k.pe) for k in variants}
    print(f"{results_dir or 'results'}: {len(variants)} variants, {len(cells)} independent "
          f"cells, {len(models)} models")
    for field, name in (("holdout", "--holdout folds"), ("label", "--globem-label targets")):
        vals = sorted({getattr(k, field) for k in variants})
        if len(vals) > 1:
            print(f"  [warning] this tree mixes {len(vals)} {name} ({', '.join(map(str, vals))}); "
                  f"they test different data and are kept as separate cells, never pooled")
    return cells
