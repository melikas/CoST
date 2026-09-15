"""Build the window cache that train.py and eval.py read (datautils.load_npz).

    python scripts/build_cache.py hrd --csv datasets/HRD_RAW_MinuteLevel.csv \
        --out datasets/cache/hrd_rescue_v1.npz --energy-out datasets/cache/hrd_energy_rescue_v1.npz
    python scripts/build_cache.py globem --csv datasets/GLOBEM_REDUCED.csv \
        --out datasets/cache/globem_rescue_v1.npz

The cache holds X (N, T, C), y, pids, window_ids, sensor_cols, n_sensors, bins_per_day,
bin_minutes and train/val/test masks. The masks only identify the labelled cohort
(datautils.load_npz takes their union, so unlabelled participants are never read as negatives);
datautils.make_folds does all splitting, from the master seed.

All configured sensors are retained unless --exclude-sensors names an explicit ablation.
`--energy-out` (HRD) writes the weekly emotional energy aligned to window_ids, which
scripts/rq3_bridge.py reads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MASK_SEEDS = (7, 13, 29)
PREPROCESSING_VERSION = "rescue-v1"


def build_masks(pids, labelled, seeds=MASK_SEEDS, val_frac=0.2, test_frac=0.2):
    """train/val/test window masks per seed, whose union is exactly the labelled windows."""
    is_lab = np.isin(pids, sorted(labelled))
    out = {}
    for s in seeds:
        rng = np.random.default_rng(s)
        who = np.array(sorted(labelled))
        rng.shuffle(who)
        n_te = max(1, int(len(who) * test_frac))
        n_va = max(1, int(len(who) * val_frac))
        te, va = set(who[:n_te]), set(who[n_te:n_te + n_va])
        in_te = is_lab & np.isin(pids, sorted(te))
        in_va = is_lab & np.isin(pids, sorted(va))
        out[f"test_mask/{s}"] = in_te
        out[f"val_mask/{s}"] = in_va
        out[f"train_mask/{s}"] = is_lab & ~in_te & ~in_va
    return out


def windows(a):
    """(dataset dict, bins_per_day) for the requested cohort."""
    if a.dataset == "hrd":
        from data_processing.hrd_config import WINDOWING
        from data_processing.hrd_dataset import prepare_hrd_dataset
        d = prepare_hrd_dataset(a.csv)
        return d, 1440 // WINDOWING.bin_minutes
    from data_processing.globem_dataset import prepare_globem_dataset
    d = prepare_globem_dataset(a.csv, window_days=a.window_days, stride_days=a.stride_days,
                               min_window_coverage=a.min_window_coverage)
    return d, 4


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=["hrd", "globem"])
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--energy-out", default=None, help="HRD only: weekly emotional energy")
    p.add_argument("--window-days", type=int, default=28, help="GLOBEM window length")
    p.add_argument("--stride-days", type=int, default=7, help="GLOBEM window stride")
    p.add_argument("--min-window-coverage", type=float, default=0.5, help="GLOBEM")
    p.add_argument("--exclude-sensors", nargs="*", default=[],
                   help="explicit ablation after full-sensor window eligibility; names match sensor_cols")
    a = p.parse_args(argv)
    if a.energy_out and a.dataset != "hrd":
        p.error("--energy-out is HRD-only")

    outputs = [Path(a.out), Path(a.out).with_suffix(".json")]
    if a.energy_out:
        outputs.append(Path(a.energy_out))
    if any(p.exists() for p in outputs):
        p.error("an output already exists; choose a new versioned cache path")
    print(f"[cache] reading {a.csv}")
    d, bpd = windows(a)
    if a.exclude_sensors:
        from data_processing.hrd_dataset import drop_sensor_channels
        d = drop_sensor_channels(d, a.exclude_sensors)
    X, y, pids, wid = d["X"], d["y"], d["pids"], d["window_ids"]
    labelled = set(d["labeled_pids"])
    for q in sorted(labelled):
        if len(np.unique(y[pids == q])) != 1:
            raise SystemExit(f"participant {q} carries more than one label")

    digest = hashlib.sha256()
    with open(a.csv, "rb") as raw:
        for block in iter(lambda: raw.read(8 * 1024 * 1024), b""):
            digest.update(block)
    metadata = {"preprocessing_version": PREPROCESSING_VERSION, "dataset": a.dataset,
                "source_file": Path(a.csv).name, "source_sha256": digest.hexdigest(),
                "label_column": "depression_status_endpoint" if a.dataset == "hrd" else "LABEL_ENDPOINT",
                "label_derivation": "exported endpoint; instrument/cutoff provenance pending",
                "normalization": "retrospective_full_participant_record",
                "timestamp_timezone": "export-local; timezone/DST provenance unresolved",
                "sensor_cols": list(d["sensor_cols"]), "excluded_sensors": a.exclude_sensors,
                "shape": list(X.shape), "participants": len(set(pids)),
                "labelled_participants": len(labelled), "unknown_label": -1,
                "bins_per_day": bpd, "window_days": X.shape[1] / bpd,
                "stride_days": 7 if a.dataset == "hrd" else a.stride_days,
                "observed_mask": "per-channel, before imputation",
                "raw_X": "completed sensor windows in original feature units; imputation indicated by observed",
                "validation": d.get("validation", {}),
                "missing_fraction_by_channel": (1 - d["observed"].mean(axis=(0, 1))).tolist(),
                "person_linkage": "participant" if a.dataset == "hrd" else "participant-year; cross-year linkage unavailable"}
    for target in outputs:
        target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, X=X, y=y, pids=pids, window_ids=wid,
                        sensor_cols=np.asarray(d["sensor_cols"]), n_sensors=int(d["n_sensors"]),
                        bins_per_day=bpd, bin_minutes=1440 // bpd,
                        observed=d["observed"], raw_X=d["raw_X"], labelled_pids=np.asarray(sorted(labelled)),
                        metadata_json=json.dumps(metadata, sort_keys=True),
                        **build_masks(pids, labelled))
    Path(a.out).with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if a.energy_out:
        np.savez(a.energy_out, ee=np.asarray(d["ee_win"], dtype=float), window_ids=wid)

    from datautils import load_npz
    c = load_npz(a.out)
    ids, lab = c.participants()
    assert set(ids) == labelled, "the loader's labelled cohort disagrees with labeled_pids"
    print(f"[cache] {a.out}: X {X.shape}, {len(set(pids.tolist()))} participants, "
          f"{len(ids)} labelled, prevalence {lab.mean():.4f}, channels {list(d['sensor_cols'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
