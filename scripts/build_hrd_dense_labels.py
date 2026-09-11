"""Per-window emotional energy and per-participant CES-D for eval.py --mil, aligned to an
existing HRD window cache.

The window cache carries one binary label per participant. The raw CSV also carries the
continuous CES-D endpoint score behind that label and a daily emotional-energy survey; this
turns both into arrays aligned to the cache's windows, so eval.py never has to read the
53.5M-row CSV. A window's energy is the mean over the survey days it spans, with exactly
the definition prepare_hrd_dataset uses: start <= day < start + window span.

    python scripts/build_hrd_dense_labels.py --csv datasets/HRD_RAW_MinuteLevel.csv \
        --npz hrd_2224103.npz --out hrd_dense_labels.npz

This file only ALIGNS labels. Which participants' labels may be read is decided per fold in
eval.probe_auc_mil, which reads them for that fold's training participants only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_loader import load_npz                                    # noqa: E402
from data_processing.data_preprocessing import (                    # noqa: E402
    _build_daily_energy_lookup, _build_label_table, _group_energy_by_participant,
    _mean_energy_in_range, _read_raw_csv)


def build(df, coh):
    """(ee_window, pid, cesd_endpoint, status_endpoint) from a raw-CSV frame and a cache."""
    labels = _build_label_table(df)
    energy = _group_energy_by_participant(_build_daily_energy_lookup(df))
    span = pd.Timedelta(minutes=coh.seq_len * coh.bin_minutes)
    starts = [pd.Timestamp(str(w).rsplit("_", 1)[1]) for w in coh.window_ids]
    ee = np.array([_mean_energy_in_range(energy.get(str(p), []), s, s + span,
                                         include_end=False)
                   for p, s in zip(coh.pids, starts)], dtype=np.float64)
    pid = labels["pid"].astype(str).to_numpy()
    cesd = pd.to_numeric(labels["ces_d_endpoint_score"], errors="coerce").to_numpy(np.float64)
    status = pd.to_numeric(labels["depression_status_endpoint"],
                           errors="coerce").to_numpy(np.float64)
    return ee, pid, cesd, status


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="HRD_RAW_MinuteLevel.csv")
    p.add_argument("--npz", required=True, help="the window cache eval.py will be run on")
    p.add_argument("--out", default="hrd_dense_labels.npz")
    a = p.parse_args(argv)

    coh = load_npz(a.npz)
    if coh.window_ids is None:
        raise SystemExit(f"{a.npz} has no window_ids, so its windows cannot be dated")
    ee, pid, cesd, status = build(_read_raw_csv(a.csv), coh)

    # The CSV must describe the cache's labelled cohort exactly: every labelled participant
    # present, and its endpoint status equal to the label the cache carries. Either failing
    # means the two files do not describe the same people, and nothing downstream is valid.
    upids, ulab = coh.participants()
    row = {q: i for i, q in enumerate(pid)}
    missing = [str(q) for q in upids if str(q) not in row]
    if missing:
        raise SystemExit(f"{len(missing)} labelled participants are absent from the CSV: "
                         f"{missing[:5]}")
    mism = [str(q) for q, lab in zip(upids, ulab) if status[row[str(q)]] != lab]
    if mism:
        raise SystemExit(f"{len(mism)} participants' CSV endpoint status disagrees with the "
                         f"cache label: {mism[:5]}")

    labelled = np.isin(coh.pids, list(coh.labelled))
    has_c = np.array([np.isfinite(cesd[row[str(q)]]) for q in upids])
    print(f"windows {len(ee)}; with an energy target {int(np.isfinite(ee).sum())} "
          f"({np.isfinite(ee[labelled]).mean():.1%} of labelled participants' windows)")
    print(f"labelled participants {len(upids)}; with a CES-D endpoint score {int(has_c.sum())}")
    np.savez_compressed(a.out, window_ids=np.asarray(coh.window_ids).astype(str),
                        ee_window=ee, pid=pid, cesd_endpoint=cesd, status_endpoint=status)
    print(f"[out] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
