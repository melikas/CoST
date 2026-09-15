"""HRD windowed dataset: the public entry point of the HRD preprocessing.

    CSV -> hrd_clean.iter_hrd()               clean columns, drop participants, fill short gaps
        -> hrd_windows._window_participant()  fixed windows, quality gate, binning
        -> prepare_hrd_dataset()              stack into (N, T, C), attach labels and energy

`prepare_hrd_dataset(csv)` returns X (N, T, C), y, pids, window_ids, ee_win (mean daily
emotional energy over the window's 7 calendar dates), label metadata and the sensor
columns. `drop_sensor_channels` removes channels after windowing. Thresholds and column
names live in hrd_config.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data_processing.hrd_clean import _baseline_status_by_pid, _endpoint_labels_and_consistent_pids, _trajectory_by_pid, iter_hrd
from data_processing.hrd_config import CHANNELS, CLEANING, DailyEnergyLookup, PathLike, WINDOWING
from data_processing.hrd_windows import _window_participant, standardise_clock_channels


def _group_energy_by_participant(
    energy_by_pid_date: DailyEnergyLookup,
) -> Dict[str, List[Tuple[pd.Timestamp, float]]]:
    """Reshape the (pid, date) lookup into pid -> [(day, energy), ...]."""
    by_participant: Dict[str, List[Tuple[pd.Timestamp, float]]] = {}
    for (pid, day), value in energy_by_pid_date.items():
        by_participant.setdefault(pid, []).append((pd.Timestamp(day), value))
    return by_participant


def _mean_energy_in_range(
    day_energy_pairs: Sequence[Tuple[Any, float]],
    start: pd.Timestamp,
    end: pd.Timestamp,
    include_end: bool,
) -> float:
    """Mean emotional energy over the answered days a window spans; NaN when there are none.

    Emotional energy is a per-DAY survey while these windows are 7 days long, so scoring a
    single day's value against a week of input would leave six sevenths of the input saying
    nothing about the target. The mean is the only target on the window's own timescale.
    """
    values: List[float] = []
    for day, energy in day_energy_pairs:
        day = pd.Timestamp(day)
        in_range = (start <= day <= end) if include_end else (start <= day < end)
        if in_range and np.isfinite(energy):
            values.append(energy)
    return float(np.mean(values)) if values else np.nan


def _resolve_sensor_cols(sensor_df: pd.DataFrame) -> List[str]:
    """The configured sensor channels that are actually present in the loaded table."""
    missing = set(CHANNELS.sensors) - set(sensor_df.columns)
    if missing:
        raise ValueError(f"missing required HRD sensor channels: {sorted(missing)}")
    return list(CHANNELS.sensors)


def prepare_hrd_dataset(
    csv_path: PathLike,
    window_hours: int = WINDOWING.window_hours,
    bin_minutes: int = WINDOWING.bin_minutes,
    label_col: str = WINDOWING.label_col,
    max_missing: float = CLEANING.max_participant_missing,
    max_gap_minutes: int = CLEANING.max_gap_minutes,
    max_window_missing: float = CLEANING.max_window_missing,
    z_score: bool = WINDOWING.z_score,
    clock_features: bool = False,
    calendar_index: bool = False,
    align_midnight: bool = True,
    cache_raw: bool = False,
    keep_observed: bool = True,
) -> Dict[str, object]:
    """Full HRD preprocessing -> non-overlapping windowed classification dataset for CoST.

    `clock_features` (default False) keeps sensor channels only, i.e. time is excluded from
    the model entirely. Set it True to append the CoST calendar covariates, scaled by the
    fixed CLOCK_MU / CLOCK_SD so the transform is identical here and in the sliding path.

    Returns a dict with:
      X                 (N, T, C) float32 windows
      y                 (N,)  int endpoint labels (0 = control, 1 = depressed)
      pids              (N,)  participant id per window
      window_ids        (N,)  unique "pid_<isotime>" per window
      ee_win            (N,)  mean emotional energy over the days the window spans (NaN if none)
      consistent_pids   set   pids with baseline == endpoint
      labeled_pids      set   pids that carry an endpoint label
      trajectory_by_pid dict  pid -> Case-1 group (Pre1_Post1 ... Pre2_Post2)
      baseline_by_pid   dict  pid -> baseline depression status
      sensor_cols, n_sensors, n_features
    """
    if cache_raw:
        raise ValueError("raw-frame caching is disabled; use the versioned window cache")
    sensor_cols = list(CHANNELS.sensors)
    window_minutes = window_hours * 60
    window_span = pd.Timedelta(minutes=window_minutes)
    label_by_pid, consistent_pids, labels_collected = {}, set(), []
    validation = {}

    started_at = time.time()
    all_windows: List[np.ndarray] = []
    all_physical: List[np.ndarray] = []
    all_labels: List[int] = []
    all_pids: List[str] = []
    # None unless asked for, so the windower's inner call is a no-op and X is unchanged.
    all_observed: Optional[List[np.ndarray]] = [] if keep_observed else None
    all_window_ids: List[str] = []
    all_window_energy: List[float] = []

    for participant_rows, labels, energy in iter_hrd(
            csv_path, max_missing=max_missing, max_gap_minutes=max_gap_minutes, summary=validation):
        pid = str(participant_rows.pid.iloc[0])
        _resolve_sensor_cols(participant_rows)
        participant_labels, stable = _endpoint_labels_and_consistent_pids(labels, label_col)
        label_by_pid.update(participant_labels)
        consistent_pids.update(stable)
        labels_collected.append(labels)
        energy_by_pid = _group_energy_by_participant(energy)
        mu = participant_rows[sensor_cols].mean().to_numpy(dtype=np.float32)
        sd = participant_rows[sensor_cols].std().replace(0., 1.).fillna(1.).to_numpy(dtype=np.float32)
        label = label_by_pid.get(pid, -1)
        for window, window_start in _window_participant(
            participant_rows, window_minutes, bin_minutes, sensor_cols, max_window_missing,
            z_score, clock_features=clock_features, calendar_index=calendar_index,
            align_midnight=align_midnight, observed_out=all_observed,
        ):
            all_windows.append(window)
            physical = window[:, :len(sensor_cols)]
            all_physical.append((physical * sd + mu) if z_score else physical.copy())
            all_labels.append(label)
            all_pids.append(pid)
            all_window_ids.append(f"{pid}_{window_start.isoformat()}")
            # The 7 calendar dates from the window's start date: the same days whatever
            # hour the window starts at.
            day0 = pd.Timestamp(window_start).normalize()
            all_window_energy.append(_mean_energy_in_range(
                energy_by_pid.get(pid, []), start=day0, end=day0 + window_span,
                include_end=False,
            ))
        if validation["raw_participants"] % 10 == 0:
            print(f"[hrd] processed {validation['raw_participants']} participants; "
                  f"{len(all_windows)} windows", flush=True)

    if not all_windows:
        raise RuntimeError("No windows produced; check window/bin/missing settings.")

    X = np.stack(all_windows, axis=0).astype(np.float32)
    label_df = pd.concat(labels_collected, ignore_index=True)
    validation["participants_with_windows"] = len(set(all_pids))
    validation["windows"] = len(all_windows)
    if clock_features:
        standardise_clock_channels(X, len(sensor_cols))
    n_features = X.shape[-1]

    n_depressed = int(np.sum(np.asarray(all_labels) == 1))
    print(
        f"[hrd] built {len(all_windows):,} windows of shape "
        f"(T={X.shape[1]}, C={n_features}) from {len(set(all_pids))} participants | "
        f"{n_depressed} depressed-endpoint windows | {time.time() - started_at:.1f}s"
    )

    return {
        "X": X,
        "raw_X": np.stack(all_physical).astype(np.float32),
        "validation": validation,
        "y": np.asarray(all_labels, dtype=int),
        "pids": np.asarray(all_pids),
        "window_ids": np.asarray(all_window_ids),
        "ee_win": np.asarray(all_window_energy, dtype=float),
        "consistent_pids": consistent_pids,
        "labeled_pids": set(label_by_pid) & set(all_pids),
        "trajectory_by_pid": _trajectory_by_pid(label_df),
        "baseline_by_pid": _baseline_status_by_pid(label_df),
        "sensor_cols": sensor_cols,
        "n_sensors": len(sensor_cols),
        "n_features": n_features,
        # (N, T, n_sensors) bool, aligned row for row with X. Absent unless keep_observed.
        **({"observed": np.stack(all_observed, axis=0)} if all_observed is not None else {}),
    }


def drop_sensor_channels(data, names):
    """Remove named sensor channels from an already-built dataset.

    Only for an explicitly named sensor ablation. Applied after windowing so eligibility
    is shared with the full-sensor control. Clock channels follow the sensor channels.
    """
    if not names:
        return data
    names = [names] if isinstance(names, str) else list(names)
    cols = list(data["sensor_cols"])
    unknown = [n for n in names if n not in cols]
    if unknown:
        raise ValueError(f"drop_sensor_channels: no such channel {unknown}; have {cols}")
    n = int(data["n_sensors"])
    keep = [i for i in range(n) if cols[i] not in names]
    if not keep:
        raise ValueError("drop_sensor_channels: that would remove every sensor channel")
    idx = keep + list(range(n, data["X"].shape[2]))
    data = dict(data)
    data["X"] = data["X"][:, :, idx]
    if "observed" in data:
        data["observed"] = data["observed"][:, :, keep]
    if "raw_X" in data:
        data["raw_X"] = data["raw_X"][:, :, keep]
    data["sensor_cols"] = [cols[i] for i in keep]
    data["n_sensors"] = len(keep)
    print(f"[data] dropped {names} -> {data['n_sensors']} sensor channels "
          f"{data['sensor_cols']}, X{data['X'].shape}")
    return data
