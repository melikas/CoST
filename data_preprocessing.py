"""HRD wearable-data preprocessing for the CoST project.

The raw file ``HRD_RAW_MinuteLevel.csv`` stores BOTH the minute-level sensor
signals AND the depression labels in one table (labels repeated on every row of
a participant). This module turns it into a windowed dataset that the CoST model
(``cost.py``) can train on for depression-endpoint classification.

Two public entry points:

  * ``load_hrd(csv)``          -> (sensor_df, label_df), the cleaned tables.
  * ``prepare_hrd_dataset(csv)`` -> dict with the windowed tensor ``X`` (N, T, C),
                                  labels ``y``, participant ids ``pids`` and the
                                  set of consistent (baseline==endpoint) pids.

Each channel is cleaned according to its nature (digital-health best practice):
  * HR              - physiological, continuous; NaN = non-wear -> short gaps are
                      interpolated, long gaps stay NaN (sparse windows are dropped).
  * activity counts - non-negative, NaN = non-wear -> clipped at 0 + interpolated.
  * calls / screen  - a missing value just means "no event" -> filled with 0.
  * sleep_level     - an ordinal sleep STAGE; absence means "awake" -> 0, never
                      interpolated.
Per-participant z-scoring is applied at the windowing step, so no statistics are
shared across participants or with the labels (leakage-free).
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Channel groups by physical nature.
HR_COLS = ["HR"]
ACTIVITY_COLS = ["Steps", "Floors", "Fairly_Active", "Lightly_Active",
                 "Sedentary", "Very_Active"]
EVENT_COLS = ["calls", "screen"]          # absence == 0 event, not "missing"
SLEEP_COLS = ["sleep_level"]              # ordinal stage; absence == awake (0)

WEAR_COLS = HR_COLS + ACTIVITY_COLS       # reflect whether the watch was worn
SENSOR_COLS = HR_COLS + ACTIVITY_COLS + EVENT_COLS + SLEEP_COLS  # full 10, fixed order
LABEL_COLS = ["depression_status_baseline", "depression_status_endpoint"]

HR_VALID_RANGE = (20.0, 250.0)            # bpm outside this is a sensor error -> NaN
NUM_TIME_FEATURES = 5                     # clock marks appended to every window


def _first_valid(s: pd.Series):
    """First non-null value of a column (labels repeat, but some rows are blank)."""
    v = s.dropna()
    return v.iloc[0] if len(v) else np.nan


# =============================================================================
# 1. CSV -> cleaned tables
# =============================================================================

def load_hrd(csv_path: str, max_missing: float = 0.30, max_gap_minutes: int = 30
             ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read HRD_RAW_MinuteLevel.csv and return (sensor_df, label_df)."""
    t0 = time.time()
    usecols = ["pid", "dateTime"] + SENSOR_COLS + LABEL_COLS
    df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)

    # Identifier + timestamp, then drop rows with no parseable time.
    df["pid"] = df["pid"].astype(str).str.lower().str.strip()
    df["timestamp"] = pd.to_datetime(df["dateTime"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["pid", "timestamp"])
    for c in SENSOR_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

    # Per-sensor range cleaning.
    lo, hi = HR_VALID_RANGE
    df["HR"] = df["HR"].where((df["HR"] >= lo) & (df["HR"] <= hi))      # bad HR -> NaN
    df[ACTIVITY_COLS] = df[ACTIVITY_COLS].clip(lower=0)                 # counts are >= 0

    # Structural absence (no event / awake) is a real 0, not a missing value.
    df[EVENT_COLS + SLEEP_COLS] = df[EVENT_COLS + SLEEP_COLS].fillna(0.0)

    # One label row per participant (labels are constant within a pid).
    label_df = df.groupby("pid")[LABEL_COLS].agg(_first_valid).reset_index()

    # Quality filter: drop participants with > max_missing wear-channel missingness.
    wear_missing = df[WEAR_COLS].isna().groupby(df["pid"]).mean().mean(axis=1)
    keep_pids = wear_missing.index[wear_missing <= max_missing]
    n_total = df["pid"].nunique()
    df = df[df["pid"].isin(keep_pids)]
    label_df = label_df[label_df["pid"].isin(keep_pids)].reset_index(drop=True)

    # Gap-limited interpolation of the continuous wear channels: fill only short
    # non-wear gaps (<= max_gap_minutes; data is 1 sample/minute) and leave long
    # gaps as NaN for the windowing step to discard.
    df[WEAR_COLS] = df.groupby("pid")[WEAR_COLS].transform(
        lambda s: s.interpolate(method="linear", limit=max_gap_minutes, limit_area="inside")
    )
    df[ACTIVITY_COLS] = df[ACTIVITY_COLS].clip(lower=0)

    sensor_df = df[["pid", "timestamp"] + SENSOR_COLS].reset_index(drop=True)
    n_consistent = int(
        (label_df["depression_status_baseline"]
         == label_df["depression_status_endpoint"]).sum()
    )
    print(
        f"[data_preprocessing] {len(sensor_df):,} rows | kept {len(keep_pids)}/{n_total} "
        f"participants (dropped {n_total - len(keep_pids)} with >{max_missing:.0%} wear-missing) | "
        f"{n_consistent} with baseline==endpoint | {time.time() - t0:.1f}s"
    )
    return sensor_df, label_df


def _label_maps(label_df: pd.DataFrame, label_col: str) -> Tuple[Dict[str, int], set]:
    """Return (label_by_pid, consistent_pids) from the participant label table."""
    label_by_pid: Dict[str, int] = {}
    for _, row in label_df.iterrows():
        v = row.get(label_col)
        if pd.notna(v):
            label_by_pid[row["pid"]] = int(v)
    base, end = "depression_status_baseline", "depression_status_endpoint"
    consistent = set(label_df.loc[label_df[base] == label_df[end], "pid"])  # NaN==NaN is False
    return label_by_pid, consistent


# =============================================================================
# 2. cleaned tables -> windowed tensor
# =============================================================================

def _clock_time_features(win_start, target_bins: int, bin_minutes: int) -> np.ndarray:
    """Clock-aligned time marks per bin: (target_bins, NUM_TIME_FEATURES)."""
    bins = np.arange(target_bins, dtype=np.float64)
    start_min = win_start.dayofweek * 1440.0 + win_start.hour * 60.0 + win_start.minute
    abs_min = start_min + bins * bin_minutes
    hod = (abs_min % 1440.0) / 60.0
    dow = (abs_min % (1440.0 * 7)) / 1440.0
    return np.stack(
        [
            bins / max(target_bins, 1),
            np.sin(2 * np.pi * hod / 24.0), np.cos(2 * np.pi * hod / 24.0),
            np.sin(2 * np.pi * dow / 7.0),  np.cos(2 * np.pi * dow / 7.0),
        ],
        axis=1,
    ).astype(np.float32)


def _window_participant(g: "pd.DataFrame", window_minutes: int, bin_minutes: int,
                        sensor_cols: List[str], max_window_missing: float,
                        z_score: bool) -> List[Tuple[np.ndarray, "pd.Timestamp"]]:
    """Vectorised fixed-size binning of one participant into [sensor|time] windows."""
    out: List[Tuple[np.ndarray, "pd.Timestamp"]] = []
    target_bins = window_minutes // bin_minutes
    n_sensors = len(sensor_cols)
    if z_score:                                       # per-participant, leakage-free
        g = g.copy()
        mu = g[sensor_cols].mean()
        sd = g[sensor_cols].std().replace(0.0, 1.0).fillna(1.0)
        g[sensor_cols] = (g[sensor_cols] - mu) / sd
    ts = g["timestamp"]
    if len(ts) < 2:
        return out
    start = ts.iloc[0]
    delta_min = (ts - start).dt.total_seconds().to_numpy() / 60.0
    num_windows = int(float(delta_min[-1]) // window_minutes)
    if num_windows < 1:
        return out
    win_idx = (delta_min // window_minutes).astype(np.int64)
    keep = win_idx < num_windows
    if not keep.any():
        return out
    sub = g.loc[keep, sensor_cols].copy()
    w = win_idx[keep]
    within = delta_min[keep] - w * window_minutes
    sub["_w"] = w
    sub["_b"] = np.clip((within // bin_minutes).astype(np.int64), 0, target_bins - 1)
    win_sizes = sub.groupby("_w").size()
    grouped = sub.groupby(["_w", "_b"])
    mean_df = grouped[sensor_cols].mean()
    count_df = grouped[sensor_cols].count()
    windows_with_data = set(mean_df.index.get_level_values(0))
    window_size = pd.Timedelta(minutes=window_minutes)
    for wi in range(num_windows):
        if int(win_sizes.get(wi, 0)) <= 10:
            continue
        sensor_values = np.full((target_bins, n_sensors), np.nan, dtype=np.float32)
        observed = np.zeros((target_bins, n_sensors), dtype=bool)
        if wi in windows_with_data:
            wm = mean_df.loc[wi]
            wc = count_df.loc[wi]
            for bi in wm.index:
                obs = wc.loc[bi].to_numpy() > 0
                sensor_values[bi] = np.where(obs, wm.loc[bi].to_numpy(dtype=np.float32), np.nan)
                observed[bi] = obs
        if float((observed.sum(axis=1) == 0).mean()) > max_window_missing:
            continue
        # interpolate within-window gaps into a clean continuous signal (no zeros,
        # so the FFT-based seasonal layer sees no padding artifacts)
        sensor_values = (
            pd.DataFrame(sensor_values)
            .interpolate(method="linear", limit_direction="both")
            .bfill().ffill().to_numpy(dtype=np.float32)
        )
        sensor_values = np.nan_to_num(sensor_values, nan=0.0)
        win_start = start + wi * window_size
        time_feat = _clock_time_features(win_start, target_bins, bin_minutes)
        out.append((np.concatenate([sensor_values, time_feat], axis=1).astype(np.float32),
                    win_start))
    return out


def prepare_hrd_dataset(
    csv_path: str,
    window_hours: int = 168,
    bin_minutes: int = 15,
    label_col: str = "depression_status_endpoint",
    max_missing: float = 0.30,
    max_gap_minutes: int = 30,
    max_window_missing: float = 0.30,
    z_score: bool = True,
) -> Dict[str, object]:
    """Full HRD preprocessing -> windowed classification dataset for CoST.

    Returns a dict with:
      X               (N, T, C) float32 windows, C = n_sensors + NUM_TIME_FEATURES
      y               (N,)      int endpoint labels (0=control, 1=depressed)
      pids            (N,)      participant id per window
      window_ids      (N,)      unique id "pid_<isotime>" per window
      consistent_pids set       pids with baseline == endpoint
      labeled_pids    set       pids that carry an endpoint label
      sensor_cols, n_sensors, n_features
    """
    sensor_df, label_df = load_hrd(csv_path, max_missing=max_missing,
                                   max_gap_minutes=max_gap_minutes)
    label_by_pid, consistent_pids = _label_maps(label_df, label_col)
    sensor_cols = [c for c in SENSOR_COLS if c in sensor_df.columns]
    if not sensor_cols:
        raise ValueError("None of the expected sensor columns are present in the CSV.")
    window_minutes = window_hours * 60

    t0 = time.time()
    X_list, y_list, pid_list, wid_list = [], [], [], []
    for pid, g in sensor_df.groupby("pid", sort=True):
        label = label_by_pid.get(pid, 0)
        for arr, win_start in _window_participant(g, window_minutes, bin_minutes,
                                                  sensor_cols, max_window_missing, z_score):
            X_list.append(arr)
            y_list.append(label)
            pid_list.append(pid)
            wid_list.append(f"{pid}_{win_start.isoformat()}")

    if not X_list:
        raise RuntimeError("No windows produced; check window/bin/missing settings.")
    X = np.stack(X_list, axis=0).astype(np.float32)
    n_features = X.shape[-1]
    print(
        f"[data_preprocessing] built {len(X_list):,} windows of shape "
        f"(T={X.shape[1]}, C={n_features}) from {len(set(pid_list))} participants | "
        f"{int(np.sum(np.asarray(y_list) == 1))} depressed-endpoint windows | "
        f"{time.time() - t0:.1f}s"
    )
    return {
        "X": X,
        "y": np.asarray(y_list, dtype=int),
        "pids": np.asarray(pid_list),
        "window_ids": np.asarray(wid_list),
        "consistent_pids": consistent_pids,
        "labeled_pids": set(label_by_pid),
        "sensor_cols": sensor_cols,
        "n_sensors": len(sensor_cols),
        "n_features": n_features,
    }
