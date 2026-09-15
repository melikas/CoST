"""Cleaned tables -> fixed windows: optional time channels, per-participant z-scoring,
binning with the per-window quality gate, and the non-overlapping windows of one
participant (`_window_participant`).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from data_processing.hrd_config import CLEANING, CLOCK_MU, CLOCK_SD, Window, _BIN_COL, _WINDOW_COL


def standardise_clock_channels(X: Optional[np.ndarray], n_sensors: int) -> Optional[np.ndarray]:
    """Scale the trailing calendar channels of `X` (N, T, C) in place, using the fixed scale.

    Uses CLOCK_MU / CLOCK_SD rather than statistics of `X`, so every caller produces the SAME
    scaling -- which is what lets an encoder pretrained on one windowing be probed with
    another. Sensor channels (the leading `n_sensors`) are untouched: they are already
    per-participant z-scored at the windowing step.
    """
    if X is None or X.shape[-1] <= n_sensors:
        return X
    X[:, :, n_sensors:] = ((X[:, :, n_sensors:] - CLOCK_MU) / CLOCK_SD).astype(np.float32)
    return X


def _window_timestamps(window_start: pd.Timestamp, target_bins: int,
                       bin_minutes: int) -> pd.DatetimeIndex:
    """The wall-clock timestamp of every bin in a window."""
    return pd.date_range(window_start, periods=target_bins, freq=f"{bin_minutes}min")


def _cost_time_features(window_start: pd.Timestamp, target_bins: int,
                        bin_minutes: int) -> np.ndarray:
    """CoST calendar covariates per bin -> (target_bins, n_time_features).

    Exactly the fields datautils._get_time_features builds in salesforce/CoST. CoST scales
    them with a StandardScaler fitted on the data; we concatenate them the same way but use
    the fixed CLOCK_MU / CLOCK_SD instead (see standardise_clock_channels for why).
    """
    ts = _window_timestamps(window_start, target_bins, bin_minutes)
    return np.stack(
        [
            ts.minute.to_numpy(),
            ts.hour.to_numpy(),
            ts.dayofweek.to_numpy(),
            ts.day.to_numpy(),
            ts.dayofyear.to_numpy(),
            ts.month.to_numpy(),
            ts.isocalendar().week.to_numpy(),      # CoST's weekofyear (modern pandas API)
        ],
        axis=1,
    ).astype(np.float32)


def _calendar_index_features(window_start: pd.Timestamp, target_bins: int,
                             bin_minutes: int) -> np.ndarray:
    """Raw [time-of-day bin, day-of-week] index per bin -> (target_bins, 2).

    For the factorized calendar PE (``--pe factorized``). Deliberately NOT standardised:
    these are embedding lookups, not covariates.
    """
    ts = _window_timestamps(window_start, target_bins, bin_minutes)
    time_of_day_bin = (ts.hour.to_numpy() * 60 + ts.minute.to_numpy()) // bin_minutes
    return np.stack([time_of_day_bin, ts.dayofweek.to_numpy()], axis=1).astype(np.float32)


def _append_time_channels(sensor_values: np.ndarray, window_start: pd.Timestamp,
                          bin_minutes: int, clock_features: bool,
                          calendar_index: bool) -> np.ndarray:
    """Concatenate the optional time channels onto a window's sensor channels.

    The two encodings are mutually exclusive; with neither requested the window stays
    sensor-only, i.e. time is excluded from the model entirely.
    """
    target_bins = sensor_values.shape[0]
    if clock_features:
        extra = _cost_time_features(window_start, target_bins, bin_minutes)
    elif calendar_index:
        extra = _calendar_index_features(window_start, target_bins, bin_minutes)
    else:
        return sensor_values
    return np.concatenate([sensor_values, extra], axis=1)


def zscore_within_participant(g: pd.DataFrame, sensor_cols: List[str]) -> pd.DataFrame:
    """Standardise one participant's sensor columns using only that participant's own stats.

    This is retrospective full-record scaling, not historical-only scaling for prospective
    monitoring. Zero-variance channels receive unit scale.
    """
    g = g.copy()
    mean = g[sensor_cols].mean()
    std = g[sensor_cols].std().replace(0.0, 1.0).fillna(1.0)
    g[sensor_cols] = (g[sensor_cols] - mean) / std
    return g


def _fill_bin_grid(mean_by_bin: pd.DataFrame, count_by_bin: pd.DataFrame,
                   target_bins: int, n_sensors: int) -> Tuple[np.ndarray, np.ndarray]:
    """Lay per-bin means onto a fixed (target_bins, n_sensors) grid.

    Returns (values, observed): bins with no sample for a channel are NaN in `values` and
    False in `observed`, which is what the quality gate below reads.
    """
    values = np.full((target_bins, n_sensors), np.nan, dtype=np.float32)
    observed = np.zeros((target_bins, n_sensors), dtype=bool)
    for bin_index in mean_by_bin.index:
        has_sample = count_by_bin.loc[bin_index].to_numpy() > 0
        values[bin_index] = mean_by_bin.loc[bin_index].to_numpy(dtype=np.float32)
        observed[bin_index] = has_sample
    return values, observed


def _edge_gap_ok(observed: np.ndarray, bin_minutes: int) -> bool:
    """True when every channel's leading and trailing run of missing bins is short enough.

    Interpolation cannot extrapolate, so edge gaps are nearest-filled; this bounds how much
    of a window's start or end may be fabricated. `observed` is (n_bins, n_channels) boolean.
    """
    max_edge_bins = CLEANING.max_edge_gap_minutes // bin_minutes
    n_bins = observed.shape[0]
    for channel in range(observed.shape[1]):
        seen = np.flatnonzero(observed[:, channel])
        if len(seen) == 0:
            return False
        leading_gap, trailing_gap = seen[0], n_bins - 1 - seen[-1]
        if leading_gap > max_edge_bins or trailing_gap > max_edge_bins:
            return False
    return True


def _window_is_usable(observed: np.ndarray, bin_minutes: int,
                      max_window_missing: float) -> bool:
    """Apply Algorithm 1 (Yan et al. 2022, Sec. 3.4.1) to one window.

    A sensor feature is unusable when (a) more than `max_window_missing` of its time bins are
    missing, or (b) its leading/trailing gap is too long to nearest-fill. Because the fixed-C
    tensor needs the full channel set, ANY unusable feature disqualifies the whole window --
    the tensor analogue of the paper's per-feature drop.
    """
    missing_fraction = 1.0 - observed.mean(axis=0)          # per channel
    if (missing_fraction > max_window_missing).any():
        return False
    return _edge_gap_ok(observed, bin_minutes)


def _interpolate_interior_gaps(values: np.ndarray) -> np.ndarray:
    """Fill the remaining gaps of a window that already passed the quality gate.

    Interior gaps are linearly interpolated; edge gaps are nearest-filled and were bounded by
    the gate, so nothing is extrapolated. The result is a clean continuous signal with no
    zero-padding, which matters because the FFT-based seasonal layer would read padding as
    real signal.
    """
    filled = (pd.DataFrame(values)
              .interpolate(method="linear", limit_direction="both")
              .to_numpy(dtype=np.float32))
    return np.nan_to_num(filled, nan=0.0)


def _build_window_array(mean_by_bin: pd.DataFrame, count_by_bin: pd.DataFrame,
                        target_bins: int, n_sensors: int, bin_minutes: int,
                        max_window_missing: float,
                        observed_out: Optional[List[np.ndarray]] = None,
                        ) -> Optional[np.ndarray]:
    """Bin -> gate -> interpolate for one window. None when the window fails the gate.

    `observed_out`, when given, receives this window's observation mask: True where a raw
    sample landed in that bin for that channel, False where the value returned below was
    manufactured by interpolation. It is appended only for windows that PASS the gate, in
    the same order the windows themselves are, so the two stay aligned one to one. The
    default None leaves every existing caller returning exactly what it returned before.

    Worth keeping because the gate admits up to `max_window_missing` (0.30) of a channel's
    bins, fills them by linear interpolation, and then throws away the record of which bins
    those were. The encoder is handed the result with no way to tell a measured value from a
    manufactured one -- and linear interpolation manufactures precisely the smooth
    low-frequency shape the trend branch and the Fourier layer are built to fit.
    """
    values, observed = _fill_bin_grid(mean_by_bin, count_by_bin, target_bins, n_sensors)
    if not _window_is_usable(observed, bin_minutes, max_window_missing):
        return None
    if observed_out is not None:
        observed_out.append(observed)
    return _interpolate_interior_gaps(values)


def _tag_samples_with_window_and_bin(
    g: pd.DataFrame, grid_start: pd.Timestamp, window_minutes: int, bin_minutes: int,
    sensor_cols: List[str], target_bins: int,
) -> Tuple[Optional[pd.DataFrame], int]:
    """Label each raw sample with the window and bin it falls into.

    Returns (tagged_frame, n_complete_windows). Samples in the trailing partial window are
    dropped, since a fixed-size tensor cannot hold one. (None, 0) when there is not even one
    complete window.
    """
    minutes_since_start = (g["timestamp"] - grid_start).dt.total_seconds().to_numpy() / 60.0
    # A timestamp marks the start of a minute interval. Minute 10079 completes a week.
    n_complete_windows = int((float(minutes_since_start[-1]) + 1.0) // window_minutes)
    if n_complete_windows < 1:
        return None, 0

    window_index = (minutes_since_start // window_minutes).astype(np.int64)
    in_complete_window = window_index < n_complete_windows
    if not in_complete_window.any():
        return None, 0

    flags = [f"observed/{c}" for c in sensor_cols if f"observed/{c}" in g]
    tagged = g.loc[in_complete_window, sensor_cols + flags].copy()
    windows = window_index[in_complete_window]
    minutes_into_window = minutes_since_start[in_complete_window] - windows * window_minutes

    tagged[_WINDOW_COL] = windows
    tagged[_BIN_COL] = np.clip(
        (minutes_into_window // bin_minutes).astype(np.int64), 0, target_bins - 1)
    return tagged, n_complete_windows


def _window_participant(
    g: pd.DataFrame,
    window_minutes: int,
    bin_minutes: int,
    sensor_cols: List[str],
    max_window_missing: float,
    z_score: bool,
    clock_features: bool = False,
    calendar_index: bool = False,
    align_midnight: bool = True,
    observed_out: Optional[List[np.ndarray]] = None,
) -> List[Window]:
    """Cut one participant's record into consecutive fixed-size windows.

    Windows do not overlap. `align_midnight` anchors the grid to midnight so a given timestep
    always maps to the same clock time, which the circadian-phase pretext task needs;
    otherwise the grid starts at the participant's first sample.
    """
    windows: List[Window] = []
    target_bins = window_minutes // bin_minutes
    n_sensors = len(sensor_cols)

    if z_score:
        g = zscore_within_participant(g, sensor_cols)
    if len(g["timestamp"]) < 2:
        return windows

    first_sample = g["timestamp"].iloc[0]
    grid_start = first_sample.floor("D") if align_midnight else first_sample

    tagged, n_complete_windows = _tag_samples_with_window_and_bin(
        g, grid_start, window_minutes, bin_minutes, sensor_cols, target_bins)
    if tagged is None:
        return windows

    samples_per_window = tagged.groupby(_WINDOW_COL).size()
    grouped = tagged.groupby([_WINDOW_COL, _BIN_COL])
    mean_by_window_bin = grouped[sensor_cols].mean()
    count_by_window_bin = grouped[sensor_cols].count()
    flags = [f"observed/{c}" for c in sensor_cols]
    if all(c in tagged for c in flags):
        count_by_window_bin = grouped[flags].sum()
        count_by_window_bin.columns = sensor_cols
    windows_with_data = set(mean_by_window_bin.index.get_level_values(0))
    window_span = pd.Timedelta(minutes=window_minutes)

    for window_i in range(n_complete_windows):
        n_samples = int(samples_per_window.get(window_i, 0))
        if n_samples <= CLEANING.min_raw_samples_per_window:
            continue

        if window_i in windows_with_data:
            mean_by_bin = mean_by_window_bin.loc[window_i]
            count_by_bin = count_by_window_bin.loc[window_i]
        else:
            # No sample landed in any bin: build an empty grid so the gate rejects it.
            mean_by_bin = count_by_bin = pd.DataFrame(columns=sensor_cols)

        sensor_values = _build_window_array(
            mean_by_bin, count_by_bin, target_bins, n_sensors, bin_minutes,
            max_window_missing, observed_out=observed_out)
        if sensor_values is None:
            continue

        window_start = grid_start + window_i * window_span
        window = _append_time_channels(
            sensor_values, window_start, bin_minutes, clock_features, calendar_index)
        windows.append((window.astype(np.float32), window_start))

    return windows
