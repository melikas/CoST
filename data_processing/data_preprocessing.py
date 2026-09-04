"""HRD wearable-data preprocessing for the CoST project.

The raw file ``HRD_RAW_MinuteLevel.csv`` stores BOTH the minute-level sensor signals AND the
depression labels in one table (labels repeated on every row of a participant). This module
turns it into a windowed dataset that the CoST model (``cost.py``) can train on.

Two public entry points:

  * ``load_hrd(csv)``               -> (sensor_df, label_df, energy_by_pid_date): the cleaned
                                       tables plus the daily emotional-energy lookup.
  * ``prepare_hrd_dataset(csv)``    -> dict with the windowed tensor ``X`` (N, T, C), labels
                                       ``y``, participant ids ``pids``, and label metadata.
  * ``prepare_hrd_energy_sliding()`` -> the "mode B" sliding-window variant for the
                                       emotional-energy probe.

Pipeline, in order:

    CSV -> load_hrd()                 clean columns, drop bad participants, fill short gaps
        -> _window_participant()      cut into fixed windows, apply the quality gate, bin
        -> prepare_hrd_dataset()      stack into (N, T, C) and attach labels

Read the CONFIGURATION section below first: every threshold, column name and magic number
this module uses lives there, together with the reason it has the value it has.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]

# A (pid, calendar date) -> emotional energy (1-5) lookup.
DailyEnergyLookup = Dict[Tuple[str, date], float]

# One produced window: the (T, C) array and the wall-clock time its first bin starts at.
Window = Tuple[np.ndarray, pd.Timestamp]


# =============================================================================
# CONFIGURATION
# -----------------------------------------------------------------------------
# Everything tunable or hardcoded is declared here, grouped by concern:
#
#   CHANNELS  which CSV columns become which model channels
#   CLEANING  the quality thresholds that decide what data is usable
#   WINDOWING the shape of a window and how time is encoded into it
#
# The dataclasses are frozen so a stray assignment elsewhere cannot silently change the
# preprocessing of a run that has already been logged.
# =============================================================================


@dataclass(frozen=True)
class ChannelConfig:
    """Which raw CSV columns become which model channels, and how sleep is binarised.

    Channels are grouped by physical nature because each group needs different cleaning:
    wear-dependent signals get gap interpolation, event streams do not.
    """

    # --- the 4 model channels, by nature -------------------------------------------------
    heart_rate: List[str] = field(default_factory=lambda: ["HR"])
    # Step count only. The Fitbit active-minute intensity levels (fairly/lightly/very) and
    # sedentary_minutes are deliberately excluded.
    activity: List[str] = field(default_factory=lambda: ["Steps"])
    # Binary, DERIVED from the raw sleep_status column -- see derive_is_asleep().
    sleep: List[str] = field(default_factory=lambda: ["is_asleep"])
    # Event stream: a missing value means "no event happened", which is a real 0.
    event: List[str] = field(default_factory=lambda: ["screen"])

    # --- raw (snake_case) CSV column -> canonical internal name ---------------------------
    # Confining the rename to the CSV read means nothing downstream has to know the export
    # schema. Columns we exclude (floors, sedentary_minutes, call, *_active_minutes) are
    # simply never read.
    csv_rename: Dict[str, str] = field(default_factory=lambda: {
        "timestamp": "dateTime",        # raw string; parsed into a real `timestamp` on load
        "heart_rate": "HR",
        "steps_minutes": "Steps",
        "sleep_status": "sleep_level",  # categorical sleep stage, values unchanged
    })

    # --- sleep_status vocabulary ----------------------------------------------------------
    # "restless" means movement, so it counts as NOT asleep. Anything outside both sets is
    # unknown and gets resolved by whether the watch was worn (see derive_is_asleep).
    sleep_stage_values: Set[str] = field(default_factory=lambda: {"asleep", "light", "deep", "rem"})
    awake_values: Set[str] = field(default_factory=lambda: {"awake", "wake", "restless"})

    # --- participant-level label columns ---------------------------------------------------
    label: List[str] = field(default_factory=lambda: [
        "depression_status_baseline",
        "depression_status_endpoint",
    ])
    # Richer labels the re-export added. `depression_trajectory` holds the paper's Case-1
    # groups (Pre1_Post1 / Pre1_Post2 / Pre2_Post1 / Pre2_Post2) used for group-stratified
    # rhythm analysis; the CES-D scores are the raw severities behind the binary status.
    # Carried through to label_df but not required by the classification model.
    extra_label: List[str] = field(default_factory=lambda: [
        "depression_trajectory",
        "ces_d_baseline_score",
        "ces_d_endpoint_score",
    ])
    # Daily self-report. Kept OUT of the sensor channels: it is a downstream target, never
    # a model input.
    energy: str = "emotional_energy"

    @property
    def wear_dependent(self) -> List[str]:
        """Channels that go missing when the watch is off -> gap-interpolated, and the ones
        the participant-level missingness filter is computed over."""
        return self.heart_rate + self.activity + self.sleep

    @property
    def sensors(self) -> List[str]:
        """The full ordered set of model input channels (the C axis of every window)."""
        return self.heart_rate + self.activity + self.sleep + self.event

    @property
    def raw_numeric(self) -> List[str]:
        """Columns read as numbers straight from the CSV (post-rename). `is_asleep` is absent
        because it is derived, not read."""
        return self.heart_rate + self.activity + self.event


@dataclass(frozen=True)
class CleaningConfig:
    """Thresholds that decide which samples, participants and windows are usable."""

    # Heart rate outside this range is a sensor error, not physiology -> set to NaN.
    hr_valid_bpm: Tuple[float, float] = (20.0, 250.0)

    # Drop a participant whose wear-dependent channels are missing more than this fraction
    # of the time, averaged over channels.
    max_participant_missing: float = 0.30

    # Longest non-wear gap that gets linearly interpolated at load time. Data is sampled once
    # per minute, so this is both a minute count and a sample count. Longer gaps stay NaN and
    # are dealt with by the per-window gate below.
    max_gap_minutes: int = 30

    # Per-window, per-channel: a channel missing more than this fraction of its time bins
    # makes the whole window unusable. This is Algorithm 1, Case 1 of Yan et al. 2022
    # (ACM TIST 13(3), Article 47, Sec. 3.4.1 / 4.1.1).
    max_window_missing: float = 0.30

    # Longest run of missing bins tolerated at a window's START or END. Linear interpolation
    # cannot extrapolate, so edge gaps are nearest-filled; this caps how much may be
    # fabricated there. Set to the same 30 min the interior already allows, so the edge rule
    # and the interior rule agree.
    #
    # This replaced an earlier rule requiring the first AND last bin to be fully observed.
    # That rule discarded 14.0% of all candidate windows -- more than the 30% missing-bin
    # gate itself (10.1%) -- because one unlucky 15-min bin at either boundary killed an
    # entire 7-day window, and it fell harder on the depressed group (15.0% vs 13.0%).
    # Measured on HRD_RAW_MinuteLevel.csv, 166 participants.
    max_edge_gap_minutes: int = 30

    # A window with this many raw samples or fewer is too sparse to bin at all. Checked
    # before the missingness gate purely to skip obviously empty windows cheaply.
    min_raw_samples_per_window: int = 10


@dataclass(frozen=True)
class WindowingConfig:
    """Window geometry and the optional calendar channels appended to each window."""

    window_hours: int = 168                  # 7 days
    bin_minutes: int = 15
    label_col: str = "depression_status_endpoint"
    z_score: bool = True                     # per-participant; see zscore_within_participant

    # CoST calendar covariates, in the order salesforce/CoST's datautils.py builds them:
    # [minute, hour, dayofweek, day, dayofyear, month, weekofyear].
    clock_field_ranges: Tuple[Tuple[int, int], ...] = (
        (0, 59), (0, 23), (0, 6), (1, 31), (1, 366), (1, 12), (1, 53),
    )

    @property
    def n_time_features(self) -> int:
        return len(self.clock_field_ranges)

    @property
    def bins_per_window(self) -> int:
        return self.window_hours * 60 // self.bin_minutes


CHANNELS = ChannelConfig()
CLEANING = CleaningConfig()
WINDOWING = WindowingConfig()


# Fixed, DATA-INDEPENDENT standardisation for the calendar covariates. Every clock field has
# a known a-priori range, so an empirical mean/std estimated from the windows buys nothing --
# and estimating it caused two real problems:
#   1. it was fitted on the POOLED windows, i.e. including held-out test participants (a
#      transductive scaler; measured effect small, <= 0.03 sigma, but it is still test data
#      influencing a transform applied to training data);
#   2. worse, the two entry points fitted it on DIFFERENT window sets -- prepare_hrd_dataset
#      over non-overlapping 168h windows, prepare_hrd_energy_sliding over trailing daily
#      windows -- so a frozen encoder pretrained through the first path was fed clock
#      channels on a different scale by the second (measured up to 0.16 sigma).
# A fixed transform removes both by construction. The values are those of a discrete uniform
# on each field's range: mean = (lo + hi) / 2, std = sqrt(((hi - lo + 1)^2 - 1) / 12).
CLOCK_MU = np.array(
    [(lo + hi) / 2.0 for lo, hi in WINDOWING.clock_field_ranges], dtype=np.float32)
CLOCK_SD = np.array(
    [np.sqrt(((hi - lo + 1) ** 2 - 1) / 12.0) for lo, hi in WINDOWING.clock_field_ranges],
    dtype=np.float32)


# --- Backwards-compatible module-level aliases ---------------------------------------------
# Other modules and the project docs refer to these names. They are views onto the config
# above, so there is still exactly one place to change a value.
HR_COLS = CHANNELS.heart_rate
ACTIVITY_COLS = CHANNELS.activity
SLEEP_COLS = CHANNELS.sleep
EVENT_COLS = CHANNELS.event
WEAR_COLS = CHANNELS.wear_dependent
SENSOR_COLS = CHANNELS.sensors
RAW_NUMERIC_COLS = CHANNELS.raw_numeric
CSV_RENAME = CHANNELS.csv_rename
SLEEP_ASLEEP = CHANNELS.sleep_stage_values
SLEEP_AWAKE = CHANNELS.awake_values
LABEL_COLS = CHANNELS.label
EXTRA_LABEL_COLS = CHANNELS.extra_label
HR_VALID_RANGE = CLEANING.hr_valid_bpm
EDGE_GAP_MINUTES = CLEANING.max_edge_gap_minutes
NUM_TIME_FEATURES = WINDOWING.n_time_features
CLOCK_RANGES = WINDOWING.clock_field_ranges

# Temporary grouping columns added to a working frame during binning.
_WINDOW_COL = "_w"
_BIN_COL = "_b"


# =============================================================================
# 1. CSV -> CLEANED TABLES
# =============================================================================

_HRD_CACHE: Dict[tuple, tuple] = {}          # at most one entry; see load_hrd(cache=True)


def clear_hrd_cache() -> None:
    """Drop the cached raw tables (~1 GiB). Call once the last consumer has run."""
    _HRD_CACHE.clear()


def _read_raw_csv(csv_path: PathLike) -> pd.DataFrame:
    """Read only the columns we consume, rename them to canonical names, and parse types.

    Rows without a parseable timestamp are dropped -- they cannot be placed on a time grid.
    """
    raw_columns = (["pid"]
                   + list(CHANNELS.csv_rename)
                   + CHANNELS.event
                   + CHANNELS.label
                   + CHANNELS.extra_label
                   + [CHANNELS.energy])
    df = pd.read_csv(csv_path, usecols=raw_columns, low_memory=False)
    df = df.rename(columns=CHANNELS.csv_rename)

    df["pid"] = df["pid"].astype(str).str.lower().str.strip()
    df["timestamp"] = pd.to_datetime(df["dateTime"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["pid", "timestamp"])

    for column in CHANNELS.raw_numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(np.float32)
    df[CHANNELS.energy] = pd.to_numeric(df[CHANNELS.energy], errors="coerce")
    return df


def derive_is_asleep(df: pd.DataFrame) -> np.ndarray:
    """Map the categorical sleep_level to a binary is_asleep channel.

    Must run AFTER heart rate has been range-cleaned, because HR is what disambiguates a
    missing sleep score: if HR is present the watch was worn but nothing was scored, which
    means awake (0); if HR is missing too we genuinely do not know, so the value stays NaN
    and the windowing gate decides. Forcing it to 0 would confound non-wear with wakefulness.
    """
    stage = df["sleep_level"].astype(str).str.lower().str.strip()
    is_asleep = np.where(
        stage.isin(CHANNELS.sleep_stage_values), 1.0,
        np.where(stage.isin(CHANNELS.awake_values), 0.0, np.nan),
    ).astype(np.float32)

    unknown = np.isnan(is_asleep)
    watch_was_worn = df["HR"].notna().to_numpy()
    is_asleep[unknown & watch_was_worn] = 0.0
    return is_asleep


def _clean_sensor_values(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the per-channel cleaning each channel's nature calls for.

    HR is range-checked, counts are clipped non-negative, sleep is binarised, and event
    channels get their structural absence turned into a real 0.
    """
    lo, hi = CLEANING.hr_valid_bpm
    for column in CHANNELS.heart_rate:
        df[column] = df[column].where((df[column] >= lo) & (df[column] <= hi))
    df[CHANNELS.activity] = df[CHANNELS.activity].clip(lower=0)

    df["is_asleep"] = derive_is_asleep(df)

    if CHANNELS.event:
        df[CHANNELS.event] = df[CHANNELS.event].fillna(0.0)
    return df


def _build_label_table(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the repeated per-row labels into one row per participant."""
    def first_valid(series: pd.Series) -> Any:
        """First non-null value; labels repeat, but some rows are blank."""
        present = series.dropna()
        return present.iloc[0] if len(present) else np.nan

    label_columns = CHANNELS.label + CHANNELS.extra_label
    return df.groupby("pid")[label_columns].agg(first_valid).reset_index()


def _participants_within_missing_budget(df: pd.DataFrame, max_missing: float) -> pd.Index:
    """Participant ids whose wear-dependent channels are complete enough to keep."""
    missing_per_participant = (df[CHANNELS.wear_dependent]
                               .isna()
                               .groupby(df["pid"])
                               .mean()
                               .mean(axis=1))
    return missing_per_participant.index[missing_per_participant <= max_missing]


def _interpolate_short_gaps(df: pd.DataFrame, max_gap_minutes: int) -> pd.DataFrame:
    """Linearly fill non-wear gaps up to `max_gap_minutes`, within each participant.

    Longer gaps stay NaN on purpose: they are real non-wear and the per-window gate should
    see them rather than have them fabricated away. `limit_area="inside"` keeps the fill
    strictly between observed samples, so nothing is extrapolated past a record's edges.
    """
    df[CHANNELS.wear_dependent] = df.groupby("pid")[CHANNELS.wear_dependent].transform(
        lambda s: s.interpolate(method="linear", limit=max_gap_minutes, limit_area="inside")
    )
    df[CHANNELS.activity] = df[CHANNELS.activity].clip(lower=0)
    return df


def _build_daily_energy_lookup(df: pd.DataFrame) -> DailyEnergyLookup:
    """Build the (pid, calendar date) -> emotional energy map used to label windows."""
    answered = df.dropna(subset=[CHANNELS.energy]).copy()
    answered["_date"] = answered["timestamp"].dt.normalize()
    daily = answered.groupby(["pid", "_date"])[CHANNELS.energy].first()
    return {(pid, day.date()): float(value) for (pid, day), value in daily.items()}


def load_hrd(
    csv_path: PathLike,
    max_missing: float = CLEANING.max_participant_missing,
    max_gap_minutes: int = CLEANING.max_gap_minutes,
    cache: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, DailyEnergyLookup]:
    """Read the raw CSV and return the cleaned (sensor_df, label_df, energy_by_pid_date).

    Parsing the 3.4 GB / 53.6 M-row CSV dominates start-up and both windowing paths need the
    same cleaned tables, so `cache=True` keeps the result for a second caller. Sharing is
    safe because no consumer mutates these tables (both windowers copy their group first).
    Costs ~1 GiB resident until clear_hrd_cache(), so only pass True when a second consumer
    actually follows.
    """
    cache_key = (str(csv_path), float(max_missing), int(max_gap_minutes))
    if cache_key in _HRD_CACHE:
        return _HRD_CACHE[cache_key]

    started_at = time.time()
    df = _read_raw_csv(csv_path)
    df = _clean_sensor_values(df)

    label_df = _build_label_table(df)

    # Quality filter, computed BEFORE interpolation so it sees genuinely missing data.
    n_participants_before = df["pid"].nunique()
    keep_pids = _participants_within_missing_budget(df, max_missing)
    df = df[df["pid"].isin(keep_pids)]
    label_df = label_df[label_df["pid"].isin(keep_pids)].reset_index(drop=True)

    df = _interpolate_short_gaps(df, max_gap_minutes)

    sensor_df = df[["pid", "timestamp"] + CHANNELS.sensors].reset_index(drop=True)
    # pid -> category AFTER the keep_pids filter, so there are no unused categories. 53.6 M
    # rows of Python strings cost 3.6 GiB as object dtype vs 1.0 GiB categorical, which is
    # what makes caching this frame affordable at --mem=32G.
    sensor_df["pid"] = sensor_df["pid"].astype("category")

    energy_by_pid_date = _build_daily_energy_lookup(df)

    n_consistent = int((label_df["depression_status_baseline"]
                        == label_df["depression_status_endpoint"]).sum())
    n_dropped = n_participants_before - len(keep_pids)
    print(
        f"[data_preprocessing] {len(sensor_df):,} rows | kept {len(keep_pids)}/"
        f"{n_participants_before} participants (dropped {n_dropped} with "
        f">{max_missing:.0%} wear-missing) | {n_consistent} with baseline==endpoint | "
        f"{time.time() - started_at:.1f}s"
    )

    tables = (sensor_df, label_df, energy_by_pid_date)
    if cache:
        _HRD_CACHE.clear()                   # keep at most one (~1 GiB) entry
        _HRD_CACHE[cache_key] = tables
    return tables


def _endpoint_labels_and_consistent_pids(
    label_df: pd.DataFrame, label_col: str
) -> Tuple[Dict[str, int], Set[str]]:
    """Return (label per participant, participants whose baseline == endpoint status).

    "Consistent" participants have a stable label, so they are the clean-label cohort;
    status-changers add label noise but more samples.
    """
    label_by_pid: Dict[str, int] = {}
    for _, row in label_df.iterrows():
        value = row.get(label_col)
        if pd.notna(value):
            label_by_pid[row["pid"]] = int(value)

    baseline, endpoint = "depression_status_baseline", "depression_status_endpoint"
    is_consistent = label_df[baseline] == label_df[endpoint]      # NaN == NaN is False
    consistent_pids = set(label_df.loc[is_consistent, "pid"])
    return label_by_pid, consistent_pids


def _trajectory_by_pid(label_df: pd.DataFrame) -> Dict[str, str]:
    """Participant -> Case-1 group (Pre1_Post1 ... Pre2_Post2), skipping those without one."""
    trajectories: Dict[str, str] = {}
    for pid, group in zip(label_df["pid"], label_df["depression_trajectory"]):
        if isinstance(group, str):
            trajectories[pid] = group
    return trajectories


def _baseline_status_by_pid(label_df: pd.DataFrame) -> Dict[str, int]:
    """Participant -> baseline depression status.

    Read straight from the status column rather than parsed out of `depression_trajectory`'s
    Pre1/Pre2 strings, so downstream code that selects on the (baseline, endpoint) pair does
    not depend on that naming convention.
    """
    baseline: Dict[str, int] = {}
    for _, row in label_df.iterrows():
        status = row["depression_status_baseline"]
        if pd.notna(status):
            baseline[row["pid"]] = int(status)
    return baseline


# =============================================================================
# 2. TIME CHANNELS
# =============================================================================

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


# =============================================================================
# 3. BINNING AND THE PER-WINDOW QUALITY GATE
#
# Shared by both windowing paths (non-overlapping and sliding) so the two can never drift
# apart on what counts as a usable window.
# =============================================================================

def zscore_within_participant(g: pd.DataFrame, sensor_cols: List[str]) -> pd.DataFrame:
    """Standardise one participant's sensor columns using only that participant's own stats.

    Leakage-free by construction: no statistic is shared across participants or with the
    labels. Zero-variance channels get sd = 1 so they pass through unchanged instead of
    producing NaN/inf.
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
        values[bin_index] = np.where(
            has_sample, mean_by_bin.loc[bin_index].to_numpy(dtype=np.float32), np.nan)
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


# =============================================================================
# 4. CLEANED TABLES -> NON-OVERLAPPING WINDOWS (depression path)
# =============================================================================

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
    n_complete_windows = int(float(minutes_since_start[-1]) // window_minutes)
    if n_complete_windows < 1:
        return None, 0

    window_index = (minutes_since_start // window_minutes).astype(np.int64)
    in_complete_window = window_index < n_complete_windows
    if not in_complete_window.any():
        return None, 0

    tagged = g.loc[in_complete_window, sensor_cols].copy()
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
    align_midnight: bool = False,
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


# =============================================================================
# 5. PUBLIC ENTRY POINT: DEPRESSION DATASET
# =============================================================================

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
    sensor_cols = [c for c in CHANNELS.sensors if c in sensor_df.columns]
    if not sensor_cols:
        raise ValueError("None of the expected sensor columns are present in the CSV.")
    return sensor_cols


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
    align_midnight: bool = False,
    cache_raw: bool = False,
    keep_observed: bool = False,
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
    sensor_df, label_df, energy_by_pid_date = load_hrd(
        csv_path, max_missing=max_missing, max_gap_minutes=max_gap_minutes, cache=cache_raw)

    label_by_pid, consistent_pids = _endpoint_labels_and_consistent_pids(label_df, label_col)
    sensor_cols = _resolve_sensor_cols(sensor_df)
    window_minutes = window_hours * 60
    window_span = pd.Timedelta(minutes=window_minutes)
    energy_by_pid = _group_energy_by_participant(energy_by_pid_date)

    started_at = time.time()
    all_windows: List[np.ndarray] = []
    all_labels: List[int] = []
    all_pids: List[str] = []
    # None unless asked for, so the windower's inner call is a no-op and X is unchanged.
    all_observed: Optional[List[np.ndarray]] = [] if keep_observed else None
    all_window_ids: List[str] = []
    all_window_energy: List[float] = []

    for pid, participant_rows in sensor_df.groupby("pid", sort=True, observed=True):
        label = label_by_pid.get(pid, 0)
        for window, window_start in _window_participant(
            participant_rows, window_minutes, bin_minutes, sensor_cols, max_window_missing,
            z_score, clock_features=clock_features, calendar_index=calendar_index,
            align_midnight=align_midnight, observed_out=all_observed,
        ):
            all_windows.append(window)
            all_labels.append(label)
            all_pids.append(pid)
            all_window_ids.append(f"{pid}_{window_start.isoformat()}")
            all_window_energy.append(_mean_energy_in_range(
                energy_by_pid.get(pid, []),
                start=pd.Timestamp(window_start),
                end=pd.Timestamp(window_start) + window_span,
                include_end=False,
            ))

    if not all_windows:
        raise RuntimeError("No windows produced; check window/bin/missing settings.")

    X = np.stack(all_windows, axis=0).astype(np.float32)
    if clock_features:
        standardise_clock_channels(X, len(sensor_cols))
    n_features = X.shape[-1]

    n_depressed = int(np.sum(np.asarray(all_labels) == 1))
    print(
        f"[data_preprocessing] built {len(all_windows):,} windows of shape "
        f"(T={X.shape[1]}, C={n_features}) from {len(set(all_pids))} participants | "
        f"{n_depressed} depressed-endpoint windows | {time.time() - started_at:.1f}s"
    )

    return {
        "X": X,
        "y": np.asarray(all_labels, dtype=int),
        "pids": np.asarray(all_pids),
        "window_ids": np.asarray(all_window_ids),
        "ee_win": np.asarray(all_window_energy, dtype=float),
        "consistent_pids": consistent_pids,
        "labeled_pids": set(label_by_pid),
        "trajectory_by_pid": _trajectory_by_pid(label_df),
        "baseline_by_pid": _baseline_status_by_pid(label_df),
        "sensor_cols": sensor_cols,
        "n_sensors": len(sensor_cols),
        "n_features": n_features,
        # (N, T, n_sensors) bool, aligned row for row with X. Absent unless keep_observed.
        **({"observed": np.stack(all_observed, axis=0)} if all_observed is not None else {}),
    }


# =============================================================================
# 6. CLEANED TABLES -> SLIDING WINDOWS (emotional-energy probe, "mode B")
# =============================================================================

def _bin_window(
    window_rows: pd.DataFrame,
    window_start: pd.Timestamp,
    target_bins: int,
    sensor_cols: List[str],
    bin_minutes: int,
    max_window_missing: float,
) -> Optional[np.ndarray]:
    """Bin the already-z-scored rows of ONE explicit window into (target_bins, n_sensors).

    Applies the same Algorithm-1 gate as the non-overlapping path and returns None when the
    window fails it. Kept separate from `_window_participant` so the depression path is
    untouched by changes here, but the gate and interpolation are shared.
    """
    minutes_into_window = (
        (window_rows["timestamp"].to_numpy() - np.datetime64(window_start))
        / np.timedelta64(1, "m")
    )
    tagged = window_rows[sensor_cols].copy()
    tagged[_BIN_COL] = np.clip(
        (minutes_into_window // bin_minutes).astype(np.int64), 0, target_bins - 1)

    grouped = tagged.groupby(_BIN_COL)
    return _build_window_array(
        grouped[sensor_cols].mean(), grouped[sensor_cols].count(),
        target_bins, len(sensor_cols), bin_minutes, max_window_missing)


def _sliding_windows_participant(
    g: pd.DataFrame,
    day_energy_pairs: Sequence[Tuple[Any, float]],
    window_minutes: int,
    bin_minutes: int,
    sensor_cols: List[str],
    max_window_missing: float,
    z_score: bool,
    clock_features: bool = False,
    calendar_index: bool = False,
) -> List[Tuple[np.ndarray, Any, float]]:
    """One TRAILING window per labelled day: day D gets the 7 days [D-6, D], labelled EE(D).

    Consecutive windows overlap by six days. That is fine because the train/val/test split is
    participant-level, so overlapping windows of one person never straddle the boundary --
    it is a statistical-independence caveat, not leakage.

    Returns [(window, day, energy), ...].
    """
    results: List[Tuple[np.ndarray, Any, float]] = []
    target_bins = window_minutes // bin_minutes

    if z_score:
        g = zscore_within_participant(g, sensor_cols)
    if len(g) < 2:
        return results

    timestamps = g["timestamp"].to_numpy()          # sensor_df is sorted by (pid, timestamp)
    window_span = pd.Timedelta(minutes=window_minutes)
    one_day = pd.Timedelta(days=1)

    for day, energy in day_energy_pairs:
        window_start = pd.Timestamp(day) - (window_span - one_day)   # (D-6) 00:00
        window_end = pd.Timestamp(day) + one_day                     # (D+1) 00:00
        lo = int(np.searchsorted(timestamps, np.datetime64(window_start)))
        hi = int(np.searchsorted(timestamps, np.datetime64(window_end)))
        if hi - lo <= CLEANING.min_raw_samples_per_window:
            continue

        sensor_values = _bin_window(g.iloc[lo:hi], window_start, target_bins, sensor_cols,
                                    bin_minutes, max_window_missing)
        if sensor_values is None:
            continue

        window = _append_time_channels(
            sensor_values, window_start, bin_minutes, clock_features, calendar_index)
        results.append((window.astype(np.float32), day, float(energy)))

    return results


def _sorted_energy_days_by_pid(
    energy_by_pid_date: DailyEnergyLookup,
) -> Dict[str, List[Tuple[date, float]]]:
    """pid -> [(date, energy), ...] sorted by date, for the trailing-window loop."""
    days_by_pid: Dict[str, List[Tuple[date, float]]] = {}
    for (pid, day), value in energy_by_pid_date.items():
        days_by_pid.setdefault(pid, []).append((day, value))
    for pid in days_by_pid:
        days_by_pid[pid].sort()
    return days_by_pid


def prepare_hrd_energy_sliding(
    csv_path: PathLike,
    window_hours: int = WINDOWING.window_hours,
    bin_minutes: int = WINDOWING.bin_minutes,
    max_missing: float = CLEANING.max_participant_missing,
    max_gap_minutes: int = CLEANING.max_gap_minutes,
    max_window_missing: float = CLEANING.max_window_missing,
    z_score: bool = WINDOWING.z_score,
    build_pretrain: bool = True,
    clock_features: bool = False,
    calendar_index: bool = False,
    energy_stride: int = 1,
) -> Dict[str, object]:
    """Mode-B dataset for the emotional-energy probe, from a SINGLE CSV read.

    Returns two windowings of the same cleaned data:
      * PROBE (sliding): one trailing 7-day window per labelled participant-day, labelled
        with that day's emotional energy. Keys `X` / `ee` / `pids` / `days`.
      * PRETRAIN (non-overlapping): the standard midnight-aligned windows over ALL
        participants, label-free. Keys `X_pretrain` / `pids_pretrain`, both None when
        `build_pretrain=False` -- which is the case when the caller reuses an already
        pretrained encoder (train_hrd.py --energy-probe) and only needs probe windows.

    The caller pretrains on `X_pretrain` minus the test participants, then encodes and probes
    `X`, so test participants stay out of pretraining exactly as in the non-sliding path.

    `energy_stride` keeps every k-th labelled day per participant. Consecutive windows share
    six of their seven days so the INPUTS are highly redundant, but the LABELS are not (each
    day's energy is a distinct target), which is why only small strides are useful. Applied
    here rather than after the fact, so discarded windows are never built at all -- both the
    ~0.5 GiB array and the windowing time scale with 1/k.
    """
    sensor_df, label_df, energy_by_pid_date = load_hrd(
        csv_path, max_missing=max_missing, max_gap_minutes=max_gap_minutes)

    sensor_cols = _resolve_sensor_cols(sensor_df)
    window_minutes = window_hours * 60
    days_by_pid = _sorted_energy_days_by_pid(energy_by_pid_date)

    started_at = time.time()
    probe_windows: List[np.ndarray] = []
    probe_pids: List[str] = []
    probe_days: List[Any] = []
    probe_energy: List[float] = []
    pretrain_windows: List[np.ndarray] = []
    pretrain_pids: List[str] = []

    for pid, participant_rows in sensor_df.groupby("pid", sort=True, observed=True):
        if build_pretrain:
            for window, _window_start in _window_participant(
                participant_rows, window_minutes, bin_minutes, sensor_cols,
                max_window_missing, z_score, clock_features=clock_features,
                calendar_index=calendar_index, align_midnight=True,
            ):
                pretrain_windows.append(window)
                pretrain_pids.append(pid)

        for window, day, energy in _sliding_windows_participant(
            participant_rows, days_by_pid.get(pid, [])[::energy_stride], window_minutes,
            bin_minutes, sensor_cols, max_window_missing, z_score,
            clock_features=clock_features, calendar_index=calendar_index,
        ):
            probe_windows.append(window)
            probe_pids.append(pid)
            probe_days.append(day)
            probe_energy.append(energy)

    if not probe_windows:
        raise RuntimeError("No sliding windows produced; check window/bin/missing settings.")

    X = np.stack(probe_windows).astype(np.float32)
    X_pretrain = np.stack(pretrain_windows).astype(np.float32) if build_pretrain else None
    if clock_features:
        # Same FIXED scale as prepare_hrd_dataset -- genuinely identical, not merely the same
        # formula over a different reference set. That is what makes these probe windows safe
        # to feed to an encoder pretrained through the other path.
        standardise_clock_channels(X, len(sensor_cols))
        standardise_clock_channels(X_pretrain, len(sensor_cols))

    print(
        f"[data_preprocessing] mode-B sliding: {len(probe_windows):,} trailing probe windows "
        f"(1 per labelled day) from {len(set(probe_pids))} participants | "
        f"{len(pretrain_windows):,} non-overlapping pretrain windows | "
        f"shape (T={X.shape[1]}, C={X.shape[-1]}) | {time.time() - started_at:.1f}s"
    )

    # Window-matched label: the mean energy over the labelled days the window actually spans.
    # `ee` above is EE(D) alone, so a 7-day input would be scored against a 1-day target.
    # Measured on HRD, that mismatch costs the task most of its signal: same-day daily
    # aggregates predict EE(D) at AUROC 0.56, the 7-day trailing mean of the SAME features at
    # 0.51, and the 7-day mean against a 7-day label back at 0.54. Computed from the FULL day
    # list, before `energy_stride` thins it, so the mean covers every labelled day in the
    # window regardless of stride.
    span_days = window_minutes // (60 * 24)
    window_energy: List[float] = []
    for pid, day in zip(probe_pids, probe_days):
        window_end = pd.Timestamp(day)
        window_energy.append(_mean_energy_in_range(
            days_by_pid.get(pid, []),
            start=window_end - pd.Timedelta(days=span_days - 1),
            end=window_end,
            include_end=True,
        ))

    return {
        "X": X,
        "ee": np.asarray(probe_energy, dtype=float),
        "pids": np.asarray(probe_pids),
        "ee_win": np.asarray(window_energy, dtype=float),
        "days": np.asarray(probe_days, dtype=object),
        "X_pretrain": X_pretrain,
        "pids_pretrain": np.asarray(pretrain_pids) if build_pretrain else None,
        "sensor_cols": sensor_cols,
        "n_sensors": len(sensor_cols),
        "n_features": X.shape[-1],
    }


def drop_sensor_channels(data, names):
    """Remove named sensor channels from an already-built dataset.

    Measured on HRD, 24 seeds, through an identical random projection and probe: dropping
    `Steps` raises the achievable AUC from 0.6884 to 0.7123. The channel is not merely
    uninformative for this endpoint, it is in the way.

    Applied after the windows are built rather than inside the loader, so the missingness
    filters and the z-scoring still see the channel and the surviving windows are exactly the
    ones every other run scored. Clock channels sit after the sensor channels and are kept.
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
    data["sensor_cols"] = [cols[i] for i in keep]
    data["n_sensors"] = len(keep)
    print(f"[data] dropped {names} -> {data['n_sensors']} sensor channels "
          f"{data['sensor_cols']}, X{data['X'].shape}")
    return data
