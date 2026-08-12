"""GLOBEM segment-level preprocessing for the CoST project.

The file ``GLOBEM_REDUCED.csv`` stores day-segment-level behavioural features (RAPIDS
aggregates from Fitbit + phone sensors) together with the per-participant depression
label (``LABEL_ENDPOINT``, repeated on every row). Each calendar day is split into
4 fixed segments (night, morning, afternoon, evening), so the sampling grid is 4
timesteps/day -- far coarser than the minute-level HRD data. This module turns the
table into windowed multivariate tensors that feed the SAME CoST pipeline used for HRD
(``train_hrd.py``): self-supervised pretraining, a linear probe fit on the label, and a
participant-level held-out test -- ALL on GLOBEM itself (no cross-dataset transfer).

Public entry point mirrors ``data_preprocessing.prepare_hrd_dataset`` (same return dict),
so ``train_hrd.py`` consumes it unchanged via ``--dataset globem``:

  * ``prepare_globem_dataset(csv)`` -> dict with X (N, T, C), labels y, pids, window_ids, ...

Design decisions (see the project discussion):
  * 12 feature channels are kept: Fitbit steps (3) + sleep (3), phone screen (2),
    location (4). The single bluetooth and wifi channels are excluded (see FEATURE_COLS).
  * Window = ``window_days`` days x 4 segments (default 28 -> T=112) so the CoST seasonal
    (FFT) layer has enough length to resolve daily (period 4) and weekly (period 28)
    rhythm; ``stride_days`` (default 7) slides the window one week at a time.
  * Missingness: the GLOBEM features are sparse (sleep ~33% present) and CoST's FFT layer
    cannot take NaN, while its encoder masks a timestep only when EVERY channel is NaN
    (models/encoder.py) -- a per-channel mask would not function. So each participant's
    channels are filled by per-participant LINEAR interpolation (interior) + nearest-value
    extension (ends) + zero for a fully-absent channel, AFTER computing the z-score
    statistics from the OBSERVED values only (leakage-free, no imputed value skews the
    scale). A window is dropped if fewer than ``min_window_coverage`` of its timesteps had
    any observation at all.
"""
from __future__ import annotations

import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils import pid_majority_label      # shared with train_hrd / hrd_rhythm

# -----------------------------------------------------------------------------
# 12 model channels (RAPIDS feature columns), grouped by source sensor.
# The bluetooth (f_blue) and wifi (f_wifi) channels are NOT used. Both are device-proximity
# counts rather than a behavioural rhythm of the participant, and each contributed a single
# channel; dropping them leaves the four multi-channel sensors that carry the daily cycle.
FEATURE_COLS = [
    "f_steps:fitbit_steps_intraday_rapids_sumsteps",
    "f_steps:fitbit_steps_intraday_rapids_avgdurationactivebout",
    "f_steps:fitbit_steps_intraday_rapids_countepisodesedentarybout",
    "f_slp:fitbit_sleep_intraday_rapids_sumdurationasleepunifiedmain",
    "f_slp:fitbit_sleep_intraday_rapids_sumdurationawakeunifiedmain",
    "f_slp:fitbit_sleep_intraday_rapids_ratiodurationasleepunifiedwithinmain",
    "f_screen:phone_screen_rapids_countepisodeunlock",
    "f_screen:phone_screen_rapids_sumdurationunlock",
    "f_loc:phone_locations_doryab_timeathome",
    "f_loc:phone_locations_doryab_locationentropy",
    "f_loc:phone_locations_doryab_totaldistance",
    "f_loc:phone_locations_doryab_numberlocationtransitions",
]

# Chronological within-day order so timestep t maps to a stable clock phase (seg = t % 4),
# matching the encoder's bins_per_day=4 circadian assumption.
SEG_ORDER = {"night": 0, "morning": 1, "afternoon": 2, "evening": 3}
SEG_PER_DAY = 4
NUM_TIME_FEATURES = 7          # CoST calendar covariates (only when clock_features=True)

# Fixed, DATA-INDEPENDENT standardisation for the calendar covariates built by
# _globem_time_features. Previously these were scaled by a mean/std fitted on the POOLED
# windows -- including the held-out test participants, a transductive transform applied to
# training data. Every field has a known a-priori range, so the empirical fit bought nothing.
#
# NOTE the ranges are NOT the ones in data_preprocessing.CLOCK_RANGES: this grid is 4
# segments/day, not 15-minute bins, so
#   * channel 0 ("minute") is the CONSTANT 0.0 -- an empirical std would be 0 (the old code
#     needed its `sd[sd == 0] = 1.0` guard for exactly this); it is a dead channel, so it is
#     centred at 0 with unit scale and passes through untouched;
#   * channel 1 ("hour") is seg*6, i.e. discrete uniform on {0, 6, 12, 18}, mean 9,
#     sd = 6*sqrt((4^2-1)/12) = sqrt(45) -- NOT hours 0..23.
# The remaining five are ordinary calendar fields: mean=(lo+hi)/2, sd=sqrt(((hi-lo+1)^2-1)/12).
_CAL_RANGES = ((0, 6), (1, 31), (1, 366), (1, 12), (1, 53))     # dow, day, doy, month, week
CLOCK_MU = np.array(
    [0.0, 9.0] + [(lo + hi) / 2.0 for lo, hi in _CAL_RANGES], dtype=np.float32)
CLOCK_SD = np.array(
    [1.0, float(np.sqrt(45.0))] + [np.sqrt(((hi - lo + 1) ** 2 - 1) / 12.0)
                                   for lo, hi in _CAL_RANGES], dtype=np.float32)


def standardise_clock_channels(X: np.ndarray, n_sensors: int) -> np.ndarray:
    """Scale the trailing NUM_TIME_FEATURES calendar channels of ``X`` (N, T, C) in place,
    with the fixed CLOCK_MU/CLOCK_SD above rather than statistics of ``X`` -- so the scaling
    can neither see held-out participants nor drift between runs. Sensor channels (the
    leading ``n_sensors``) are untouched; they are already per-participant z-scored."""
    if X is None or X.shape[-1] <= n_sensors:
        return X
    X[:, :, n_sensors:] = ((X[:, :, n_sensors:] - CLOCK_MU) / CLOCK_SD).astype(np.float32)
    return X

# Weekday every window is made to start on (0=Mon .. 6=Sun; -1 disables anchoring and reproduces
# the old behaviour of starting at each participant's first observed day). Monday is both the
# conventional week start and the modal enrolment weekday in GLOBEM. See _window_participant for
# why this matters: without it the Fourier phase origin differs per participant.
ANCHOR_WEEKDAY = 0


def _to_binary_label(v) -> int:
    """LABEL_ENDPOINT (bool / 'True' / 'False') -> 1 (depressed) / 0 (control)."""
    return 1 if str(v).strip().lower() in ("true", "1", "1.0") else 0


def _globem_time_features(dates: pd.DatetimeIndex, start: int, n_steps: int) -> np.ndarray:
    """CoST-style calendar covariates per timestep, analogous to
    data_preprocessing._cost_time_features but on the 4-segment/day grid: the segment
    stands in for the intra-day clock (hour = seg * 6). Returns (n_steps, NUM_TIME_FEATURES)."""
    feats = np.empty((n_steps, NUM_TIME_FEATURES), dtype=np.float32)
    for i in range(n_steps):
        t = start + i
        seg = t % SEG_PER_DAY
        d = dates[t // SEG_PER_DAY]
        feats[i] = (
            0.0, seg * 6.0, d.dayofweek, d.day, d.dayofyear, d.month,
            d.isocalendar()[1],
        )
    return feats


# =============================================================================
# 1. CSV -> cleaned long table
# =============================================================================

def load_globem(csv_path: str, label_col: str = "LABEL_ENDPOINT",
                weekly_col: str = "LABEL_WEEKLY"
                ) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Read GLOBEM_REDUCED.csv -> (long sensor table sorted by pid/date/segment,
    label_by_pid). Rows with an unknown segment or unparseable date are dropped.

    ``label_by_pid`` is the ONE-per-participant ``label_col`` (endpoint) map. When present, the
    time-varying ``weekly_col`` (LABEL_WEEKLY) is parsed into a per-row int column ``_weekly``
    (0/1, NaN where no weekly survey that date) so prepare_globem_dataset can label each window
    by the weekly survey inside its date span."""
    t0 = time.time()
    header = pd.read_csv(csv_path, nrows=0).columns
    has_weekly = weekly_col in header
    usecols = ["pid", "date", "segment", label_col] + FEATURE_COLS
    if has_weekly:
        usecols = usecols + [weekly_col]
    df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)

    df["pid"] = df["pid"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["seg_order"] = df["segment"].astype(str).str.lower().str.strip().map(SEG_ORDER)
    df = df.dropna(subset=["date", "seg_order"])
    df["seg_order"] = df["seg_order"].astype(int)
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    # per-row weekly label -> 0/1 (NaN where the weekly survey is absent that date)
    df["_weekly"] = (df[weekly_col].map(lambda v: np.nan if pd.isna(v) else _to_binary_label(v))
                     if has_weekly else np.nan)
    df = df.sort_values(["pid", "date", "seg_order"]).reset_index(drop=True)

    # One endpoint label per participant (constant within a pid; some rows are blank).
    lab = df.groupby("pid")[label_col].agg(lambda s: s.dropna().iloc[0] if s.notna().any()
                                           else np.nan)
    label_by_pid = {p: _to_binary_label(v) for p, v in lab.items() if pd.notna(v)}
    n_wk = int(df["_weekly"].notna().sum()) if has_weekly else 0
    print(f"[globem_preprocessing] {len(df):,} rows | {df['pid'].nunique()} participants | "
          f"{len(label_by_pid)} endpoint-labeled ({sum(label_by_pid.values())} depressed) | "
          f"weekly-label rows={n_wk:,} | {time.time() - t0:.1f}s")
    return df, label_by_pid


# =============================================================================
# 2. per-participant continuous grid -> filled, z-scored array + observation mask
# =============================================================================

def _prep_participant(g: pd.DataFrame, z_score: bool
                      ) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """One participant -> (filled (T_total, C) NaN-free array, observed (T_total,) bool,
    the continuous day index). The grid is reindexed to every (calendar-day x segment) from
    the first to the last day, so gaps become NaN timesteps; z-score stats come from the
    OBSERVED values only, then interior gaps are linearly interpolated and the ends carried."""
    dates = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
    idx = pd.MultiIndex.from_product([dates, range(SEG_PER_DAY)], names=["date", "seg_order"])
    grid = g.set_index(["date", "seg_order"])[FEATURE_COLS].reindex(idx)
    arr = grid.to_numpy(dtype=np.float32)                        # (n_days*4, C), date-major

    observed = ~np.isnan(arr).all(axis=1)                        # timestep with >=1 feature
    with warnings.catch_warnings():                              # all-NaN channels -> NaN mu/sd
        warnings.simplefilter("ignore", RuntimeWarning)          # (empty slice); replaced below
        mu = np.nanmean(arr, axis=0); sd = np.nanstd(arr, axis=0)  # observed-only statistics
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)

    filled = pd.DataFrame(arr).interpolate(limit_direction="both").to_numpy(dtype=np.float32)
    filled = np.nan_to_num(filled, nan=0.0)                      # fully-absent channel -> 0
    if z_score:
        filled = ((filled - mu) / sd).astype(np.float32)
    return filled, observed, dates


def _window_participant(filled: np.ndarray, observed: np.ndarray, dates: pd.DatetimeIndex,
                        window_steps: int, stride_steps: int, min_window_coverage: float,
                        clock_features: bool, anchor_weekday: int = ANCHOR_WEEKDAY
                        ) -> List[Tuple[np.ndarray, pd.Timestamp]]:
    """Slide fixed windows over the participant's timeline; keep those with enough real
    coverage. Returns [(window (window_steps, C[+time]), window-start day), ...].

    `anchor_weekday` (0=Mon .. 6=Sun, or -1 to disable) fixes the WEEKDAY every window starts
    on. Without it the grid starts at the participant's first observed day, so the phase origin
    of the window differs per participant according to their enrolment date -- on GLOBEM that is
    three different weekdays. Intra-day alignment was never the problem (segments are anchored to
    clock time, so the 24 h bin is comparable for everyone); the weekly bin and every bin that is
    not a multiple of the daily period are the ones that inherit an arbitrary per-participant
    phase offset, which makes their absolute phase incomparable across people.

    Costs at most 6 days at the start of each participant's record. Only delivers a CONSTANT
    weekday for every window when the stride is a whole number of weeks -- `prepare_globem_dataset`
    warns otherwise."""
    start_step = 0
    if anchor_weekday is not None and anchor_weekday >= 0:
        offs = [i for i in range(min(7, len(dates))) if dates[i].dayofweek == anchor_weekday]
        if not offs:
            return []                                  # <7 days of record: no anchored window
        start_step = offs[0] * SEG_PER_DAY
    out: List[Tuple[np.ndarray, pd.Timestamp]] = []
    for s in range(start_step, filled.shape[0] - window_steps + 1, stride_steps):
        e = s + window_steps
        if observed[s:e].mean() < min_window_coverage:
            continue
        arr = filled[s:e]
        if clock_features:
            arr = np.concatenate([arr, _globem_time_features(dates, s, window_steps)], axis=1)
        out.append((arr.astype(np.float32), dates[s // SEG_PER_DAY]))
    return out


# =============================================================================
# 3. full pipeline -> windowed classification dataset
# =============================================================================

def _window_weekly_label(wk_map: Dict[pd.Timestamp, int], win_start: pd.Timestamp,
                         window_days: int) -> int:
    """Weekly label for one window = majority of the LABEL_WEEKLY surveys whose date falls in
    the window's span [win_start, win_start + window_days). Returns 0/1, or -1 when no weekly
    survey lies inside the window (that window is unlabeled -> pretraining only)."""
    hi = win_start + pd.Timedelta(days=window_days)
    vals = [v for d, v in wk_map.items() if win_start <= d < hi]
    if not vals:
        return -1
    # Shared rule (utils.pid_majority_label) rather than a local `int(round(...))`: this site
    # and the participant-level collapse must agree on tied windows. The comment here used to
    # claim "ties -> 1" while the code actually returned 0 (round is half-to-even) -- the
    # shared helper makes the policy explicit and true in one place. Ties resolve to 0.
    return pid_majority_label(vals)


def prepare_globem_dataset(
    csv_path: str,
    window_days: int = 28,
    stride_days: int = 7,
    label_col: str = "LABEL_ENDPOINT",
    z_score: bool = True,
    clock_features: bool = False,
    min_window_coverage: float = 0.5,
    weekly_labels: bool = False,
    anchor_weekday: int = ANCHOR_WEEKDAY,
) -> Dict[str, object]:
    """GLOBEM -> windowed dataset for CoST, with the SAME dict keys as
    data_preprocessing.prepare_hrd_dataset so train_hrd.py consumes it unchanged.

    X is (N, T, C) with T = window_days * 4 and C = len(FEATURE_COLS) (+ NUM_TIME_FEATURES
    when clock_features).

    Labelling (``weekly_labels``):
      * False -> ONE endpoint label per participant, repeated on all that pid's windows.
      * True  -> each window gets the time-varying WEEKLY label (LABEL_WEEKLY) of the survey
        inside its date span; windows with NO weekly survey get y=-1 (UNLABELED -> used for
        self-supervised pretraining only, never in the probe/test). The per-participant split
        still needs one label per pid, so ``pid_summary_label`` = the majority of that pid's
        labelled weekly windows (used ONLY to stratify/balance the participant-level split).

    ``consistent_pids`` == ``labeled_pids`` (pids with >=1 usable label); the trajectory /
    baseline maps stay empty (kept only for interface compatibility)."""
    df, label_by_pid = load_globem(csv_path, label_col=label_col)
    window_steps = window_days * SEG_PER_DAY
    stride_steps = max(1, stride_days * SEG_PER_DAY)
    if anchor_weekday is not None and anchor_weekday >= 0 and stride_days % 7 != 0:
        print(f"[globem_preprocessing] WARNING: anchor_weekday is set but stride_days="
              f"{stride_days} is not a whole number of weeks, so only the FIRST window of each "
              f"participant starts on the anchor weekday and later ones drift. Use a stride that "
              f"is a multiple of 7, or pass anchor_weekday=-1.")

    t0 = time.time()
    X_list, y_list, pid_list, wid_list = [], [], [], []
    for pid, g in df.groupby("pid", sort=True):
        filled, observed, dates = _prep_participant(g, z_score)
        if filled.shape[0] < window_steps:
            continue
        # per-participant weekly survey map (date -> 0/1); endpoint label as the fallback mode
        wk_map = (g.dropna(subset=["_weekly"]).groupby("date")["_weekly"].first().astype(int).to_dict()
                  if weekly_labels else {})
        endpoint_label = label_by_pid.get(pid, 0)
        for arr, win_start in _window_participant(filled, observed, dates, window_steps,
                                                  stride_steps, min_window_coverage,
                                                  clock_features, anchor_weekday):
            label = (_window_weekly_label(wk_map, win_start, window_days)
                     if weekly_labels else endpoint_label)
            X_list.append(arr)
            y_list.append(label)
            pid_list.append(pid)
            wid_list.append(f"{pid}_{win_start.date().isoformat()}")

    if not X_list:
        raise RuntimeError("No windows produced; check window/stride/coverage settings.")
    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=int)
    pids_arr = np.asarray(pid_list)
    n_sensors = len(FEATURE_COLS)
    # Report the realised phase origin. Every window must start on the same weekday for the
    # absolute Fourier phase of the non-daily bins to mean the same thing across participants.
    _wd = pd.to_datetime([w.rsplit("_", 1)[1] for w in wid_list]).dayofweek
    _names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    _dist = {_names[k]: int(v) for k, v in zip(*np.unique(_wd, return_counts=True))}
    print(f"[globem_preprocessing] window start weekday: {_dist}"
          + ("  -> single origin, phases are comparable across participants"
             if len(_dist) == 1 else
             "  -> MULTIPLE origins: the phase of every non-daily frequency bin carries an "
             "arbitrary per-participant offset"))
    if clock_features:
        # FIXED, data-independent scaling -- never statistics of X, which would be fitted on
        # the pooled windows including held-out test participants. See CLOCK_RANGES above.
        standardise_clock_channels(X, n_sensors)
    n_features = X.shape[-1]

    if weekly_labels:
        labeled_w = y >= 0                                       # windows with a weekly label
        labeled_pids = set(pids_arr[labeled_w])
        # one summary label per pid (majority of its labelled weekly windows). Used to balance
        # the split AND -- via the shared rule -- as the ground truth the subject-level metrics
        # are scored against, so the two can no longer disagree.
        pid_summary = {p: pid_majority_label(y[(pids_arr == p) & labeled_w])
                       for p in labeled_pids}
        n_lab_w, n_dep_w = int(labeled_w.sum()), int(np.sum(y == 1))
        print(f"[globem_preprocessing] built {len(X_list):,} windows of shape "
              f"(T={X.shape[1]}, C={n_features}) from {len(set(pid_list))} participants | "
              f"weekly-labelled windows={n_lab_w:,} ({n_dep_w} depressed / "
              f"{n_lab_w - n_dep_w} non; {len(X_list) - n_lab_w} unlabelled->pretrain-only) | "
              f"{len(labeled_pids)} pids with >=1 weekly label | {time.time() - t0:.1f}s")
    else:
        labeled_pids = {p for p in set(pid_list) if p in label_by_pid}
        pid_summary = {p: label_by_pid[p] for p in labeled_pids}
        print(f"[globem_preprocessing] built {len(X_list):,} windows of shape "
              f"(T={X.shape[1]}, C={n_features}) from {len(set(pid_list))} participants | "
              f"{int(np.sum(y == 1))} depressed windows | {time.time() - t0:.1f}s")

    # GLOBEM is four annual cohorts (DS1..DS4 in the paper, 155/218/137/195 participants).
    # This release renumbered pids globally to INS-W_001..705, so cohort identity survives
    # only in the calendar year of a participant's first record -- which reproduces those
    # four counts exactly. Used by train_hrd's --holdout cross-dataset splits.
    pid_ds = {p: f"DS{yr - 2017}" for p, yr in
              df.groupby("pid")["date"].min().dt.year.items()}

    return {
        "X": X,
        "y": y,
        "pids": pids_arr,
        "pid_ds": pid_ds,
        "window_ids": np.asarray(wid_list),
        "consistent_pids": set(labeled_pids),
        "labeled_pids": set(labeled_pids),
        "pid_summary_label": pid_summary,      # one label/pid for split stratification only
        "weekly_labels": bool(weekly_labels),
        "trajectory_by_pid": {},
        "baseline_by_pid": {},
        "sensor_cols": list(FEATURE_COLS),
        "n_sensors": n_sensors,
        "n_features": n_features,
    }
