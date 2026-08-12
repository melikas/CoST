"""HRD wearable-data preprocessing for the CoST project.

The raw file ``HRD_RAW_MinuteLevel.csv`` stores BOTH the minute-level sensor
signals AND the depression labels in one table (labels repeated on every row of
a participant). This module turns it into a windowed dataset that the CoST model
(``cost.py``) can train on for depression-endpoint classification.

The dataset was re-exported with snake_case column names (``timestamp``,
``heart_rate``, ``steps_minutes``, ``fairly_active_minutes``,
``lightly_active_minutes``, ``very_active_minutes``, ``sleep_status``, ``screen``,
plus new ``floors``, ``sedentary_minutes``, ``call``, ``depression_trajectory``
and ``ces_d_*_score`` columns). ``CSV_RENAME`` maps the raw columns we consume
back to the canonical internal names used everywhere else, so the rename is
confined to the CSV read and nothing downstream changes.

Two public entry points:

  * ``load_hrd(csv)``          -> (sensor_df, label_df, ee_by_pid_date), the cleaned
                                  tables plus the daily emotional_energy lookup.
  * ``prepare_hrd_dataset(csv)`` -> dict with the windowed tensor ``X`` (N, T, C),
                                  labels ``y``, participant ids ``pids`` and the
                                  set of consistent (baseline==endpoint) pids.

The model uses 4 sensor channels: HR + Steps + is_asleep (binary sleep, derived from
sleep_status) + screen. Floors, calls, the Sedentary intensity level and the Fitbit
active-minute intensity levels (Fairly/Lightly/Very_Active) are excluded. Each channel is
cleaned by its nature:
  * HR / Steps  - physiological / count; NaN = non-wear -> short gaps interpolated,
                  long gaps stay NaN (sparse windows dropped); counts clipped >= 0.
  * is_asleep   - sleep stages -> 1, awake/restless -> 0; NaN split by HR wear
                  (worn+unscored -> 0, non-wear -> NaN); short gaps interpolated.
  * screen      - event stream; NaN = no event -> 0.

Per-window missing values follow **Algorithm 1 of Yan et al. 2022 (ACM TIST 13(3),
Article 47), Case 1 (Sec. 3.4.1 / 4.1.1)**: within each participant-window, a sensor
feature with > ``max_window_missing`` (30%) missing time-bins, or with an edge gap longer
than ``EDGE_GAP_MINUTES`` (see :func:`_edge_gap_ok`), is unusable; remaining gaps
are filled by nearest-neighbour LINEAR interpolation. Because the fixed-C tensor needs
the full channel set, an unusable feature disqualifies the whole window (the tensor
analogue of the paper's per-feature drop). Per-participant z-scoring is applied at the
windowing step, so no statistics are shared across participants or with the labels
(leakage-free).
"""

import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Channel groups by physical nature.
HR_COLS = ["HR"]
ACTIVITY_COLS = ["Steps"]                 # step count only; the Fitbit active-minute intensity
                                          # levels (Fairly/Lightly/Very_Active) and Sedentary are dropped
SLEEP_COLS = ["is_asleep"]                # binary, DERIVED from sleep_level (_derive_is_asleep)
EVENT_COLS = ["screen"]                   # event stream; NaN = no event -> 0 (absence is a real 0)

WEAR_COLS = HR_COLS + ACTIVITY_COLS + SLEEP_COLS  # wear-dependent -> gap-interp + missingness
SENSOR_COLS = HR_COLS + ACTIVITY_COLS + SLEEP_COLS + EVENT_COLS  # 4 model channels (Floors, calls,
                                          # Sedentary + *_Active intensity dropped; sleep -> is_asleep)
RAW_NUMERIC_COLS = HR_COLS + ACTIVITY_COLS + EVENT_COLS   # numeric columns (canonical names, post-rename)

# Raw (snake_case) CSV column -> canonical internal name. The raw ``timestamp`` is renamed to
# ``dateTime`` so the load body can parse it into the datetime ``timestamp`` exactly as before;
# ``screen`` and the label columns already carry their expected names. The excluded columns
# (``floors``, ``sedentary_minutes``, ``call`` and the *_active_minutes intensity levels
# are simply never read.)
CSV_RENAME = {
    "timestamp": "dateTime",                  # raw timestamp string -> parsed into `timestamp`
    "heart_rate": "HR",
    "steps_minutes": "Steps",
    "sleep_status": "sleep_level",            # categorical sleep stage (values unchanged)
}

# sleep_status -> binary is_asleep. restless = movement -> NOT asleep (0), per the experiment.
SLEEP_ASLEEP = {"asleep", "light", "deep", "rem"}     # sleep stages -> 1
SLEEP_AWAKE = {"awake", "wake", "restless"}           # awake / not-asleep -> 0
                                                      # (unknown / NaN -> resolved by HR wear)
LABEL_COLS = ["depression_status_baseline", "depression_status_endpoint"]
# Richer per-participant labels the re-export added. depression_trajectory holds the paper's
# Case-1 groups (Pre1_Post1 / Pre1_Post2 / Pre2_Post1 / Pre2_Post2) for group-stratified rhythm
# analysis; the CES-D scores are the raw depression severities behind the binary status. Carried
# through to label_df / the output dict but not required by the classification model.
EXTRA_LABEL_COLS = ["depression_trajectory", "ces_d_baseline_score", "ces_d_endpoint_score"]

HR_VALID_RANGE = (20.0, 250.0)            # bpm outside this is a sensor error -> NaN
NUM_TIME_FEATURES = 7                     # CoST calendar covariates appended to every window

# Fixed, DATA-INDEPENDENT standardisation for the calendar covariates produced by
# _cost_time_features -> [minute, hour, dayofweek, day, dayofyear, month, weekofyear].
# Every one of these fields has a known a-priori range, so an empirical mean/std estimated
# from the windows buys nothing -- and estimating it caused two real problems:
#   1. it was fitted on the POOLED windows, i.e. including held-out test participants
#      (a transductive scaler; measured effect was small, <= 0.03 sigma, but it is still
#      test data influencing a transform applied to training data);
#   2. worse, the two entry points fitted it on DIFFERENT window sets -- prepare_hrd_dataset
#      over non-overlapping 168h windows, prepare_hrd_energy_sliding over trailing daily
#      windows -- so a frozen encoder pretrained through the first path was fed clock
#      channels on a different scale by the second (measured up to 0.16 sigma).
# A fixed transform removes both by construction: it cannot see test data, and it is
# identical everywhere. Values are those of a discrete uniform on the field's range:
# mean = (lo+hi)/2, std = sqrt(((hi-lo+1)^2 - 1)/12).
CLOCK_RANGES = ((0, 59), (0, 23), (0, 6), (1, 31), (1, 366), (1, 12), (1, 53))
CLOCK_MU = np.array([(lo + hi) / 2.0 for lo, hi in CLOCK_RANGES], dtype=np.float32)
CLOCK_SD = np.array([np.sqrt(((hi - lo + 1) ** 2 - 1) / 12.0) for lo, hi in CLOCK_RANGES],
                    dtype=np.float32)


def standardise_clock_channels(X: np.ndarray, n_sensors: int) -> np.ndarray:
    """Scale the trailing NUM_TIME_FEATURES calendar channels of ``X`` (N, T, C) in place.

    Uses the fixed CLOCK_MU/CLOCK_SD above rather than statistics of ``X``, so every caller
    produces the SAME scaling -- which is what lets an encoder pretrained on one windowing
    be probed with another. Sensor channels (the leading ``n_sensors``) are untouched; they
    are already per-participant z-scored at the windowing step."""
    if X is None or X.shape[-1] <= n_sensors:
        return X
    X[:, :, n_sensors:] = ((X[:, :, n_sensors:] - CLOCK_MU) / CLOCK_SD).astype(np.float32)
    return X


def _first_valid(s: pd.Series):
    """First non-null value of a column (labels repeat, but some rows are blank)."""
    v = s.dropna()
    return v.iloc[0] if len(v) else np.nan


def _derive_is_asleep(df: pd.DataFrame) -> np.ndarray:
    """Map sleep_level -> binary is_asleep, disambiguating missingness by wear (needs a
    cleaned HR column). Sleep stages -> 1; awake/wake/restless -> 0. For unknown/NaN sleep:
    if HR is present the watch was worn but nothing was scored -> awake (0); if HR is also
    missing we genuinely don't know -> stays NaN (real non-wear, left for the windowing to
    drop -- never forced to 0, which would confound non-wear with wakefulness)."""
    s = df["sleep_level"].astype(str).str.lower().str.strip()
    a = np.where(s.isin(SLEEP_ASLEEP), 1.0,
                 np.where(s.isin(SLEEP_AWAKE), 0.0, np.nan)).astype(np.float32)
    unknown = np.isnan(a)                                   # unknown / NaN sleep_level
    a[unknown & df["HR"].notna().to_numpy()] = 0.0          # worn but unscored -> awake
    return a                                                # unknown & HR missing -> NaN


# =============================================================================
# 1. CSV -> cleaned tables
# =============================================================================

_HRD_CACHE: Dict[tuple, tuple] = {}          # at most one entry; see load_hrd(cache=True)


def clear_hrd_cache() -> None:
    """Drop the cached raw tables (~1 GiB). Call once the last consumer has run."""
    _HRD_CACHE.clear()


def load_hrd(csv_path: str, max_missing: float = 0.30, max_gap_minutes: int = 30,
             cache: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read HRD_RAW_MinuteLevel.csv and return (sensor_df, label_df, ee_by_pid_date).

    Parsing the 3.4 GB / 53.6 M-row CSV dominates start-up, and both windowing paths
    (prepare_hrd_dataset and prepare_hrd_energy_sliding) need the SAME cleaned tables.
    ``cache=True`` keeps the result so a second caller reuses it instead of re-parsing.
    Sharing is safe: no consumer mutates these tables (both windowers copy their group
    first). Costs ~1 GiB resident until clear_hrd_cache(), so only pass True when a
    second consumer actually follows.
    """
    key = (str(csv_path), float(max_missing), int(max_gap_minutes))
    if key in _HRD_CACHE:
        return _HRD_CACHE[key]
    t0 = time.time()
    # Read the raw (snake_case) columns we consume, then rename to canonical internal names.
    raw_cols = (["pid"] + list(CSV_RENAME) + ["screen"] + LABEL_COLS + EXTRA_LABEL_COLS
                + ["emotional_energy"])
    df = pd.read_csv(csv_path, usecols=raw_cols, low_memory=False).rename(columns=CSV_RENAME)

    # Identifier + timestamp, then drop rows with no parseable time.
    df["pid"] = df["pid"].astype(str).str.lower().str.strip()
    df["timestamp"] = pd.to_datetime(df["dateTime"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["pid", "timestamp"])
    for c in RAW_NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    df["emotional_energy"] = pd.to_numeric(df["emotional_energy"], errors="coerce")

    # Per-sensor range cleaning.
    lo, hi = HR_VALID_RANGE
    df["HR"] = df["HR"].where((df["HR"] >= lo) & (df["HR"] <= hi))      # bad HR -> NaN
    df[ACTIVITY_COLS] = df[ACTIVITY_COLS].clip(lower=0)                 # counts are >= 0

    # Sleep: binary is_asleep, derived AFTER HR cleaning so NaN can be split into
    # "worn but unscored -> awake" vs "non-wear -> stays NaN" (see _derive_is_asleep).
    df["is_asleep"] = _derive_is_asleep(df)

    # Structural absence (no event) is a real 0, not a missing value.
    if EVENT_COLS:
        df[EVENT_COLS] = df[EVENT_COLS].fillna(0.0)

    # One label row per participant (labels are constant within a pid).
    label_df = df.groupby("pid")[LABEL_COLS + EXTRA_LABEL_COLS].agg(_first_valid).reset_index()

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
    # pid -> category AFTER the keep_pids filter, so there are no unused categories.
    # 53.6 M rows of Python strings cost 3.6 GiB as object dtype vs 1.0 GiB categorical,
    # which is what makes caching this frame affordable at --mem=32G.
    sensor_df["pid"] = sensor_df["pid"].astype("category")

    # Daily self-reported emotional_energy (1-5), constant within a pid-day; keyed by
    # (pid, calendar date) for the per-window last-day lookup in prepare_hrd_dataset. Kept
    # OUT of SENSOR_COLS so it never enters the model as an input -- it is a downstream label.
    df_ee = df.dropna(subset=["emotional_energy"]).copy()
    df_ee["_date"] = df_ee["timestamp"].dt.normalize()
    ee_daily = df_ee.groupby(["pid", "_date"])["emotional_energy"].first()
    ee_by_pid_date = {(p, d.date()): float(v) for (p, d), v in ee_daily.items()}

    n_consistent = int(
        (label_df["depression_status_baseline"]
         == label_df["depression_status_endpoint"]).sum()
    )
    print(
        f"[data_preprocessing] {len(sensor_df):,} rows | kept {len(keep_pids)}/{n_total} "
        f"participants (dropped {n_total - len(keep_pids)} with >{max_missing:.0%} wear-missing) | "
        f"{n_consistent} with baseline==endpoint | {time.time() - t0:.1f}s"
    )
    out = (sensor_df, label_df, ee_by_pid_date)
    if cache:
        _HRD_CACHE.clear()                   # keep at most one (~1 GiB) entry
        _HRD_CACHE[key] = out
    return out


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

def _cost_time_features(win_start, target_bins: int, bin_minutes: int) -> np.ndarray:
    """CoST calendar covariates, EXACTLY as datautils._get_time_features in salesforce/CoST:
    the RAW [minute, hour, dayofweek, day, dayofyear, month, weekofyear] of each bin's timestamp
    -> (target_bins, NUM_TIME_FEATURES). CoST standardises them with a StandardScaler fitted on
    the data and concatenates them to the feature axis; we concatenate them the same way but
    scale them with the FIXED CLOCK_MU/CLOCK_SD instead (see standardise_clock_channels for
    why a fitted scaler was wrong here), appended as extra input channels."""
    ts = pd.date_range(win_start, periods=target_bins, freq=f"{bin_minutes}min")
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


def _calendar_index_features(win_start, target_bins: int, bin_minutes: int) -> np.ndarray:
    """Raw [time-of-day bin, day-of-week] index per bin -> (target_bins, 2), for the
    factorized calendar PE (``--pe factorized``). Deliberately NOT standardised: these are
    embedding lookups, not covariates."""
    ts = pd.date_range(win_start, periods=target_bins, freq=f"{bin_minutes}min")
    tod = (ts.hour.to_numpy() * 60 + ts.minute.to_numpy()) // bin_minutes
    return np.stack([tod, ts.dayofweek.to_numpy()], axis=1).astype(np.float32)


# Longest run of missing bins tolerated at a window's START or END. Linear interpolation
# cannot extrapolate, so edge gaps are nearest-filled by ``limit_direction="both"``; this
# caps how much may be fabricated there. Set to the SAME 30 min the interior already allows
# (load_hrd's max_gap_minutes), so the edge rule and the interior rule agree.
#
# This replaces the original endpoint rule, which required the first AND last bin to be
# fully observed. That rule discarded 14.0% of all candidate windows -- more than the 30%
# missing-bin quality gate itself (10.1%) -- because a single unlucky 15-min bin at either
# boundary killed an entire 7-day window, and it fell harder on the depressed group
# (15.0% vs 13.0%). Measured on HRD_RAW_MinuteLevel.csv, 166 participants.
EDGE_GAP_MINUTES = 30


def _edge_gap_ok(observed: np.ndarray, bin_minutes: int) -> bool:
    """True when every channel's leading and trailing run of missing bins is short enough
    to be nearest-filled. ``observed`` is a ``(n_bins, n_channels)`` boolean array."""
    max_edge = EDGE_GAP_MINUTES // bin_minutes
    n_bins = observed.shape[0]
    for c in range(observed.shape[1]):
        seen = np.flatnonzero(observed[:, c])
        if len(seen) == 0 or seen[0] > max_edge or (n_bins - 1 - seen[-1]) > max_edge:
            return False
    return True


def _window_participant(g: "pd.DataFrame", window_minutes: int, bin_minutes: int,
                        sensor_cols: List[str], max_window_missing: float,
                        z_score: bool, clock_features: bool = False,
                        calendar_index: bool = False,
                        align_midnight: bool = False
                        ) -> List[Tuple[np.ndarray, "pd.Timestamp"]]:
    """Vectorised fixed-size binning of one participant into windows.

    Each window is sensor-only by default; ``clock_features`` appends the 7 standardised
    CoST calendar covariates and ``calendar_index`` (``--pe factorized``) appends the 2 raw
    [time-of-day, day-of-week] index channels instead. The two are mutually exclusive."""
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
    # Anchor the grid to midnight so a timestep always maps to the same clock time (needed
    # by the circadian-phase pretext); otherwise start at the participant's first sample.
    start = ts.iloc[0].floor("D") if align_midnight else ts.iloc[0]
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
        # -- Algorithm 1 (Yan et al. 2022, Sec. 3.4.1) applied per sensor feature over the
        #    window's time bins. (a) A feature with more than max_window_missing (30%) missing
        #    bins is unusable. (b) Edge gaps are nearest-filled rather than interpolated, so a
        #    feature whose leading/trailing gap exceeds EDGE_GAP_MINUTES is unusable. Because
        #    the fixed-C tensor needs the full channel set, any unusable feature disqualifies
        #    the whole window (tensor analogue of the per-feature drop).
        miss_frac = 1.0 - observed.mean(axis=0)               # per-channel missing fraction
        if (miss_frac > max_window_missing).any():
            continue
        if not _edge_gap_ok(observed, bin_minutes):           # bounded edge gap
            continue
        # (c) nearest-neighbour LINEAR interpolation of the interior gaps; edge gaps are
        # nearest-filled and bounded by the gate above, so nothing is extrapolated. The result
        # is a clean continuous signal (no zeros, so the FFT-based seasonal layer sees no padding).
        sensor_values = (
            pd.DataFrame(sensor_values)
            .interpolate(method="linear", limit_direction="both")
            .to_numpy(dtype=np.float32)
        )
        sensor_values = np.nan_to_num(sensor_values, nan=0.0)
        win_start = start + wi * window_size
        if clock_features:
            time_feat = _cost_time_features(win_start, target_bins, bin_minutes)
            arr = np.concatenate([sensor_values, time_feat], axis=1)
        elif calendar_index:
            arr = np.concatenate(
                [sensor_values, _calendar_index_features(win_start, target_bins, bin_minutes)],
                axis=1)
        else:
            arr = sensor_values
        out.append((arr.astype(np.float32), win_start))
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
    clock_features: bool = False,
    calendar_index: bool = False,
    align_midnight: bool = False,
    cache_raw: bool = False,
) -> Dict[str, object]:
    """Full HRD preprocessing -> windowed classification dataset for CoST.

    ``clock_features`` (default False) keeps sensor channels only -- time is excluded
    from the model entirely. Set it True to append NUM_TIME_FEATURES CoST-style calendar
    covariates (raw minute/hour/dayofweek/day/dayofyear/month/weekofyear) to every window --
    the same channels salesforce/CoST's datautils.py builds, but scaled by the fixed
    CLOCK_MU/CLOCK_SD rather than a StandardScaler fitted on the windows, so the transform
    is identical here and in prepare_hrd_energy_sliding and never sees held-out data.

    Returns a dict with:
      X               (N, T, C) float32 windows, C = n_sensors (+ NUM_TIME_FEATURES
                      when clock_features is True)
      y               (N,)      int endpoint labels (0=control, 1=depressed)
      pids            (N,)      participant id per window
      window_ids      (N,)      unique id "pid_<isotime>" per window
      consistent_pids set       pids with baseline == endpoint
      labeled_pids    set       pids that carry an endpoint label
      trajectory_by_pid dict    pid -> Case-1 group (Pre1_Post1 ... Pre2_Post2)
      sensor_cols, n_sensors, n_features
    """
    # ee_by_pid_date (the daily emotional_energy lookup) is only used by the sliding
    # energy path (prepare_hrd_energy_sliding); the depression windowing ignores it.
    sensor_df, label_df, _ = load_hrd(csv_path, max_missing=max_missing,
                                      max_gap_minutes=max_gap_minutes, cache=cache_raw)
    label_by_pid, consistent_pids = _label_maps(label_df, label_col)
    sensor_cols = [c for c in SENSOR_COLS if c in sensor_df.columns]
    if not sensor_cols:
        raise ValueError("None of the expected sensor columns are present in the CSV.")
    window_minutes = window_hours * 60

    t0 = time.time()
    X_list, y_list, pid_list, wid_list = [], [], [], []
    for pid, g in sensor_df.groupby("pid", sort=True, observed=True):
        label = label_by_pid.get(pid, 0)
        for arr, win_start in _window_participant(g, window_minutes, bin_minutes,
                                                  sensor_cols, max_window_missing, z_score,
                                                  clock_features=clock_features,
                                                  calendar_index=calendar_index,
                                                  align_midnight=align_midnight):
            X_list.append(arr)
            y_list.append(label)
            pid_list.append(pid)
            wid_list.append(f"{pid}_{win_start.isoformat()}")

    if not X_list:
        raise RuntimeError("No windows produced; check window/bin/missing settings.")
    X = np.stack(X_list, axis=0).astype(np.float32)
    if clock_features:
        # Standardise the appended CoST calendar covariates with the FIXED scale (see
        # standardise_clock_channels): data-independent, so it neither sees the held-out
        # participants nor drifts between this path and prepare_hrd_energy_sliding.
        standardise_clock_channels(X, len(sensor_cols))
    n_features = X.shape[-1]
    print(
        f"[data_preprocessing] built {len(X_list):,} windows of shape "
        f"(T={X.shape[1]}, C={n_features}) from {len(set(pid_list))} participants | "
        f"{int(np.sum(np.asarray(y_list) == 1))} depressed-endpoint windows | "
        f"{time.time() - t0:.1f}s"
    )
    # Case-1 group label per participant (Pre1_Post1 ... Pre2_Post2), for group-stratified
    # rhythm analysis; NaN for participants without an endpoint survey.
    trajectory_by_pid = {
        p: t for p, t in zip(label_df["pid"], label_df["depression_trajectory"])
        if isinstance(t, str)
    }
    # baseline status per pid (endpoint status is already `label_by_pid` == y). Built directly
    # from depression_status_baseline rather than parsing `depression_trajectory`'s Pre1/Pre2
    # strings, so the (baseline, endpoint) pair used to pick trajectory-group participants
    # (e.g. dep->dep, non->non, dep->non, non->dep) doesn't depend on that naming convention.
    baseline_by_pid = {
        row["pid"]: int(row["depression_status_baseline"])
        for _, row in label_df.iterrows() if pd.notna(row["depression_status_baseline"])
    }
    return {
        "X": X,
        "y": np.asarray(y_list, dtype=int),
        "pids": np.asarray(pid_list),
        "window_ids": np.asarray(wid_list),
        "consistent_pids": consistent_pids,
        "labeled_pids": set(label_by_pid),
        "trajectory_by_pid": trajectory_by_pid,
        "baseline_by_pid": baseline_by_pid,
        "sensor_cols": sensor_cols,
        "n_sensors": len(sensor_cols),
        "n_features": n_features,
    }


# =============================================================================
# 3. cleaned tables -> SLIDING windows (emotional-energy probe, "mode B")
# =============================================================================

def _bin_window(g_win, win_start, target_bins, sensor_cols, bin_minutes, max_window_missing):
    """Bin the already-z-scored rows of ONE window ([win_start, win_start +
    target_bins*bin_minutes)) into a (target_bins, n_sensors) array, applying the SAME
    Algorithm-1 gate as _window_participant: a channel with more than max_window_missing
    missing bins, or an edge gap longer than EDGE_GAP_MINUTES, makes the window unusable
    -> return None. Interior gaps are linearly interpolated. Kept separate from
    _window_participant so the non-overlapping (depression) path is untouched."""
    n_sensors = len(sensor_cols)
    within = (g_win["timestamp"].to_numpy() - np.datetime64(win_start)) / np.timedelta64(1, "m")
    b = np.clip((within // bin_minutes).astype(np.int64), 0, target_bins - 1)
    sub = g_win[sensor_cols].copy()
    sub["_b"] = b
    grouped = sub.groupby("_b")
    mean_df = grouped[sensor_cols].mean()
    count_df = grouped[sensor_cols].count()
    sensor_values = np.full((target_bins, n_sensors), np.nan, dtype=np.float32)
    observed = np.zeros((target_bins, n_sensors), dtype=bool)
    for bi in mean_df.index:
        obs = count_df.loc[bi].to_numpy() > 0
        sensor_values[bi] = np.where(obs, mean_df.loc[bi].to_numpy(dtype=np.float32), np.nan)
        observed[bi] = obs
    if (1.0 - observed.mean(axis=0) > max_window_missing).any():   # per-channel missing-bin gate
        return None
    if not _edge_gap_ok(observed, bin_minutes):                    # bounded edge gap
        return None
    sensor_values = (pd.DataFrame(sensor_values)
                     .interpolate(method="linear", limit_direction="both")
                     .to_numpy(dtype=np.float32))
    return np.nan_to_num(sensor_values, nan=0.0)


def _sliding_windows_participant(g, day_ee_pairs, window_minutes, bin_minutes,
                                 sensor_cols, max_window_missing, z_score,
                                 clock_features=False, calendar_index=False):
    """One TRAILING window per labelled day (mode B): for day D the window covers the 7
    calendar days [D-6, D] (nowcast) and is labelled EE(D). Windows overlap by 6 days --
    fine because the train/val/test split is participant-level, so overlapping windows of
    one person never straddle the split boundary. When ``clock_features`` is True the CoST
    calendar covariates are appended as [sensor | clock], EXACTLY as _window_participant does,
    so the tensor matches an encoder pretrained with --with-clock-features. Returns
    [(arr, day, ee), ...]."""
    out = []
    target_bins = window_minutes // bin_minutes
    if z_score:                                       # per-participant, leakage-free (as elsewhere)
        g = g.copy()
        mu = g[sensor_cols].mean()
        sd = g[sensor_cols].std().replace(0.0, 1.0).fillna(1.0)
        g[sensor_cols] = (g[sensor_cols] - mu) / sd
    if len(g) < 2:
        return out
    tvals = g["timestamp"].to_numpy()                 # sensor_df is sorted by (pid, timestamp)
    span = pd.Timedelta(minutes=window_minutes)
    one_day = pd.Timedelta(days=1)
    for day, ee_val in day_ee_pairs:
        win_start = pd.Timestamp(day) - (span - one_day)   # (D-6) 00:00 for a 7-day window
        win_end = pd.Timestamp(day) + one_day              # (D+1) 00:00
        lo = int(np.searchsorted(tvals, np.datetime64(win_start)))
        hi = int(np.searchsorted(tvals, np.datetime64(win_end)))
        if hi - lo <= 10:                                  # too few raw samples (matches the >10 gate)
            continue
        arr = _bin_window(g.iloc[lo:hi], win_start, target_bins, sensor_cols,
                          bin_minutes, max_window_missing)
        if arr is not None:
            if clock_features:                             # append [sensor | clock] like _window_participant
                time_feat = _cost_time_features(win_start, target_bins, bin_minutes)
                arr = np.concatenate([arr, time_feat], axis=1)
            elif calendar_index:
                arr = np.concatenate(
                    [arr, _calendar_index_features(win_start, target_bins, bin_minutes)], axis=1)
            out.append((arr.astype(np.float32), day, float(ee_val)))
    return out


def prepare_hrd_energy_sliding(
    csv_path, window_hours=168, bin_minutes=15, max_missing=0.30,
    max_gap_minutes=30, max_window_missing=0.30, z_score=True, build_pretrain=True,
    clock_features=False, calendar_index=False, energy_stride=1,
):
    """Mode-B dataset for the emotional-energy probe, from a SINGLE CSV read.

    Returns two windowings of the same cleaned data:
      * PROBE  (sliding): one trailing 7-day window per labelled participant-day, labelled
                with that day's emotional_energy. Keys ``X`` / ``ee`` / ``pids`` / ``days``.
      * PRETRAIN (non-overlapping): the standard midnight-aligned windows over ALL
                participants (label-free SSL). Keys ``X_pretrain`` / ``pids_pretrain``.
                Skipped (both keys ``None``) when ``build_pretrain=False`` -- used when the
                caller REUSES an already-pretrained encoder (train_hrd.py --energy-probe)
                and only needs the probe windows.

    The caller pretrains on ``X_pretrain`` minus the test participants, then encodes ``X``
    and probes it -- so test participants stay out of pretraining exactly as in the
    non-sliding path. The only new property is that probe windows overlap WITHIN a
    participant (a statistical-independence caveat, not leakage).

    ``energy_stride`` keeps every k-th labelled day per participant. Consecutive windows
    share six of their seven days, so the INPUTS are highly redundant; the LABELS are not
    (each day's EE is a distinct target), which is why the useful range is small strides.
    Applied here rather than after the fact so the discarded windows are never built at
    all -- both the ~0.5 GiB array and the windowing time scale with 1/k."""
    sensor_df, label_df, ee_by_pid_date = load_hrd(csv_path, max_missing=max_missing,
                                                   max_gap_minutes=max_gap_minutes)
    sensor_cols = [c for c in SENSOR_COLS if c in sensor_df.columns]
    if not sensor_cols:
        raise ValueError("None of the expected sensor columns are present in the CSV.")
    window_minutes = window_hours * 60

    # per-participant (date, emotional_energy) pairs, sorted by date
    days_by_pid: Dict[str, List] = {}
    for (p, d), v in ee_by_pid_date.items():
        days_by_pid.setdefault(p, []).append((d, v))
    for p in days_by_pid:
        days_by_pid[p].sort()

    t0 = time.time()
    Xs, ps, ds, es = [], [], [], []                   # probe (sliding, one per labelled day)
    Xp, pp = [], []                                   # pretrain (non-overlapping, all pids)
    for pid, g in sensor_df.groupby("pid", sort=True, observed=True):
        if build_pretrain:
            for arr, win_start in _window_participant(g, window_minutes, bin_minutes,
                                                      sensor_cols, max_window_missing, z_score,
                                                      clock_features=clock_features,
                                                      calendar_index=calendar_index,
                                                      align_midnight=True):
                Xp.append(arr); pp.append(pid)
        for arr, day, ee_val in _sliding_windows_participant(
                g, days_by_pid.get(pid, [])[::energy_stride], window_minutes, bin_minutes,
                sensor_cols, max_window_missing, z_score, clock_features=clock_features,
                calendar_index=calendar_index):
            Xs.append(arr); ps.append(pid); ds.append(day); es.append(ee_val)

    if not Xs:
        raise RuntimeError("No sliding windows produced; check window/bin/missing settings.")
    X = np.stack(Xs).astype(np.float32)
    X_pre = np.stack(Xp).astype(np.float32) if build_pretrain else None
    if clock_features:
        # Same FIXED scale as prepare_hrd_dataset -- now genuinely identical, not merely the
        # same formula over a different reference set. That is what makes these probe windows
        # safe to feed to an encoder pretrained through the other path.
        ns = len(sensor_cols)
        standardise_clock_channels(X, ns)
        standardise_clock_channels(X_pre, ns)
    print(
        f"[data_preprocessing] mode-B sliding: {len(Xs):,} trailing probe windows "
        f"(1 per labelled day) from {len(set(ps))} participants | "
        f"{len(Xp):,} non-overlapping pretrain windows | shape (T={X.shape[1]}, "
        f"C={X.shape[-1]}) | {time.time() - t0:.1f}s"
    )
    return {
        "X": X, "ee": np.asarray(es, dtype=float), "pids": np.asarray(ps),
        "days": np.asarray(ds, dtype=object),
        "X_pretrain": X_pre,
        "pids_pretrain": np.asarray(pp) if build_pretrain else None,
        "sensor_cols": sensor_cols, "n_sensors": len(sensor_cols),
        "n_features": X.shape[-1],
    }
