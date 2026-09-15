"""HRD CSV -> cleaned tables, one participant at a time (`iter_hrd`): read and rename
columns, derive the binary sleep indicator, clean sensor values, drop participants over the
missingness budget, interpolate short gaps, and build the label and daily emotional-energy
tables.
"""
from __future__ import annotations

from typing import Any, Dict, Set, Tuple

import numpy as np
import pandas as pd

from data_processing.hrd_config import CHANNELS, CLEANING, DailyEnergyLookup


def _parse_raw_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rename one raw frame to canonical names and parse types. Rows without a parseable
    timestamp are dropped: they cannot be placed on a time grid."""
    df = df.rename(columns=CHANNELS.csv_rename)

    df["pid"] = df["pid"].astype(str).str.lower().str.strip()
    df["timestamp"] = pd.to_datetime(df["dateTime"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["pid", "timestamp"])

    for column in CHANNELS.raw_numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(np.float32)
    df[CHANNELS.energy] = pd.to_numeric(df[CHANNELS.energy], errors="coerce")
    return df


def iter_hrd(csv_path, max_missing=CLEANING.max_participant_missing,
             max_gap_minutes=CLEANING.max_gap_minutes, summary=None, chunksize=100_000):
    """Yield cleaned participant tables from a participant-contiguous CSV.

    Memory is bounded by one participant plus a CSV chunk. Reappearing participant blocks
    fail rather than silently making two subjects or losing data. The local export is
    checked for this property during the complete build; arbitrary input order requires an
    explicit external sort, not a random window split.
    """
    cols = (["pid"] + list(CHANNELS.csv_rename) + CHANNELS.event + CHANNELS.label
            + CHANNELS.extra_label + [CHANNELS.energy])
    summary = {} if summary is None else summary
    summary.update(raw_rows=0, raw_participants=0, retained_participants=0,
                   rejected_missingness=[], invalid_timestamp_rows=0)
    current, pieces, seen = None, [], set()

    def prepare(pieces):
        raw = pd.concat(pieces, ignore_index=True)
        parsed = _parse_raw_frame(raw)
        summary["invalid_timestamp_rows"] += len(raw) - len(parsed)
        if parsed.empty:
            raise ValueError("participant block has no parseable timestamps")
        parsed = _clean_sensor_values(parsed)
        labels = _build_label_table(parsed)
        energy = _build_daily_energy_lookup(parsed)
        grid = _minute_grid(parsed)
        pid = str(grid.pid.iloc[0])
        summary["raw_participants"] += 1
        if not len(_participants_within_missing_budget(grid, max_missing)):
            summary["rejected_missingness"].append(pid)
            return None
        summary["retained_participants"] += 1
        return _interpolate_short_gaps(grid, max_gap_minutes), labels, energy

    with pd.read_csv(csv_path, usecols=cols, low_memory=False, chunksize=chunksize) as reader:
        for chunk in reader:
            summary["raw_rows"] += len(chunk)
            chunk["pid"] = chunk["pid"].astype(str).str.lower().str.strip()
            # Contiguous runs expose an interleaved ID instead of silently grouping it away.
            for _, block in chunk.groupby(chunk.pid.ne(chunk.pid.shift()).cumsum(), sort=False):
                pid = block.pid.iloc[0]
                if current is not None and pid != current:
                    ready = prepare(pieces)
                    if ready is not None:
                        yield ready
                    seen.add(current)
                    pieces = []
                if pid in seen:
                    raise ValueError(f"HRD CSV participant {pid} occurs in noncontiguous blocks")
                current = pid
                pieces.append(block.copy())
    if pieces:
        ready = prepare(pieces)
        if ready is not None:
            yield ready


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

    # The local export already encodes no-event minutes as zero. A missing export value
    # is not proof of observation and stays missing.
    return df


def _build_label_table(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the repeated per-row labels into one row per participant."""
    def first_valid(series: pd.Series) -> Any:
        """First non-null value; labels repeat, but some rows are blank."""
        present = series.dropna()
        return present.iloc[0] if len(present) else np.nan

    label_columns = CHANNELS.label + CHANNELS.extra_label
    conflicts = df.groupby("pid")[label_columns].nunique(dropna=True).gt(1)
    if conflicts.any().any():
        raise ValueError("conflicting repeated participant labels/scores in HRD export")
    return df.groupby("pid")[label_columns].agg(first_valid).reset_index()


def _minute_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Insert absent minutes before measuring coverage or filling gaps.

    Original per-channel observation flags survive interpolation and windowing. The grid
    follows the supplied local timestamps; no timezone or DST correction is inferred.
    """
    frames = []
    for pid, g in df.groupby("pid", sort=True, observed=True):
        g = g.sort_values("timestamp")
        t = pd.DatetimeIndex(g.timestamp)
        if t.has_duplicates or not (t == t.floor("min")).all():
            raise ValueError(f"HRD identifier {pid} has duplicate or off-minute timestamps")
        grid = pd.date_range(t[0], t[-1], freq="min", name="timestamp")
        values = g.set_index("timestamp")[CHANNELS.sensors].reindex(grid)
        for c in CHANNELS.sensors:
            values[f"observed/{c}"] = values[c].notna()
        values["pid"] = pid
        frames.append(values.reset_index())
    if not frames:
        raise ValueError("HRD export contains no valid participants")
    result = pd.concat(frames, ignore_index=True)
    result["pid"] = result["pid"].astype("category")
    return result


def _participants_within_missing_budget(df: pd.DataFrame, max_missing: float) -> pd.Index:
    """Participant ids whose wear-dependent channels are complete enough to keep."""
    missing_per_participant = (df[CHANNELS.wear_dependent]
                               .isna()
                               .groupby(df["pid"], observed=True)
                               .mean()
                               .mean(axis=1))
    return missing_per_participant.index[missing_per_participant <= max_missing]


def _interpolate_short_gaps(df: pd.DataFrame, max_gap_minutes: int) -> pd.DataFrame:
    """Linearly fill non-wear gaps up to `max_gap_minutes`, within each participant.

    Longer gaps stay NaN on purpose: they are real non-wear and the per-window gate should
    see them rather than have them fabricated away. `limit_area="inside"` keeps the fill
    strictly between observed samples, so nothing is extrapolated past a record's edges.
    """
    if max_gap_minutes < 0:
        raise ValueError("max_gap_minutes must be nonnegative")
    if max_gap_minutes == 0:
        return df
    def fill(s):
        missing = s.isna()
        runs = missing.ne(missing.shift(fill_value=False)).cumsum()
        length = missing.groupby(runs).transform("sum")
        candidate = s.interpolate(method="linear", limit_area="inside")
        return s.where(~missing | (length > max_gap_minutes), candidate)
    df[CHANNELS.wear_dependent] = df.groupby("pid", observed=True)[CHANNELS.wear_dependent].transform(fill)
    df[CHANNELS.activity] = df[CHANNELS.activity].clip(lower=0)
    return df


def _build_daily_energy_lookup(df: pd.DataFrame) -> DailyEnergyLookup:
    """Build the (pid, calendar date) -> emotional energy map used to label windows."""
    answered = df.dropna(subset=[CHANNELS.energy]).copy()
    answered["_date"] = answered["timestamp"].dt.normalize()
    daily = answered.groupby(["pid", "_date"])[CHANNELS.energy].first()
    return {(pid, day.date()): float(value) for (pid, day), value in daily.items()}


def _endpoint_labels_and_consistent_pids(
    label_df: pd.DataFrame, label_col: str
) -> Tuple[Dict[str, int], Set[str]]:
    """Return (label per participant, participants whose baseline == endpoint status).

    A changed baseline/endpoint status is a measured transition, not presumed label noise.
    """
    label_by_pid: Dict[str, int] = {}
    for _, row in label_df.iterrows():
        value = row.get(label_col)
        if pd.notna(value):
            if value not in (0, 1, False, True):
                raise ValueError(f"invalid binary endpoint label for {row['pid']}")
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
