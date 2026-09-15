"""Configuration of the HRD preprocessing: which CSV columns become which channels,
the cleaning thresholds and the windowing geometry, each with the reason for its value.
The dataclasses are frozen; the module-level names are views onto them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

import numpy as np
import pandas as pd


PathLike = Union[str, Path]


# A (pid, calendar date) -> emotional energy (1-5) lookup.
DailyEnergyLookup = Dict[Tuple[str, date], float]


# One produced window: the (T, C) array and the wall-clock time its first bin starts at.
Window = Tuple[np.ndarray, pd.Timestamp]


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


# Temporary grouping columns added to a working frame during binning.
_WINDOW_COL = "_w"


_BIN_COL = "_b"
