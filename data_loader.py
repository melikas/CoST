"""Cohort and window loading.

The one job that matters here is getting the LABELLED COHORT right. `y` in a window cache is
0/1 for every participant, including the ones present only to pretrain on -- on HRD that is
38 of 152. Reading labels off `y` alone silently files those 38 as negatives, which inflates
the negative class by 50% and corrupts every fold. The labelled cohort is instead the union
of the cache's train/val/test masks, which is exactly the set the run that produced the cache
treated as labelled.

Per fold, the split follows the original protocol: pretrain on every window whose participant
is not in the test fold -- unlabelled participants included, since pretraining uses no labels
-- and probe on the labelled participants only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Cohort", "load_npz", "FoldData"]

_MASK_KINDS = ("train_mask", "val_mask", "test_mask")


@dataclass
class FoldData:
    """Index arrays into the cohort's windows, for one fold."""
    pretrain: np.ndarray        # windows to pretrain on (labelled + unlabelled, non-test)
    probe_train: np.ndarray     # labelled windows of the fold's training participants
    probe_test: np.ndarray      # labelled windows of the fold's held-out participants

    def summary(self):
        return {"pretrain": int(len(self.pretrain)),
                "probe_train": int(len(self.probe_train)),
                "probe_test": int(len(self.probe_test))}


@dataclass
class Cohort:
    X: np.ndarray               # (N, T, D) windows
    y: np.ndarray               # (N,) window labels; meaningful only for `labelled`
    pids: np.ndarray            # (N,) participant id per window
    n_sensors: int
    bins_per_day: int
    labelled: frozenset         # participants that actually carry a label
    # "pid_<isotime>" per window. RQ2 needs each window's elapsed start time to require that
    # a personal baseline be CONTIGUOUS: windows are indexed by position, but the quality
    # gate drops some, so position stops proxying time. Measured on this cohort, 3.8% of
    # consecutive stored windows are more than 7 days apart, median gap 21 days.
    window_ids: object = None
    source: str = ""

    @property
    def seq_len(self):
        return int(self.X.shape[1])

    @property
    def n_features(self):
        return int(self.X.shape[-1])

    @property
    def bin_minutes(self):
        return int(round(24 * 60 / self.bins_per_day))

    def participants(self):
        """(ids, labels) for the labelled cohort only, one row per participant."""
        ids, labs = [], []
        for p in sorted(self.labelled):
            v = np.unique(self.y[self.pids == p])
            if len(v) != 1:
                raise ValueError(f"participant {p} carries {len(v)} distinct labels")
            ids.append(p)
            labs.append(int(v[0]))
        return np.array(ids), np.array(labs)

    def participant_years(self):
        """{pid: study year} read from the window ids -- the grouping LODO splits on.

        GLOBEM ran over four annual cohorts and the benchmark holds one out at a time. The
        year is taken from each window's ISO date rather than a separate column, so nothing
        extra has to survive preprocessing. `rsplit` on the LAST underscore, because GLOBEM
        pids contain underscores themselves ("INS-W_001_2018-04-09").

        A participant appearing in two years would break the disjointness the split relies
        on, so that is checked by the caller rather than assumed -- verified on this cohort
        as 4 years and 0 participants spanning more than one.
        """
        if self.window_ids is None:
            raise ValueError("participant_years needs window_ids")
        out = {}
        for w, p in zip(np.asarray(self.window_ids).astype(str),
                        np.asarray(self.pids).astype(str)):
            y = w.rsplit("_", 1)[1][:4]
            if not (len(y) == 4 and y.isdigit()):
                raise ValueError(f"window id {w!r} does not end in an ISO date")
            out.setdefault(p, {}).setdefault(y, 0)
            out[p][y] += 1
        mixed = {p: sorted(c) for p, c in out.items() if len(c) > 1}
        if mixed:
            raise ValueError(f"{len(mixed)} participants span more than one study year, so a "
                             f"year split would not be participant-disjoint: "
                             f"{dict(list(mixed.items())[:3])}")
        return {p: next(iter(c)) for p, c in out.items()}

    def fold_data(self, fold):
        """Index arrays for one `cv.Fold`.

        Pretraining sees every non-test window, which is what makes the unlabelled
        participants useful and what the original protocol did. The test participants are
        excluded from pretraining too -- not just from the probe -- so the representation a
        held-out participant is scored under never saw that participant.
        """
        test = set(fold.test_pids)
        train = set(fold.train_pids)
        in_test = np.isin(self.pids, list(test))
        is_lab = np.isin(self.pids, list(self.labelled))
        return FoldData(
            pretrain=np.flatnonzero(~in_test),
            probe_train=np.flatnonzero(is_lab & np.isin(self.pids, list(train))),
            probe_test=np.flatnonzero(is_lab & in_test))


def labelled_from_masks(z, pids):
    """Participants named by the cache's train/val/test masks, asserted seed-invariant."""
    per_seed = {}
    for k in z.files:
        kind, _, seed = k.partition("/")
        if kind in _MASK_KINDS and seed:
            per_seed.setdefault(seed, set()).update(np.asarray(pids)[z[k]].tolist())
    if not per_seed:
        raise ValueError(
            "the cache has no train/val/test masks, so the labelled cohort cannot be "
            "identified; folding on y would include unlabelled participants as negatives")
    uniq = {frozenset(v) for v in per_seed.values()}
    if len(uniq) != 1:
        sizes = sorted(len(u) for u in uniq)
        raise ValueError(f"the cache's labelled cohort differs across its {len(per_seed)} "
                         f"seeds (sizes {sizes}); folds built from it would not be comparable")
    return next(iter(uniq))


def load_npz(path):
    """Load a window cache into a `Cohort`."""
    z = np.load(path, allow_pickle=True)
    for k in ("X", "y", "pids", "n_sensors", "bins_per_day"):
        if k not in z.files:
            raise ValueError(f"{path} is missing {k!r}")
    pids = z["pids"]
    c = Cohort(X=z["X"], y=z["y"], pids=pids,
               n_sensors=int(z["n_sensors"]), bins_per_day=int(z["bins_per_day"]),
               labelled=labelled_from_masks(z, pids),
               window_ids=z["window_ids"] if "window_ids" in z.files else None,
               source=str(path))
    if len(c.X) != len(c.y) or len(c.X) != len(c.pids):
        raise ValueError(f"{path}: X/y/pids lengths disagree")
    if not c.labelled:
        raise ValueError(f"{path}: the labelled cohort is empty")
    return c
