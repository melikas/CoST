"""Window cache loading and participant-level folds.

`load_npz` reads a cache built by scripts/build_cache.py into a `Cohort`. The labelled
cohort is the union of the cache's masks, never `y` alone, so unlabelled participants
are not read as negatives. `make_folds` (repeated grouped K-fold, stratified on the
label) splits by participant from one master seed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json

import numpy as np


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
    raw_X: object = None
    observed: object = None
    sensor_cols: tuple = ()
    metadata: dict = field(default_factory=dict)

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
            if v[0] not in (0, 1):
                raise ValueError(f"labelled participant {p} has a nonbinary or unknown endpoint")
            labs.append(int(v[0]))
        return np.array(ids), np.array(labs)

    def participant_years(self):
        """{pid: study year} read from the window ids -- the grouping LODO splits on.

        GLOBEM ran over four annual cohorts and the benchmark holds one out at a time. The
        year is taken from each window's ISO date rather than a separate column, so nothing
        extra has to survive preprocessing. `rsplit` on the LAST underscore, because GLOBEM
        pids contain underscores themselves ("INS-W_001_2018-04-09").

        This checks identifier consistency only. Different year-specific identifiers may
        belong to the same physical person; cross-year person disjointness is not verified.
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
        """Index arrays for one `Fold`.

        Pretraining sees every non-test window, which is what makes the unlabelled
        participants useful and what the original protocol did. The test participants are
        excluded from pretraining too -- not just from the probe -- so the representation a
        held-out participant is scored under never saw that participant.
        """
        test = set(fold.test_pids)
        train = set(fold.train_pids)
        if not test or not train or test & train:
            raise ValueError("fold needs nonempty, disjoint training and test identifiers")
        if not (test | train) <= self.labelled:
            raise ValueError("fold names identifiers outside the labelled cohort")
        in_test = np.isin(self.pids, list(test))
        excluded = in_test.copy()
        if fold.heldout_year is not None:
            years = self.participant_years()
            if any(years[p] != fold.heldout_year for p in test):
                raise ValueError("LODO test identifiers disagree with the held-out year")
            if any(years[p] == fold.heldout_year for p in train):
                raise ValueError("LODO training identifiers include the held-out year")
            # Domain exclusion includes unlabelled records, not just probe-test IDs.
            excluded |= np.array([years[p] == fold.heldout_year for p in self.pids])
        is_lab = np.isin(self.pids, list(self.labelled))
        return FoldData(
            pretrain=np.flatnonzero(~excluded),
            probe_train=np.flatnonzero(is_lab & np.isin(self.pids, list(train))),
            probe_test=np.flatnonzero(is_lab & in_test))


def labelled_from_masks(z, pids):
    """Participants named by the cache's train/val/test masks, asserted seed-invariant."""
    per_seed = {}
    for k in z.files:
        kind, _, seed = k.partition("/")
        if kind in _MASK_KINDS and seed:
            per_seed.setdefault(seed, set()).update(np.asarray(pids)[z[k]].tolist())
    if not per_seed and "labelled_pids" in z:
        return frozenset(z["labelled_pids"].astype(str))
    if not per_seed:
        raise ValueError(
            "the cache has no train/val/test masks, so the labelled cohort cannot be "
            "identified; folding on y would include unlabelled participants as negatives")
    uniq = {frozenset(v) for v in per_seed.values()}
    if len(uniq) != 1:
        sizes = sorted(len(u) for u in uniq)
        raise ValueError(f"the cache's labelled cohort differs across its {len(per_seed)} "
                         f"seeds (sizes {sizes}); folds built from it would not be comparable")
    labelled = next(iter(uniq))
    if "labelled_pids" in z and labelled != frozenset(z["labelled_pids"].astype(str)):
        raise ValueError("explicit labelled identifiers disagree with legacy masks")
    return labelled


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
               source=str(path), raw_X=z["raw_X"] if "raw_X" in z else None,
               observed=z["observed"] if "observed" in z else None,
               sensor_cols=tuple(z["sensor_cols"].astype(str)) if "sensor_cols" in z else (),
               metadata=json.loads(str(z["metadata_json"])) if "metadata_json" in z else {})
    z.close()
    if c.X.ndim != 3 or c.y.ndim != 1 or not np.isfinite(c.X).all():
        raise ValueError(f"{path}: expected finite (windows,time,features) input and 1D labels")
    if len(c.X) != len(c.y) or len(c.X) != len(c.pids):
        raise ValueError(f"{path}: X/y/pids lengths disagree")
    if not c.labelled:
        raise ValueError(f"{path}: the labelled cohort is empty")
    if not c.labelled <= set(c.pids):
        raise ValueError(f"{path}: labelled identifiers without windows")
    if c.window_ids is not None and (len(c.window_ids) != len(c.X) or
                                     len(np.unique(c.window_ids)) != len(c.window_ids)):
        raise ValueError(f"{path}: invalid or duplicate window identifiers")
    for key in ("raw_X", "observed"):
        value = getattr(c, key)
        if value is not None and value.shape != (*c.X.shape[:2], c.n_sensors):
            raise ValueError(f"{path}: {key} does not align with sensor windows")
    if c.raw_X is not None and not np.isfinite(c.raw_X).all():
        raise ValueError(f"{path}: nonfinite completed physical-unit windows")
    c.participants()
    return c


@dataclass(frozen=True)
class Fold:
    """One trained model's worth of work: which participants are held out, and the three
    independent seeds that produced it."""
    repeat: int
    fold: int
    split_seed: int
    model_seed: int
    probe_seed: int
    test_pids: tuple
    train_pids: tuple
    heldout_year: str | None = None

    @property
    def tag(self) -> str:
        return f"r{self.repeat}f{self.fold}"

    def as_dict(self):
        d = asdict(self)
        d["test_pids"] = list(self.test_pids)
        d["train_pids"] = list(self.train_pids)
        return d


def _streams(master_seed):
    """Three independent SeedSequence children. Distinct by construction, not by luck."""
    ss = np.random.SeedSequence(master_seed)
    return [np.random.default_rng(c) for c in ss.spawn(3)]


def make_folds(pids, labels, n_folds=10, n_repeats=3, master_seed=20260906):
    """Stratified, participant-disjoint folds: `n_repeats` independent partitions of every
    participant into `n_folds` groups, stratified on the participant-level label.

    A participant appears in exactly one test fold per repeat, so one repeat yields one
    out-of-fold prediction for every participant -- which is what lets the repeat be scored
    as a single AUROC over all of them rather than as an average of small noisy folds.
    """
    pids = np.asarray(pids)
    labels = np.asarray(labels)
    if len(pids) != len(labels):
        raise ValueError(f"pids and labels differ in length: {len(pids)} vs {len(labels)}")
    if len(set(pids.tolist())) != len(pids):
        raise ValueError("make_folds expects one row per participant, not per window")
    classes, counts = np.unique(labels, return_counts=True)
    if n_folds < 2 or n_repeats < 1 or len(classes) != 2 or not np.array_equal(classes, [0, 1]):
        raise ValueError("binary grouped CV needs two classes, >=2 folds and >=1 repeat")
    if counts.min() < n_folds:
        raise ValueError("each class needs at least n_folds participants for defined test AUROC")

    rng_split, rng_model, rng_probe = _streams(master_seed)
    out = []
    for rep in range(n_repeats):
        split_seed = int(rng_split.integers(1, 2 ** 31 - 1))
        part = np.empty(len(pids), dtype=int)
        # Stratify: deal each class round-robin into folds, from an order this repeat's
        # split_seed alone decides.
        rr = np.random.default_rng(split_seed)
        for cls in np.unique(labels):
            idx = np.flatnonzero(labels == cls)
            rr.shuffle(idx)
            part[idx] = np.arange(len(idx)) % n_folds
        for f in range(n_folds):
            te = np.flatnonzero(part == f)
            tr = np.flatnonzero(part != f)
            if len(te) == 0 or len(np.unique(labels[tr])) < 2:
                raise ValueError(
                    f"repeat {rep} fold {f} is degenerate ({len(te)} test, "
                    f"{len(np.unique(labels[tr]))} train classes) -- reduce n_folds")
            out.append(Fold(
                repeat=rep, fold=f, split_seed=split_seed,
                model_seed=int(rng_model.integers(1, 2 ** 31 - 1)),
                probe_seed=int(rng_probe.integers(1, 2 ** 31 - 1)),
                test_pids=tuple(pids[te].tolist()), train_pids=tuple(pids[tr].tolist())))
    _assert_decoupled(out)
    return out


def make_year_folds(pids, labels, years, master_seed):
    """Leave-one-year-out: fold k holds out every participant of the k-th study year and trains on
    the labelled participants of all other years, as in the GLOBEM cross-year benchmark.

    Identifiers are participant-years. A student enrolled in two years has two unlinked
    identifiers, so these folds are year-disjoint, not verified person-disjoint."""
    pids, labels = np.asarray(pids), np.asarray(labels)
    if len(set(pids.tolist())) != len(pids):
        raise ValueError("make_year_folds expects one row per participant, not per window")
    _, rng_model, rng_probe = _streams(master_seed)
    of = np.array([years[p] for p in pids])
    out = []
    for f, year in enumerate(sorted(set(of))):
        te, tr = of == year, of != year
        if len(np.unique(labels[te])) < 2 or len(np.unique(labels[tr])) < 2:
            raise ValueError(f"held-out year {year} needs both classes on each side")
        out.append(Fold(repeat=0, fold=f, split_seed=master_seed,
                        model_seed=int(rng_model.integers(1, 2 ** 31 - 1)),
                        probe_seed=int(rng_probe.integers(1, 2 ** 31 - 1)),
                        test_pids=tuple(pids[te].tolist()), train_pids=tuple(pids[tr].tolist()),
                        heldout_year=year))
    _assert_decoupled(out)
    return out


def _assert_decoupled(folds):
    """The invariant DECISION 3 asked for, enforced rather than documented."""
    for f in folds:
        if len({f.split_seed, f.model_seed, f.probe_seed}) != 3:
            raise AssertionError(f"seeds collided on {f.tag}: {f.split_seed}, "
                                 f"{f.model_seed}, {f.probe_seed}")
    per_repeat = {}
    for f in folds:
        per_repeat.setdefault(f.repeat, set()).add(f.split_seed)
    for rep, seeds in per_repeat.items():
        if len(seeds) != 1:
            raise AssertionError(f"repeat {rep} has {len(seeds)} split seeds; a repeat is "
                                 "one partition and must have exactly one")
    if len({f.model_seed for f in folds}) != len(folds):
        raise AssertionError("model seeds are not unique across folds")
