"""The evaluation protocol: repeated grouped K-fold over every labelled participant, with
strictly decoupled seeds and a Nadeau-Bengio corrected margin.

WHY NOT THE OLD SINGLE HOLDOUT. It tested on 36 of 114 labelled participants and fit the
probe on 78, and the resulting detectable margin was 0.100 AUROC -- a property of the
cohort, not of any model. Measured over 267 seed-runs the per-seed SD of participant-level
AUROC is 0.1041, of which the Hanley-McNeil sampling term at n=36 is 0.0990: 90% of the
variance is test-set sampling noise, which is exactly the part that shrinks when every
participant is tested. Under 10-fold x 3 repeats the margin falls to 0.032.

That does not turn RQ3 into a win -- DSSL vs its random-init control is -0.0083, genuinely
zero. It makes the negative result provable: the gaps that matter (RF-on-raw over DSSL,
0.052; random projection over DSSL, 0.041) cross 0.032 and do not cross 0.100.

SEED DECOUPLING. Previously `seed == split_seed == model_seed`, so data variance and
optimisation variance were confounded and could not be separated after the fact. Here they
are drawn from one master seed through independent streams and are guaranteed distinct:

    split_seed  -- which participants land in which fold. Varies per REPEAT.
    model_seed  -- weight init and batch order. Varies per FOLD within a repeat.
    probe_seed  -- probe solver state. Varies per fold.

Holding split_seed fixed within a repeat is what makes the folds of that repeat a partition
of the same 114 participants; varying model_seed within it is what stops a single unlucky
initialisation from being read as a property of the split.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

__all__ = ["Fold", "make_folds", "nadeau_bengio", "required_margin", "paired_test",
           "delong_test", "make_lodo_folds"]


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


# --------------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------------
def nadeau_bengio(n_folds, n_repeats):
    """Variance-inflation factor for correlated resampled estimates: 1/K + n_test/n_train.

    Under K-fold, n_test/n_train = (1/F)/(1 - 1/F) = 1/(F-1). The naive 1/K is what makes
    K-fold CV look far more precise than it is; the second term is the price of folds that
    share training data.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    return 1.0 / (n_folds * n_repeats) + 1.0 / (n_folds - 1)


def required_margin(sd, n_folds, n_repeats, t_multiplier=2.010):
    """Smallest difference in means this design can call significant."""
    return t_multiplier * sd * math.sqrt(nadeau_bengio(n_folds, n_repeats))


def paired_test(a, b, n_folds, n_repeats):
    """Nadeau-Bengio corrected paired t-test between two arms scored on the same folds.

    Returns mean difference, corrected SE, t, and the design's detectable margin. The
    correction is the whole point: without it, 30 correlated folds are treated as 30
    independent samples and every arm looks significant.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"arms differ in shape: {a.shape} vs {b.shape}")
    d = a - b
    n = len(d)
    if n < 2:
        raise ValueError("need at least 2 paired estimates")
    sd = float(d.std(ddof=1))
    se = math.sqrt(nadeau_bengio(n_folds, n_repeats)) * sd
    margin = required_margin(sd, n_folds, n_repeats)
    mean = float(d.mean())

    # Zero variance is degenerate, not insignificant. If every fold produced the SAME
    # non-zero difference the effect is perfectly consistent and t is unbounded; if every
    # difference is zero there is no effect at all. Testing `se > 0` alone reported both as
    # "not significant", which is wrong in the first case and right only by accident in the
    # second.
    if sd == 0.0:
        nonzero = mean != 0.0
        return {"n": n, "mean_diff": mean, "sd": 0.0, "se_corrected": 0.0,
                "t": float("inf") if nonzero else 0.0,
                "wins": int((d > 0).sum()), "required_margin": 0.0,
                "significant": bool(nonzero), "degenerate": True}

    return {
        "n": n,
        "mean_diff": mean,
        "sd": sd,
        "se_corrected": float(se),
        "t": mean / float(se),
        "wins": int((d > 0).sum()),
        "required_margin": float(margin),
        "significant": bool(abs(mean) > margin),
        "degenerate": False,
    }


# --------------------------------------------------------------------------------------
# Pooled out-of-fold comparison
# --------------------------------------------------------------------------------------
def _structural(y, s):
    """DeLong's structural components (V10 over positives, V01 over negatives).

    psi(x, y) = 1 if x > y, 1/2 if tied, 0 otherwise. The mean of either component is the
    AUC itself, which is what makes their empirical covariance an estimator of its variance.
    """
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        raise ValueError("DeLong needs both classes present")
    psi = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :])
    return psi.mean(axis=1), psi.mean(axis=0)          # V10 (len m), V01 (len n)


def delong_test(y, s1, s2):
    """DeLong's test for two AUCs measured on the SAME subjects.

    The right test here for exactly the reason the fold-averaged one was wrong: two arms are
    scored on identical participants, so their AUCs are strongly positively correlated, and a
    test that ignores that correlation throws away most of the power. DeLong estimates the
    covariance directly, so var(AUC1 - AUC2) subtracts 2*cov rather than summing variances.

    Returns both AUCs, their difference, the standard error OF THE DIFFERENCE, z and a
    two-sided p. `var` is exact-zero when the two score vectors induce identical rankings;
    that is reported as degenerate rather than as an infinitely significant result.
    """
    y = np.asarray(y).astype(int)
    s1, s2 = np.asarray(s1, dtype=float), np.asarray(s2, dtype=float)
    if not (len(y) == len(s1) == len(s2)):
        raise ValueError(f"length mismatch: {len(y)}, {len(s1)}, {len(s2)}")
    v10_1, v01_1 = _structural(y, s1)
    v10_2, v01_2 = _structural(y, s2)
    m, n = len(v10_1), len(v01_1)
    a1, a2 = float(v10_1.mean()), float(v10_2.mean())

    s10 = np.cov(np.vstack([v10_1, v10_2]), ddof=1)
    s01 = np.cov(np.vstack([v01_1, v01_2]), ddof=1)
    var = ((s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m +
           (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n)
    diff = a1 - a2
    if var <= 0:
        return {"auc1": a1, "auc2": a2, "diff": diff, "se": 0.0,
                "z": 0.0 if diff == 0 else float("inf"),
                "p": 1.0 if diff == 0 else 0.0, "n": len(y), "n_pos": m,
                "degenerate": True}
    se = math.sqrt(var)
    z = diff / se
    return {"auc1": a1, "auc2": a2, "diff": diff, "se": se, "z": z,
            "p": math.erfc(abs(z) / math.sqrt(2)), "n": len(y), "n_pos": m,
            "degenerate": False}


def make_lodo_folds(pids, labels, years, n_repeats=3, master_seed=20260906):
    """Leave-one-study-year-out -- the GLOBEM benchmark's own split.

    Xu et al. evaluate cross-dataset generalisation by holding out one of the four GLOBEM
    study years and training on the other three, and report balanced accuracy at the WINDOW
    unit. Reproducing that split is what makes our numbers comparable to their 0.547; their
    within-dataset setup instead takes the first 80% of every user's data for training and
    the last 20% for test, so the same people sit on both sides and a model can score by
    remembering a personal baseline. Only the cross-dataset number is a fair target.

    The partition is DETERMINISTIC -- it is the calendar, not a sample -- so `split_seed` is
    inert here and carried only so one Fold type serves both protocols. `n_repeats` varies
    the model and probe seeds alone, which averages optimisation variance without pretending
    the data split was resampled. Verified on this cohort: 4 years, 0 participants spanning
    more than one, so the split is participant-disjoint by construction.
    """
    pids, labels, years = np.asarray(pids), np.asarray(labels), np.asarray(years).astype(str)
    if not (len(pids) == len(labels) == len(years)):
        raise ValueError("pids, labels and years must be parallel, one row per participant")
    uy = sorted(set(years.tolist()))
    if len(uy) < 2:
        raise ValueError(f"leave-one-year-out needs >=2 years, found {uy}")

    rng_split, rng_model, rng_probe = _streams(master_seed)
    out = []
    for rep in range(n_repeats):
        split_seed = int(rng_split.integers(1, 2 ** 31 - 1))     # inert; see docstring
        for f, y in enumerate(uy):
            te = np.flatnonzero(years == y)
            tr = np.flatnonzero(years != y)
            if len(np.unique(labels[tr])) < 2 or len(np.unique(labels[te])) < 2:
                raise ValueError(f"year {y} leaves a single-class split; LODO needs both "
                                 "classes on each side")
            out.append(Fold(
                repeat=rep, fold=f, split_seed=split_seed,
                model_seed=int(rng_model.integers(1, 2 ** 31 - 1)),
                probe_seed=int(rng_probe.integers(1, 2 ** 31 - 1)),
                test_pids=tuple(pids[te].tolist()), train_pids=tuple(pids[tr].tolist())))
    _assert_decoupled(out)
    return out
