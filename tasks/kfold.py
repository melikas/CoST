"""Participant-level stratified k-fold, so every labelled person is tested exactly once.

The holdout this replaces evaluates 36 participants per seed out of 152 labelled ones, and
the 24 seeds overlap, so their estimates are correlated (rho = 0.46 measured). Both facts
push the same way: the Nadeau-Bengio corrected test needs a difference of 0.10 to 0.13 AUC
before it will call anything separable, which is larger than the gap between the best and
worst arm in the project. Two thirds of the labelled data sits out of the evaluation on
every run.

Under k-fold each participant contributes one held-out prediction and the AUC is computed
over all of them, which is where the variance goes. What this does NOT do is lower the bar
by fiat: the margin the new design requires has to be estimated from a permuted-label null
and written down BEFORE the real arms are compared, or it is just a smaller number chosen
after seeing the results. See kfold_eval.py --null.
"""
import numpy as np


def participant_folds(pids, pid_label, k=5, seed=0):
    """[(train_pids, test_pids), ...] -- k label-stratified, participant-disjoint folds.

    Every labelled participant appears in exactly one test fold, and in the training set of
    every other. Windows never cross: a participant is wholly in or wholly out, which is the
    property that makes the held-out AUC a statement about new people.

    Stratifying matters at this prevalence -- 52 of 152 are positive, so an unstratified
    fifth can hold as few as 4 positives by chance, and an AUC over that is mostly noise.
    """
    labelled = sorted(p for p in np.unique(np.asarray(pids)) if pid_label.get(p, -1) >= 0)
    if len(labelled) < k:
        raise ValueError(f"{len(labelled)} labelled participants cannot make {k} folds")
    rng = np.random.default_rng(seed)
    assign = {}
    for cls in (0, 1):
        members = [p for p in labelled if pid_label[p] == cls]
        rng.shuffle(members)
        # deal round-robin from a shuffled deck: fold sizes differ by at most one per class
        for i, p in enumerate(members):
            assign[p] = i % k
    return [(np.array([p for p in labelled if assign[p] != f]),
             np.array([p for p in labelled if assign[p] == f])) for f in range(k)]


def split_masks(pids, train_pids, test_pids, y, val_frac=0.25, seed=0):
    """(train, val, test) window masks for one fold, with the validation split carved out of
    TRAIN participants -- never the test ones, and never by window.

    A validation split taken by window would put the same person on both sides, and the
    penalty and probe family chosen on it would be chosen against people the probe has
    already seen.
    """
    pids = np.asarray(pids)
    lab = np.asarray(y) >= 0
    rng = np.random.default_rng(seed + 1000)
    tr = np.asarray(sorted(train_pids))
    rng.shuffle(tr)
    n_val = max(1, int(round(val_frac * len(tr))))
    val_pids, fit_pids = set(tr[:n_val]), set(tr[n_val:])
    return (np.isin(pids, list(fit_pids)) & lab,
            np.isin(pids, list(val_pids)) & lab,
            np.isin(pids, list(test_pids)) & lab)
