"""Folds: everyone tested once, nobody on both sides, and stratified."""
import numpy as np
import pytest

from tasks.kfold import participant_folds, split_masks

PIDS = np.array([f"p{i:03d}" for i in range(152)])
LAB = {p: (1 if i < 52 else 0) for i, p in enumerate(PIDS)}


def test_every_participant_is_tested_exactly_once():
    folds = participant_folds(PIDS, LAB, k=5, seed=0)
    seen = np.concatenate([te for _, te in folds])
    assert len(seen) == len(set(seen)) == 152


def test_train_and_test_never_share_a_participant():
    for tr, te in participant_folds(PIDS, LAB, k=5, seed=0):
        assert not (set(tr) & set(te))
        assert len(tr) + len(te) == 152


def test_folds_are_stratified():
    """At 52 of 152 positive, an unstratified fifth can hold 4 positives by chance."""
    for _, te in participant_folds(PIDS, LAB, k=5, seed=0):
        pos = sum(LAB[p] for p in te)
        assert abs(pos - 52 / 5) <= 1, pos


def test_unlabelled_participants_are_excluded():
    lab = dict(LAB)
    for p in PIDS[:20]:
        lab[p] = -1
    seen = np.concatenate([te for _, te in participant_folds(PIDS, lab, k=5, seed=0)])
    assert len(seen) == 132 and not (set(seen) & set(PIDS[:20]))


def test_a_different_seed_gives_a_different_partition():
    a = participant_folds(PIDS, LAB, k=5, seed=0)
    b = participant_folds(PIDS, LAB, k=5, seed=1)
    assert any(set(x[1]) != set(y[1]) for x, y in zip(a, b))


def test_validation_is_carved_from_train_participants_only():
    win_pids = np.repeat(PIDS, 3)
    y = np.repeat([LAB[p] for p in PIDS], 3)
    tr_p, te_p = participant_folds(PIDS, LAB, k=5, seed=0)[0]
    tr, va, te = split_masks(win_pids, tr_p, te_p, y, seed=0)
    assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
    assert not (set(win_pids[va]) & set(te_p)), "a test participant reached validation"
    assert set(win_pids[tr]) | set(win_pids[va]) == set(tr_p)
    assert set(win_pids[te]) == set(te_p)


def test_unlabelled_windows_never_enter_any_split():
    win_pids = np.repeat(PIDS, 3)
    y = np.repeat([LAB[p] for p in PIDS], 3).astype(int)
    y[::7] = -1
    tr_p, te_p = participant_folds(PIDS, LAB, k=5, seed=0)[0]
    for m in split_masks(win_pids, tr_p, te_p, y, seed=0):
        assert (y[m] >= 0).all()


def test_too_few_participants_is_an_error():
    with pytest.raises(ValueError, match="cannot make"):
        participant_folds(PIDS[:3], LAB, k=5)
