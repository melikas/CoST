"""The combined RQ3 rungs: right blocks, right widths, and it refuses a mismatched one."""
import numpy as np
import pytest

from experiment_q3 import combined_rungs

N = 40
V = np.random.default_rng(0).normal(size=(N, 16)).astype(np.float32)
SKIP = np.random.default_rng(1).normal(size=(N, 16)).astype(np.float32)
COS = np.random.default_rng(2).normal(size=(N, 5)).astype(np.float32)
STR = np.random.default_rng(3).normal(size=(N, 7)).astype(np.float32)
RI = np.random.default_rng(4).normal(size=(N, 16)).astype(np.float32)
FULL = {"Cosinor (paper)": COS, "Structured rhythm": STR, "Random-init": RI}


def test_widths_and_membership():
    out = combined_rungs(FULL, V, SKIP)
    assert set(out) == {"Raw skip only", "DSSL + rhythm", "DSSL + rhythm + raw skip",
                        "Random-init + rhythm + raw skip"}
    assert out["DSSL + rhythm"].shape == (N, 16 + 5 + 7)
    assert out["DSSL + rhythm + raw skip"].shape == (N, 16 + 5 + 7 + 16)
    assert out["Random-init + rhythm + raw skip"].shape == (N, 16 + 5 + 7 + 16)
    # the control differs from the treatment in the LEARNED block and nowhere else
    a, b = out["DSSL + rhythm + raw skip"], out["Random-init + rhythm + raw skip"]
    assert np.allclose(a[:, 16:], b[:, 16:]), "control and treatment differ outside V"
    assert not np.allclose(a[:, :16], b[:, :16]), "control and treatment share V"
    assert np.allclose(out["DSSL + rhythm"], np.concatenate([V, COS, STR], 1))


def test_rhythm_rungs_absent_when_no_rhythm_arm():
    """Both rhythm rungs are guarded by try/except upstream, so this really happens."""
    out = combined_rungs({"Random-init": RI}, V, SKIP)
    assert set(out) == {"Raw skip only"}


def test_one_rhythm_arm_is_enough():
    out = combined_rungs({"Structured rhythm": STR, "Random-init": RI}, V, SKIP)
    assert out["DSSL + rhythm"].shape == (N, 16 + 7)


def test_refuses_a_block_that_is_not_per_window():
    """A per-SUBJECT block would concatenate happily against a per-subject arm and give a
    plausible number, so the mismatch has to raise rather than be discovered in the table."""
    bad = dict(FULL, **{"Cosinor (paper)": COS[:N // 2]})
    with pytest.raises(ValueError, match="not all per-window"):
        combined_rungs(bad, V, SKIP)
