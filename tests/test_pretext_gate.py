"""The gate must refuse a pair that is learnable but useless, not just one that is solved."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from pretext_difficulty import _ceilings  # noqa: E402


def verdict_for(top1_participant, ceiling, baseline=0.7198):
    """The decision the gate prints, extracted so the rule can be tested without a forward
    pass over 3007 windows -- which is what made it go unchecked while it was wrong."""
    hard = top1_participant <= 0.30
    if not hard:
        return "DO NOT SUBMIT"
    if ceiling is None:
        return "MEASURE THE CEILING FIRST"
    if ceiling <= baseline:
        return "DO NOT SUBMIT"
    return "SUBMIT"


def test_solved_at_init_is_refused():
    assert verdict_for(0.8223, 0.7151) == "DO NOT SUBMIT"


def test_hard_but_below_the_baseline_is_refused():
    """The case this project is actually in, and the one the old gate passed: the only pair
    that is hard caps the representation below what an untrained baseline already gets."""
    assert verdict_for(0.0312, 0.6658) == "DO NOT SUBMIT"


def test_hard_and_above_the_baseline_is_the_only_pass():
    assert verdict_for(0.0312, 0.7500) == "SUBMIT"


def test_a_ceiling_exactly_at_the_baseline_is_not_enough():
    assert verdict_for(0.0312, 0.7198) == "DO NOT SUBMIT"


def test_an_unmeasured_ceiling_does_not_pass_on_difficulty_alone():
    assert verdict_for(0.0312, None) == "MEASURE THE CEILING FIRST"


def test_ceilings_are_read_from_the_measurement_not_hard_coded():
    p = ROOT / "results" / "positive_pair_ceiling.json"
    if not p.exists():
        pytest.skip("ceilings not measured in this checkout")
    c = _ceilings(p)
    assert "participant pair" in c and "window pair" in c
    assert 0.5 < c["participant pair"] < 1.0


def test_a_missing_ceilings_file_is_empty_not_an_error():
    assert _ceilings(ROOT / "results" / "does_not_exist.json") == {}


def test_the_gate_help_still_parses():
    out = subprocess.run([sys.executable, str(ROOT / "analysis" / "pretext_difficulty.py"),
                          "--help"], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0 and "--baseline" in out.stdout
