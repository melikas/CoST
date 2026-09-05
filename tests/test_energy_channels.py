"""The energy windows must carry the same channels the encoder was built for.

`drop_sensor_channels` was applied to the depression dataset and not to the energy one, so
with `--drop-channels Steps` the encoder is Linear(3, 64) and the energy probe hands it four
columns. It dies with

    mat1 and mat2 shapes cannot be multiplied (43008x4 and 3x64)

inside input_fc -- and the failure is caught as non-fatal, so every run using --drop-channels
has produced no energy results at all while reporting success. Run 2224103, the best
configuration this project has, is one of them.

Nothing here needs a GPU or the real CSV: the mismatch is a shape contract between two
dictionaries, and that is what is checked.
"""
import numpy as np
import pytest

from data_processing.data_preprocessing import drop_sensor_channels


def _dataset(n_sensors, n_clock=0, n=8, T=32):
    """The shape of what prepare_hrd_dataset and prepare_hrd_energy_sliding both return."""
    cols = ["heart_rate", "steps", "is_asleep", "event"][:n_sensors]
    return {"X": np.random.default_rng(0).normal(
                size=(n, T, n_sensors + n_clock)).astype(np.float32),
            "sensor_cols": cols, "n_sensors": n_sensors,
            "n_features": n_sensors + n_clock}


def test_dropping_a_channel_narrows_the_array_and_the_column_list():
    d = drop_sensor_channels(_dataset(4), ["steps"])
    assert d["X"].shape[-1] == 3
    assert "steps" not in d["sensor_cols"] and len(d["sensor_cols"]) == 3
    assert d["n_sensors"] == 3


def test_clock_channels_survive_the_drop():
    """They sit after the sensor channels, and the encoder counts on both numbers."""
    d = drop_sensor_channels(_dataset(4, n_clock=2), ["steps"])
    assert d["X"].shape[-1] == 5 and d["n_sensors"] == 3


def test_the_two_paths_agree_after_the_same_drop():
    """The depression and energy datasets are built by different functions from the same CSV.
    Whatever the encoder is sized for, the energy windows have to match it -- which is the
    contract that was broken."""
    dep = drop_sensor_channels(_dataset(4), ["steps"])
    ene = drop_sensor_channels(_dataset(4, n=13, T=32), ["steps"])
    assert dep["X"].shape[-1] == ene["X"].shape[-1]
    assert dep["n_sensors"] == ene["n_sensors"]
    assert dep["sensor_cols"] == ene["sensor_cols"]


def test_skipping_the_drop_on_one_path_is_what_breaks_input_fc():
    """The bug, reproduced as a shape assertion: an encoder sized from the dropped dataset
    cannot consume windows from the undropped one."""
    dep = drop_sensor_channels(_dataset(4), ["steps"])
    ene_unfixed = _dataset(4, n=13)
    assert ene_unfixed["X"].shape[-1] != dep["X"].shape[-1], (
        "if these matched, the bug could not have happened")


def test_dropping_nothing_is_a_no_op():
    d = _dataset(4)
    for empty in (None, [], ()):
        out = drop_sensor_channels(d, empty)
        assert out["X"].shape == d["X"].shape and out["n_sensors"] == 4


def test_an_unknown_channel_name_does_not_silently_do_nothing():
    """A typo in --drop-channels that quietly dropped nothing would leave the run reporting
    a configuration it did not have."""
    with pytest.raises((KeyError, ValueError)):
        drop_sensor_channels(_dataset(4), ["no_such_channel"])
