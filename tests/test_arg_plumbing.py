"""Every architecture argument CoST accepts must actually be passed by the training run.

This is the test for a failure that cost two full GPU sweeps and produced results that
described a model nobody trained.

`noise_weight`, `noise_depth`, `noise_mask_frac` and `noise_span` were added to CoST, to
CoSTEncoder and to model_build -- and not to the `CoST(...)` call in train_hrd.py, which is
the one the training run uses. model_build's docstring says model construction lives in one
place; that call is the exception, and trusting the docstring instead of checking is what
let it through.

The failure was silent and total. metrics.json is written from vars(args), and so is the
variant-directory tag, so runs 2438763 and 2438765 were named `_nw0.3`, recorded
noise_weight=0.3, and trained the baseline: CoST took its own default of 0.0 and built no
branch. Both encoders came out at exactly 110.32 MB, the same size as the baseline's, and
RQ1 and RQ3 completed and wrote tables. Nothing failed until RQ2 tried to load one of those
checkpoints into a model that did have the branch.

A flag that reaches the config and the folder name but not the model is worse than one that
crashes, because every downstream artifact then describes a configuration that was never run.
"""
import inspect
from pathlib import Path

import pytest

from cost import CoST

ROOT = Path(__file__).resolve().parent.parent

# Not architecture: these are training callbacks with no effect on what is built or saved,
# and `noise_branch` is deliberately derived from noise_weight inside CoST so that a
# non-zero weight cannot be paired with an absent branch.
NOT_ARCHITECTURE = {"after_epoch_callback", "after_iter_callback", "noise_branch"}


def _call_arguments(path, callee):
    """The keyword names `callee` is constructed with in `path`.

    Two call shapes have to be understood, because the project uses both: train_hrd names
    every argument at the call site, while model_build assembles a `kw = dict(...)` and
    calls `CoST(**kw)`. Reading only the first shape would report model_build as passing
    nothing, which is how a test can be green, precise, and about the wrong thing.
    """
    import ast
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == callee):
            continue
        names |= {k.arg for k in node.keywords if k.arg}
        if any(k.arg is None for k in node.keywords):
            # CoST(**kw): take the keys of every dict literal in the enclosing module that
            # is built with dict(...) or updated into one.
            for n2 in ast.walk(tree):
                if isinstance(n2, ast.Call) and getattr(n2.func, "id", None) == "dict":
                    names |= {k.arg for k in n2.keywords if k.arg}
                if (isinstance(n2, ast.Call)
                        and getattr(getattr(n2.func, "attr", None), "__str__", str)() == "update"):
                    for a in n2.args:
                        if isinstance(a, ast.Dict):
                            names |= {k.value for k in a.keys if isinstance(k, ast.Constant)}
    return names


def test_train_hrd_passes_every_architecture_argument():
    accepted = set(inspect.signature(CoST.__init__).parameters) - {"self"}
    passed = _call_arguments(ROOT / "train_hrd.py", "CoST")
    missing = sorted(accepted - passed - NOT_ARCHITECTURE)
    assert not missing, (
        f"train_hrd.py builds CoST without {missing}. Those will silently take CoST's "
        f"defaults while metrics.json and the variant tag report the flag's value, so the "
        f"run will describe a model it did not train.")


# Read only inside CoST.fit, so a builder that never trains does not need them:
# `decomp_aug` picks the pretraining positive pair and `n_sensors` is handed to
# PretrainDataset. Both are verified above to be present in train_hrd, which does train.
TRAINING_ONLY = {"decomp_aug", "n_sensors"}


def test_model_build_passes_every_architecture_argument():
    """The other constructor -- used by the RQ scripts and by every random-init control. If
    the two disagree, a control is not the architecture it controls for."""
    accepted = set(inspect.signature(CoST.__init__).parameters) - {"self"}
    passed = _call_arguments(ROOT / "model_build.py", "CoST")
    missing = sorted(accepted - passed - NOT_ARCHITECTURE - TRAINING_ONLY)
    assert not missing, (
        f"model_build.py builds CoST without {missing}. It builds every random-init "
        f"control, so a missing architecture argument makes the control a different "
        f"network from the one it is the control for.")


def test_the_training_only_arguments_really_are_training_only():
    """The exemption above is only safe while these are unread outside fit(). If one starts
    shaping the network, model_build must pass it or every control silently diverges."""
    import ast
    src = Path(ROOT / "cost.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    # Scoped to the CoST class. PretrainDataset has its own self.n_sensors and reads it in
    # __getitem__, which is a different object entirely -- searching the whole module found
    # those and reported a failure about the wrong class.
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "CoST")
    fit = next(n for n in ast.walk(cls)
               if isinstance(n, ast.FunctionDef) and n.name == "fit")
    lo, hi = fit.lineno, fit.end_lineno
    for name in TRAINING_ONLY:
        uses = [n.lineno for n in ast.walk(cls)
                if isinstance(n, ast.Attribute) and n.attr == name
                and isinstance(n.value, ast.Name) and n.value.id == "self"]
        outside = [ln for ln in uses if not (lo <= ln <= hi)
                   and not src.split(chr(10))[ln - 1].strip().startswith("self." + name + " =")]
        assert not outside, (
            f"self.{name} is read at line(s) {outside}, outside CoST.fit. It is no longer "
            f"training-only, so model_build must pass it.")


def test_every_noise_argument_has_a_command_line_flag():
    """A knob reachable only from Python is a knob no sweep can set."""
    src = Path(ROOT / "train_hrd.py").read_text(encoding="utf-8")
    for name in ("noise_weight", "noise_depth", "noise_mask_frac", "noise_span"):
        flag = "--" + name.replace("_", "-")
        assert f'"{flag}"' in src, f"{name} has no {flag} flag"


@pytest.mark.parametrize("name", ["noise_weight", "noise_depth", "noise_mask_frac",
                                  "noise_span"])
def test_the_flag_value_reaches_the_constructor_not_just_the_config(name):
    """metrics.json is written from vars(args), so a flag always looks applied there. What
    matters is whether the same value is handed to CoST."""
    passed = _call_arguments(ROOT / "train_hrd.py", "CoST")
    assert name in passed, f"{name} is recorded in the config but never given to the model"
