"""Small helpers shared by the CoST pipeline. Everything here is reachable from
``cost.py`` (the NaN-padding / centering used when a window is split) or from the
training entry points (``init_dl_program``)."""
import random

from sklearn.model_selection import train_test_split

import numpy as np
import torch


def torch_pad_nan(arr, left=0, right=0, dim=0):
    for n, side in ((left, "left"), (right, "right")):
        if n > 0:
            shape = list(arr.shape)
            shape[dim] = n
            pad = torch.full(shape, np.nan)
            arr = torch.cat((pad, arr) if side == "left" else (arr, pad), dim=dim)
    return arr


def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in (np.float16, np.float32, np.float64)
    pad = target_length - array.shape[axis]
    if pad <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    npad[axis] = (pad // 2, pad - pad // 2) if both_side else (0, pad)
    return np.pad(array, pad_width=npad, mode="constant", constant_values=np.nan)


def split_with_nan(x, sections, axis=0):
    """Split into `sections` equal parts, NaN-padding the short tail."""
    assert x.dtype in (np.float16, np.float32, np.float64)
    arrs = np.array_split(x, sections, axis=axis)
    return [pad_nan_to_target(a, arrs[0].shape[axis], axis=axis) for a in arrs]


def centerize_vary_length_series(x):
    """Roll each series so its non-NaN span sits in the middle of the window."""
    prefix = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix + suffix) // 2 - prefix
    offset[offset < 0] += x.shape[1]
    rows, cols = np.ogrid[: x.shape[0], : x.shape[1]]
    return x[rows, cols - offset[:, None]]


def init_dl_program(device_name, seed=None, use_cudnn=True, deterministic=False,
                    benchmark=False, use_tf32=False, max_threads=None):
    """Seed every RNG and set the torch/cuDNN flags. Returns the torch device(s)."""
    if max_threads is not None:
        torch.set_num_threads(max_threads)
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)
        try:
            import mkl
        except ImportError:
            pass
        else:
            mkl.set_num_threads(max_threads)

    if seed is not None:                       # distinct streams per library, as upstream CoST
        random.seed(seed)
        np.random.seed(seed + 1)
        torch.manual_seed(seed + 2)

    names = [device_name] if isinstance(device_name, (str, int)) else device_name
    devices = []
    for i, t in enumerate(reversed(names)):
        dev = torch.device(t)
        devices.append(dev)
        if dev.type == "cuda":
            assert torch.cuda.is_available()
            torch.cuda.set_device(dev)
            if seed is not None:
                torch.cuda.manual_seed(seed + 3 + i)
    devices.reverse()

    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
    return devices[0] if len(devices) == 1 else devices


def pid_majority_label(labels):
    """One participant-level label from that participant's per-window labels: the MAJORITY.

    Canonical definition, shared so the three places that need a per-participant label cannot
    drift apart: the split's `pid_summary_label` (globem_preprocessing), the probe's
    subject-level metrics (_eval_protocols.participant_aggregate) and the supervised baselines'
    (train_hrd.participant_aggregate).

    In WEEKLY mode a participant's windows carry DIFFERENT labels over time (each window takes
    the weekly survey inside its own date span), so collapsing them is a real choice. The
    subject-level score is the MEAN of that participant's window predictions -- an implicit
    "is this person depressed overall?" -- so the label it is scored against has to be the same
    kind of summary. Taking one window's label instead (the code used the chronologically first)
    scored ~1 in 5 participants against an arbitrary week: on this dataset 66% of participants
    have mixed weekly labels and for 139/703 (19.8%) the first week disagrees with the majority.

    TIES (equal 0s and 1s) resolve to 0 -- and that is now stated in the code rather than
    inherited from `round`. The previous implementation was `int(round(mean))`, which gives 0
    on a tie only because Python/NumPy round half-to-EVEN; written that way the policy is
    invisible, reads like half-up to most people, and would silently flip if the expression
    were ever refactored (e.g. to `mean > 0.5`, or `np.floor(mean + 0.5)`, both of which give
    1). It is not a rare corner either: on binary weekly labels an exact 50/50 split lands on
    roughly one participant in six, so the tie direction decides real labels.

    0 is the deliberate choice: it means "not called depressed on the strength of a tie",
    keeping the positive class a positive assertion. globem_preprocessing._window_weekly_label
    and pid_summary_label use this same function so the split label, the probe's subject-level
    ground truth and the baselines cannot disagree on a tied participant.

    In endpoint (non-weekly) mode every window of a participant shares one label, so the
    majority IS that label and this is a no-op."""
    m = float(np.asarray(labels).mean())
    return 1 if m > 0.5 else 0          # explicit: ties (m == 0.5) -> 0


def stratified_pid_holdout(unique_pids, pid_label, frac, seed):
    """Split participant ids into (rest, held) at the participant level."""
    pids = sorted(unique_pids)
    if len(pids) < 2 or frac <= 0:
        return set(pids), set()
    y = [pid_label.get(p, 0) for p in pids]
    n_held = min(max(1, int(round(len(pids) * frac))), len(pids) - 1)
    try:
        rest, held = train_test_split(pids, test_size=n_held, stratify=y, random_state=seed)
    except ValueError:
        rng = np.random.default_rng(seed)
        perm = list(rng.permutation(np.array(pids)))
        held, rest = perm[:n_held], perm[n_held:]
    return set(rest), set(held)
