"""Where every generated artifact lives, and which research question owns it.

One table, one rule: a file's name determines its folder, so the same artifact cannot be
written to two places and no folder can accumulate files nobody claims. Adding an output means
adding a row here first -- if it does not answer a question in
`docs/RQ_Minimal_Experiment_Design.md`, it does not get written.

    <variant_dir>/
        metrics.json, report.md, encoder.pt, plain_encoder.pt   run-level, no single RQ
        pretrain/            SSL training diagnostics -- evidence the encoder converged,
                             not an answer to any RQ
        RQ1/                 does the representation encode cyclic and trend constructs?
        RQ2/                 can unlabeled personal baselines detect within-person deviation?
        RQ3/                 utility and limits on the depression endpoint
        RQ0_descriptive/     documented in METHODOLOGY.md sec.6 but claimed by no RQ in the
                             design document. Kept, and quarantined: nothing in RQ1-RQ3 reads
                             from here, and no statistical claim may rest on it.

`hrd_rhythm_separability_*` sits in RQ3 because the design document's mapping table makes the
depression endpoint an RQ3 question; it is produced by `train_hrd.py` only because the encoder
is already in memory there, which is a scheduling fact, not an ownership one.
"""
from pathlib import Path

# (filename prefix, folder). Longest prefix wins, so "cosinor_cache_all" beats
# "cosinor_cache". A prefix with no folder ("") stays at the variant root.
_OWNER = [
    # ---- run level -----------------------------------------------------------------
    ("metrics.json", ""),
    ("report.md", ""),
    ("encoder.pt", ""),
    ("plain_encoder", ""),
    # ---- pretraining diagnostics ---------------------------------------------------
    ("pretrain_loss", "pretrain"),
    ("val_loss", "pretrain"),
    ("loss_iters", "pretrain"),
    ("gradnorm_weights", "pretrain"),
    # ---- RQ1: E1.2 recovery, E1.3 chronobiology tether, E1.5 controls --------------
    ("decomposition_recovery", "RQ1"),
    ("rhythm_axis_probe", "RQ1"),
    ("rq1", "RQ1"),
    ("position_geometry", "RQ1"),   # E1.4: does the temporal frame organise V?
    ("cosinor_cache.npz", "RQ1"),
    # ---- RQ2: within-person deviation ----------------------------------------------
    ("rq2", "RQ2"),
    # ---- RQ3: utility, ablation, limits, and the endpoint separability table -------
    ("rq3", "RQ3"),
    ("hrd_rhythm_separability", "RQ3"),
    ("hrd_rhythm.", "RQ3"),
    ("cosinor_cache_all.npz", "RQ3"),
    ("paper_cosinor", "RQ3"),          # the paper-cosinor view is a separability rung
    # Does the trained encoder gain more from opening the readout than its untrained
    # control does? It is scored as a utility ladder at the benchmark window unit, so
    # it sits with RQ3 rather than with the RQ1 recovery diagnostics.
    ("readout_interaction", "RQ3"),
    # RQ1 split into the amplitude and phase halves of the seasonal component, which
    # three other measurements say move in opposite directions under pretraining.
    ("block_recovery", "RQ1"),
    # ---- descriptive, owned by no RQ -----------------------------------------------
    ("hrd_tsne", "RQ0_descriptive"),
    ("hrd_umap", "RQ0_descriptive"),
    ("frequency_", "RQ0_descriptive"),
    ("circadian_similarity", "RQ0_descriptive"),
    ("participant_trajectory", "RQ0_descriptive"),
    ("signal_embedding", "RQ0_descriptive"),
]

RQ_DIRS = ("RQ1", "RQ2", "RQ3", "RQ0_descriptive", "pretrain")


def owner(name):
    """The folder that owns `name`, or "" for the variant root. Raises on an unknown name:
    an artifact with no owner is exactly what this module exists to prevent."""
    stem = Path(name).name
    hit = max((p for p, _ in _OWNER if stem.startswith(p)), key=len, default=None)
    if hit is None:
        raise KeyError(
            f"{stem!r} has no owning research question. Add it to rq_paths._OWNER, or -- if it "
            f"answers no question in docs/RQ_Minimal_Experiment_Design.md -- do not write it.")
    return dict(_OWNER)[hit]


def rq_path(variant_dir, name, create=True):
    """Absolute path for `name`, with its owning folder created.

    Readers pass `create=False`: resolving a path to check whether a run produced something
    must not leave an empty folder behind in that run -- collecting results from a finished
    sweep was silently adding directories to it."""
    d = Path(variant_dir) / owner(name)
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d / Path(name).name
