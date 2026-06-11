"""Smoke test: every (backbone, pe) variant builds, pretrains and encodes.

Runs on CPU with tiny random data so it finishes in seconds. For each variant it
does a forward+backward pretraining step (the CoST MoCo loss) and an `encode`
call, asserting the representation shape. This verifies imports, parameter
shapes and forward passes without needing the real HRD CSV or a GPU.

Run (from the repo root):  python scripts/test_pe_variants.py
"""
import os
import sys

import numpy as np
import torch

# allow running as `python scripts/test_pe_variants.py` from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cost import CoST
from models.positional_encoding import ABSOLUTE_PES, ATTENTION_PES

# tiny problem: N windows, T steps, C channels
N, T, C = 8, 32, 4
REPR_DIMS, HIDDEN_DIMS, DEPTH = 32, 16, 2
KERNELS = [1, 2, 4]

TRANSFORMER_PES = list(ABSOLUTE_PES) + list(ATTENTION_PES)  # 8 PEs + time2vec
VARIANTS = (
    [("transformer", pe) for pe in TRANSFORMER_PES]
    + [("tcn", "none"), ("tcn", "time2vec")]
)


def run_variant(backbone, pe):
    torch.manual_seed(0)
    np.random.seed(0)
    data = np.random.randn(N, T, C).astype(np.float32)
    model = CoST(
        input_dims=C, kernels=KERNELS, alpha=5e-4, max_train_length=T,
        output_dims=REPR_DIMS, hidden_dims=HIDDEN_DIMS, depth=DEPTH,
        backbone=backbone, pe=pe, device="cpu", lr=1e-3, batch_size=N,
    )
    # one pretraining step (forward + backward through the MoCo loss)
    loss_log = model.fit(data, n_iters=2, verbose=False)
    # encode -> (N, repr_dims)
    reprs = model.encode(data, mode="forecasting").squeeze(1)
    assert reprs.shape == (N, REPR_DIMS), reprs.shape
    assert np.isfinite(reprs).all(), "non-finite representation"
    return float(loss_log[-1]) if loss_log else float("nan")


def main():
    print(f"{'backbone':<12} {'pe':<11} {'last_loss':>10}   status")
    print("-" * 48)
    failures = []
    for backbone, pe in VARIANTS:
        try:
            loss = run_variant(backbone, pe)
            print(f"{backbone:<12} {pe:<11} {loss:>10.4f}   OK")
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"{backbone:<12} {pe:<11} {'-':>10}   FAIL: {exc}")
            failures.append((backbone, pe, repr(exc)))
    print("-" * 48)
    if failures:
        print(f"{len(failures)} variant(s) FAILED:")
        for b, pe, exc in failures:
            print(f"  {b}/{pe}: {exc}")
        raise SystemExit(1)
    print(f"All {len(VARIANTS)} variants passed.")


if __name__ == "__main__":
    main()
