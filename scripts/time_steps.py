"""Seconds per training update of the configured DSSL and of the CoST reference adapter on
this device, and the projected time of the configured budget. slurm/smoke.sbatch runs it on
the GPU type the study uses; compare the projection with the jobs' --time limit.

    python scripts/time_steps.py [--dataset hrd globem] [--device cuda] [--updates 20]

Training updates only: checkpoint writes, pretext-loss monitoring, encoding and evaluation
are not included.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from cost import DSSL, REFERENCE_SHARED
from datautils import load_npz


def elapsed(kwargs, X, updates):
    model = DSSL(**kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    model.fit(X, n_iters=updates, log_every=updates, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', nargs='+', choices=['hrd', 'globem'], default=['hrd', 'globem'])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--updates', type=int, default=20, help='timed updates, after 3 warm-up updates')
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)        # the study's settings
    torch.backends.cudnn.benchmark = False
    if args.device == 'cuda':
        print('device:', torch.cuda.get_device_name(0), flush=True)
    for dataset in args.dataset:
        cfg = json.loads((ROOT / 'configs' / f'{dataset}.json').read_text())
        c = load_npz(ROOT / cfg['cache'])
        X = c.X[:256].astype('float32')
        base = dict(input_dims=c.n_sensors, seq_len=c.seq_len, bins_per_day=c.bins_per_day,
                    device=args.device, model_seed=1)
        shared = {k: cfg['model'][k] for k in REFERENCE_SHARED if k in cfg['model']}
        for label, kwargs in (('dssl', dict(base, method='dssl', **cfg['model'])),
                              ('cost_reference', dict(base, method='cost_reference', **shared))):
            per = (elapsed(kwargs, X, 3 + args.updates) - elapsed(kwargs, X, 3)) / args.updates
            print(f"{dataset} {label}: {per:.3f} s/update -> {per * cfg['steps'] / 3600:.2f} h "
                  f"for {cfg['steps']} updates (training only)", flush=True)


if __name__ == '__main__':
    main()
