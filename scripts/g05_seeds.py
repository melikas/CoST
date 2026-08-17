"""G0.5 at n=6 -- E1.3 (chronobiological markers): CoST vs RANDOM-INIT, all seeds.

The seed-42 gate ran on ONE random draw. `random_init_model` seeds on ctx.seed, so a
single seed is a single realization of a random encoder, and random projections carry
their own variance. This script pairs the two arms across every seed instead.

It needs NO retraining and NO encoder.pt. The random-init arm depends only on
(config, data, seed); the TRAINED values for all seeds are already stored in each
`rq1/rq1.json` under `axis_probe` from the original sweep. So the comparison is a
paired read of existing results plus one random-init encode per seed -- which is why
this works for the five seeds whose encoders run.sh deleted (KEEP_ENC=1 on SEEDS[0]).

Writes results_hrd/<run>/g05_seeds.json and prints the paired verdict. Never touches
any rq1.json.

    python scripts/g05_seeds.py --run-dir results_hrd/19937323 --cache-dir $SLURM_TMPDIR/hrd_cache
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tasks._experiment_common import _build_model, _dataset, encode
from baselines.cosinor import paper_cosinor_features
from tasks.rhythm import rhythm_axis_probe


def one_variant(vdir, cache_dir, device):
    """Random-init E1.3 for one (variant, seed), paired against its stored trained run."""
    meta = json.loads((vdir / "metrics.json").read_text(encoding="utf-8"))
    cfg, seed = meta["config"], int(meta["config"]["seed"])
    prev = json.loads((vdir / "rq1" / "rq1.json").read_text(encoding="utf-8"))
    trained = prev.get("axis_probe")
    if not isinstance(trained, dict) or trained.get("status") != "OK":
        return None, f"no usable axis_probe in {vdir.name}/rq1/rq1.json"

    data = _dataset(cfg, cache_dir)
    X, pids = data["X"], np.asarray(data["pids"]).astype(str)
    test_mask = np.isin(pids, list(map(str, meta["test_pids"])))
    Xs = X[:, :, :data["n_sensors"]]

    # Same construction as tasks/_experiment_common.random_init_model: seeded, frozen.
    torch.manual_seed(seed)
    rnd = _build_model(cfg, X, data["n_sensors"], device)
    rnd.net.eval()

    # Cache hit: the cosinor fit for THIS seed's test set was written by the sweep.
    cf = paper_cosinor_features(Xs, cfg["bin_minutes"], need_mask=test_mask,
                                window_ids=data.get("window_ids"), pids=pids,
                                cache_path=vdir / "rq1" / "cosinor_cache.npz")
    ri = rhythm_axis_probe(encode(rnd, X, cfg), Xs, test_mask, pids, cfg["bin_minutes"],
                           vdir / "rq1", seed, cf, data["n_sensors"],
                           table_tag="rq1_randinit_seedcheck",
                           sensor_cols=data.get("sensor_cols"))
    if not ri:
        return None, f"random-init probe empty for {vdir.name}"

    # Markers carrying `gain_over_raw` are exactly the latent rows -- raw-PCA rows and the
    # cross-channel aggregate are excluded, matching experiment_q1.py's headline.
    keys = [k for k, v in trained.items()
            if isinstance(v, dict) and "gain_over_raw" in v and k in ri]
    return {"seed": seed, "variant": cfg["backbone"] + "/" + cfg["pe"],
            "delta": {k: trained[k]["value"] - ri[k]["value"] for k in keys},
            "trained": {k: trained[k]["value"] for k in keys},
            "random_init": {k: ri[k]["value"] for k in keys}}, None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="e.g. results_hrd/19937323")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--gpu", type=int, default=0)
    a = p.parse_args()

    device = torch.device(f"cuda:{a.gpu}" if a.gpu >= 0 and torch.cuda.is_available() else "cpu")
    run = Path(a.run_dir)
    rows, skipped = [], []
    for vdir in sorted(d for d in run.iterdir() if d.is_dir() and (d / "metrics.json").exists()):
        r, err = one_variant(vdir, a.cache_dir, device)
        (skipped.append(err) if r is None else rows.append(r))
        print(f"[g05] {vdir.name}: {'ok' if r else 'SKIPPED -- ' + err}", flush=True)

    out = {"run": run.name, "rows": rows, "skipped": skipped, "by_variant": {}}
    for var in sorted({r["variant"] for r in rows}):
        sub = [r for r in rows if r["variant"] == var]
        markers = sorted(set.intersection(*[set(r["delta"]) for r in sub]))
        print(f"\n================ {var}   (n={len(sub)} seeds) ================")
        print(f"  {'marker':34s} {'CoST':>16s} {'random-init':>16s} {'delta':>16s}   p")
        summary = {}
        for k in markers:
            t = np.array([r["trained"][k] for r in sub])
            g = np.array([r["random_init"][k] for r in sub])
            d = t - g
            # n<6 has a Wilcoxon floor above 0.05; report it, do not hide it.
            pv = float(wilcoxon(d).pvalue) if len(d) >= 6 and np.any(d != 0) else float("nan")
            print(f"  {k:34s} {t.mean():+8.3f}±{t.std(ddof=1):.3f} "
                  f"{g.mean():+8.3f}±{g.std(ddof=1):.3f} "
                  f"{d.mean():+8.3f}±{d.std(ddof=1):.3f}   {pv:.4f}")
            summary[k] = {"trained_mean": float(t.mean()), "random_init_mean": float(g.mean()),
                          "delta_mean": float(d.mean()), "delta_sd": float(d.std(ddof=1)),
                          "wilcoxon_p": pv, "n_seeds": len(sub)}
        alld = np.array([v for r in sub for v in r["delta"].values()])
        frac = float(np.mean(alld > 0))
        print(f"  ---- markers won by CoST: {frac:.3f}  "
              f"({int((alld > 0).sum())}/{len(alld)} marker x seed cells) | "
              f"median delta {np.median(alld):+.4f}")
        print(f"  >>> {'PASS' if frac >= 0.625 and np.median(alld) > 0 else 'FAIL'} "
              f"(PASS needs frac >= 0.625 AND median delta > 0)")
        out["by_variant"][var] = {"markers": summary, "frac_won_by_cost": frac,
                                  "median_delta": float(np.median(alld)), "n_seeds": len(sub)}

    (run / "g05_seeds.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n[saved] {run / 'g05_seeds.json'}")


if __name__ == "__main__":
    main()
