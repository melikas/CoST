"""RQ2 -> RQ3 bridge on a trained run's stored representations: T1 (RQ3-state) and T2
(circular readout). Pre-registered 2026-09-13. Nothing is retrained; RQ1 and RQ2 are untouched.

    python scripts/rq3_bridge.py --run results_eq_h --npz hrd_2224103.npz \
        --energy ee_windows.npz --out results_eq_h_bridge --device cuda [--only r0f0]
    python scripts/rq3_bridge.py --run results_eq_h --out results_eq_h_bridge --report

T1  RQ3-state: does the representation track within-person affect the way it tracks rhythm?
    Held-out labelled participants of every fold. d_t = week t's standardised distance from
    the same person's 4 preceding contiguous weeks (eval.personal_baseline / eval.dscore,
    RQ2's own reference: no fitting, no labels). Ground truth m_t = |E_t - mean E over the
    same 4 weeks|, E = weekly emotional energy (mean of the daily 1-5 ratings). C_mood is the
    within-person concordance of d with m over every pair of a participant's scored weeks
    (ties in d count 1/2), pooled over the fold; null 0.5. C_drop is the same against the
    signed energy drop. Trained vs untrained by the Nadeau-Bengio corrected paired test over
    the folds; WIN = that interval excludes 0 on the positive side.

T2  Endpoint depression through the circular readout [trend | amp | cos | sin], computed
    exactly from the stored amplitude and angle blocks for DSSL and Random-init, with
    isotropic pair scaling, through eval.probe_linear (the frozen protocol of record).
    WIN = combined DeLong p < 0.05 and positive against BOTH Random-init (circular) and the
    raw window. The angle rungs are scored alongside and must reproduce the run's own
    primary table.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval as E                                    # noqa: E402
from cv import paired_test                          # noqa: E402
from data_loader import load_npz                    # noqa: E402

PARTS = ("trend", "amp", "phase")


def concordance(d, m, pids):
    """Within-person pairs with m_i != m_j: 1 when d orders them as m does, 1/2 for a tie."""
    num = den = 0.0
    for p in np.unique(pids):
        i = np.flatnonzero(pids == p)
        dd = d[i][:, None] - d[i][None]
        dm = m[i][:, None] - m[i][None]
        u = np.triu(np.ones((len(i), len(i)), bool), 1) & (dm != 0)
        num += ((dd * dm > 0) + 0.5 * (dd == 0))[u].sum()
        den += u.sum()
    return (float(num / den) if den else float("nan")), int(den)


def t1(reps, coh, fold, tdays, ee):
    te = np.isin(coh.pids, list(fold.test_pids))
    span = float(7 * (E.BASELINE_R - 1))
    ref_e, _, ok_e = E.personal_baseline(ee[:, None], coh.pids, E.BASELINE_R, tdays, span)
    ref_e = ref_e[:, 0]
    out = {}
    for name, V in reps.items():
        mu, sd, ok = E.personal_baseline(V, coh.pids, E.BASELINE_R, tdays, span)
        d = E.dscore(V, mu, sd)
        s = te & ok & ok_e & np.isfinite(ee) & np.isfinite(ref_e) & np.isfinite(d)
        c, n = concordance(d[s], np.abs(ee - ref_e)[s], coh.pids[s])
        c_drop, _ = concordance(d[s], (ref_e - ee)[s], coh.pids[s])
        out[name] = {"C_mood": c, "C_drop": c_drop, "n_pairs": n, "n_weeks": int(s.sum()),
                     "n_participants": len(set(coh.pids[s]))}
    return out


def t2(parts, flat, coh, fold, y_by_pid):
    fd = coh.fold_data(fold)
    tr, te = fd.probe_train, fd.probe_test
    rq3 = {"_labels": {str(p): y_by_pid[p] for p in fold.test_pids}}
    rungs = {"Raw window": (flat, None, 0)}
    for name, pt in parts.items():
        a, ph = pt["amp"], pt["phase"]
        if a.shape != ph.shape:
            raise ValueError(f"{name}: T2 needs the angle readout (one phase column per amplitude)")
        circ = name.replace("angle", "circular") if "angle" in name else f"{name} (circular)"
        rungs[name if "angle" in name else f"{name} (angle)"] = (
            np.concatenate([pt[k] for k in PARTS], axis=1), None, 0)
        rungs[circ] = (np.concatenate([pt["trend"], a, np.cos(ph), np.sin(ph)], axis=1),
                       pt["trend"].shape[1] + a.shape[1], a.shape[1])
    for name, (V, ps, pw) in rungs.items():
        rq3[name + E.LINEAR] = E.probe_linear(V[tr], coh.y[tr], V[te], coh.pids[te], coh.y[te],
                                              fold.probe_seed, groups=coh.pids[tr],
                                              pair_start=ps, pair_width=pw)
    return rq3


def report(out, plan):
    recs = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("r*f*.json"))}
    if not recs:
        raise SystemExit(f"no fold results under {out}")
    nf, nr = plan["protocol"]["n_folds"], plan["protocol"]["n_repeats"]
    names = list(next(iter(recs.values()))["t1"])
    print(f"{len(recs)} folds\n\nT1 -- RQ3-state: within-person concordance of the representation's "
          "distance from its 4-week\npersonal baseline with the week's emotional-energy change "
          "(held-out participants; null 0.5)")
    for key, what in (("C_mood", "|energy change|"), ("C_drop", "signed energy drop")):
        series = {n: np.array([recs[t]["t1"][n][key] for t in recs]) for n in names}
        print(f"\n{key}: against {what}")
        print(E._table([[n, E._fmt(np.nanmean(v)), E._fmt(np.nanstd(v, ddof=1)),
                         int(np.isfinite(v).sum())] for n, v in series.items()],
                       ["representation", f"mean {key}", "SD", "folds"]))
        rows = []
        for n in (n for n in names if n.startswith("DSSL")):
            for ref in ("Random-init", "Raw window"):
                m = np.isfinite(series[n]) & np.isfinite(series[ref])
                if m.sum() >= 2:
                    rows.append([n, f"vs {ref}"] + E._contrast(
                        paired_test(series[n][m], series[ref][m], nf, nr)))
        if rows:
            print(E._table(rows, ["arm", "against"] + E.CONTRAST_COLS))
    pairs = [recs[t]["t1"][names[0]] for t in recs]
    print(f"\n  per fold: median {int(np.median([p['n_participants'] for p in pairs]))} participants, "
          f"{int(np.median([p['n_weeks'] for p in pairs]))} scored weeks, "
          f"{int(np.median([p['n_pairs'] for p in pairs]))} week pairs")
    print("\nT2 -- endpoint depression, circular readout, frozen linear protocol "
          "(angle rungs = reproduction check)\n")
    print(E.linear_report({t: {"fold": t, "rq3": r["rq3"]} for t, r in recs.items()}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--npz", default="hrd_2224103.npz")
    ap.add_argument("--energy", default="ee_windows.npz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default=None, help="comma-separated fold tags, e.g. r0f0")
    ap.add_argument("--report", action="store_true", help="only print the tables")
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    if a.report:
        return report(out, plan)

    coh = load_npz(a.npz)
    en = np.load(a.energy)
    if not np.array_equal(np.asarray(coh.window_ids).astype(str), en["window_ids"].astype(str)):
        raise SystemExit("the energy file is not aligned to the cohort's window_ids")
    ee = en["ee"].astype(float)
    tdays = E.window_start_days(coh.window_ids)
    flat = np.nan_to_num(coh.X[:, :, :coh.n_sensors].reshape(len(coh.X), -1), nan=0.0)
    y_by_pid = {p: int(round(float(coh.y[coh.pids == p].mean()))) for p in np.unique(coh.pids)}
    arms = [f"{x['phase_readout']}_{x['weights']}" for x in plan["arms"]
            if x["phase_readout"] == "angle"]
    tags = (a.only.split(",") if a.only else
            [f"r{f['repeat']}f{f['fold']}" for f in plan["folds"]])
    for tag in tags:
        dest = out / f"{tag}.json"
        if dest.exists():
            continue
        fold = E.fold_from_plan(plan, tag)
        ri = E._blank_model(coh, plan, "angle", a.device, seed=fold.model_seed).encode(
            coh.X, parts=True)
        parts = {"Random-init": {k: ri[k] for k in PARTS}}
        for arm in arms:
            z = np.load(run / arm / tag / "repr.npz")
            parts[f"DSSL {arm}"] = {k: z[k] for k in PARTS}
            if not np.array_equal(z["full"], np.concatenate([z[k] for k in PARTS], axis=1)):
                raise AssertionError(f"{arm}/{tag}: full != [trend | amp | phase]")
        full = {n: np.concatenate([pt[k] for k in PARTS], axis=1) for n, pt in parts.items()}
        rec = {"fold": tag, "t1": t1({**full, "Raw window": flat}, coh, fold, tdays, ee),
               "rq3": t2(parts, flat, coh, fold, y_by_pid)}
        dest.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(f"{tag} done: " + ", ".join(f"{n} C_mood={v['C_mood']:.3f}"
                                          for n, v in rec["t1"].items()), flush=True)
    if not a.only:
        report(out, plan)


if __name__ == "__main__":
    main()
