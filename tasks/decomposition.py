"""Decomposition Recovery Score (DRS): how well the frozen CoST encoder's
trend / seasonal latents recover the trend and seasonality of the input channels.

Read-only, post-hoc analysis (the model is never modified or trained here). For a
reference, each input sensor channel is decomposed per window by closed-form
harmonic regression -- a low-order polynomial trend tau_c(t) plus K circadian
harmonics sigma_c(t) (cosinor with a smooth trend). A linear ridge probe then
asks how much of each channel's reference component is linearly decodable from the
encoder's d-dim latent (fit on train windows, scored on held-out test windows):

    HEADLINE -- recovery from the FULL representation (assumes no clean split):
    rho^F->T_c = R2( [V^(T);V^(S)] -> tau_c )    trend captured by the rep   (want high)
    rho^F->S_c = R2( [V^(T);V^(S)] -> sigma_c )  rhythm captured by the rep  (want high)

    DIAGNOSTIC -- how the two branches split that information:
    rho^T_c    = R2( V^(T) -> tau_c )     own-branch trend recovery
    rho^S_c    = R2( V^(S) -> sigma_c )   own-branch rhythm recovery
    rho^S->T_c = R2( V^(S) -> tau_c )     trend leaking into the SFD branch
    rho^T->S_c = R2( V^(T) -> sigma_c )   rhythm leaking into the TFD branch

Per-channel scores are aggregated with variance weights (channels with no real trend /
rhythm get ~0 weight). The HEADLINE numbers rec_full_trend / rec_full_rhythm answer the
question that matters -- "does the representation encode the trend / the 24 h rhythm at
all?" -- without depending on a trend/seasonal split that does not actually hold. The
branch recovery + leak and the net gap DIS = 0.5*((rec_T-leak_T)+(rec_S-leak_S)) are kept
only as a secondary disentanglement diagnostic (DIS < 0 = the branches are entangled). We
do NOT collapse them into one score: the old geomean index was degenerate (0 whenever
DIS <= 0) and uncorrelated with downstream prediction.

Called from train_hrd.py with the in-memory model/data; emits, per variant:
  * decomposition_recovery.{csv,md}  - per-channel table (Full recovery + branch + leak)
  * decomposition_recovery.png       - per-channel Full recovery vs own-branch vs leak
  * decomposition_recovery.json      - aggregates (rec_full_*, branch recovery, leak, DIS)
"""
import json
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
REC_FULL_COLOR   = "#0072B2"                     # headline: recovery from Full (blue)
REC_BRANCH_COLOR = "#56B4E9"                     # diagnostic: own-branch recovery (sky)
LEAK_COLOR       = "#D55E00"                      # diagnostic: cross-branch leak (vermillion)

# Ridge penalty grid + selection rule follow the original CoST evaluation protocol
# (salesforce/CoST, tasks/_eval_protocols.py::fit_ridge), so the probe strength is never a
# hand-set constant: fit on train for every lambda, pick the one that minimises the validation
# error, and score the test set with that single lambda* only.
#
# The paper's 13 values (0.1 ... 1000) are kept as a SUBSET but the grid is extended at both
# ends, because sklearn's `alpha` is not normalised by the sample count: after StandardScaler
# the Gram diagonal is ~n, so the penalty only bites once alpha is comparable to n. A DRS probe
# here sees ~2.5e5 rows (windows x subsampled timesteps), which makes the paper's ceiling of
# 1000 effectively alpha->0 -- and indeed run 19606825 pinned lambda* at 1000 for Full->sigma
# and at the 0.1 floor for the branch terms, i.e. both ends were truncating. Sweeping a longer
# grid is nearly free (see _ridge_fit), and `ridge_alpha_at_bound` in the JSON reports whether
# a selection still landed on an endpoint.
RIDGE_ALPHAS = (1e-3, 1e-2, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000,
                2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 5e5, 1e6, 1e7)
_GRAM_CHUNK = 20_000                             # rows per pass when accumulating F^T F


def harmonic_reference(Xs, period_bins, trend_degree=3, n_harmonics=3):
    """Per-window, per-channel reference: x ~ [poly trend] + [K circadian harmonics].

    Closed-form least squares with one fixed design matrix M (t is 0..T-1 in every
    window). Returns the fitted trend tau and seasonal sigma, each (N, T, C).
    """
    N, T, C = Xs.shape
    t = np.arange(T)
    tn = t / max(T, 1)                                       # normalise poly basis
    w = 2.0 * np.pi / period_bins
    M_tau = np.stack([tn ** p for p in range(trend_degree + 1)], axis=1)     # (T, deg+1)
    M_sig = np.stack([f(w * k * t) for k in range(1, n_harmonics + 1)
                      for f in (np.cos, np.sin)], axis=1)                    # (T, 2K)
    M = np.concatenate([M_tau, M_sig], axis=1).astype(np.float32)
    pinv = np.linalg.pinv(M)                                                 # (P, T)

    Xflat = np.nan_to_num(Xs, nan=0.0).transpose(1, 0, 2).reshape(T, N * C)  # (T, N*C)
    beta = pinv @ Xflat                                                      # (P, N*C)
    p_tau = M_tau.shape[1]
    tau = (M_tau @ beta[:p_tau]).reshape(T, N, C).transpose(1, 0, 2)         # (N, T, C)
    sig = (M_sig @ beta[p_tau:]).reshape(T, N, C).transpose(1, 0, 2)
    return tau.astype(np.float32), sig.astype(np.float32)


def extract_components(model, X, batch_size=128, time_stride=4):
    """Frozen encoder trend / seasonal sequences V^(T), V^(S), each (N, T', d),
    subsampled every `time_stride` steps (the components are smooth)."""
    idx_t = np.arange(0, X.shape[1], time_stride)
    org = model.net.training
    model.net.eval()
    VT, VS = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).float().to(model.device)
            trend, season = model.net(xb)                    # (b, T, d) each
            VT.append(trend[:, idx_t].cpu().numpy())
            VS.append(season[:, idx_t].cpu().numpy())
    model.net.train(org)
    return np.concatenate(VT), np.concatenate(VS), idx_t


def _ridge_fit(Ftr, Ytr):
    """Standardise + centre on the TRAIN rows, then return ``predict(alpha, F)``.

    Equivalent to make_pipeline(StandardScaler(), Ridge(alpha)).fit(Ftr, Ytr).predict(F), but
    the O(n*p^2) Gram and its eigendecomposition are paid ONCE and reused for every penalty,
    since W(a) = Q diag(1/(lam+a)) Q^T F~^T Y~ with F~ standardised and Y~ centred on the train
    statistics. Sweeping the whole grid therefore costs little more than a single fit.

    Returns a closure rather than a stacked (n_alphas, n_rows, C) array: at ~2.5e5 rows that
    array is ~100 MB per 13 penalties, so materialising a long grid would dominate the memory
    of the whole analysis. One penalty at a time keeps it at O(n_rows * C).
    """
    mu = Ftr.mean(axis=0, dtype=np.float64)
    sd = Ftr.std(axis=0, dtype=np.float64)
    sd[sd == 0] = 1.0                                        # dead latent dims -> no scaling
    ym = Ytr.mean(axis=0, dtype=np.float64)

    p, C = Ftr.shape[1], Ytr.shape[1]
    G, b = np.zeros((p, p)), np.zeros((p, C))
    for i in range(0, len(Ftr), _GRAM_CHUNK):                # chunked: F is float32 and tall
        Z = (Ftr[i:i + _GRAM_CHUNK].astype(np.float64) - mu) / sd
        G += Z.T @ Z
        b += Z.T @ (Ytr[i:i + _GRAM_CHUNK].astype(np.float64) - ym)
    lam, Q = np.linalg.eigh(G)
    Qb = Q.T @ b

    def predict(alpha, F):
        W = Q @ (Qb / (lam + alpha)[:, None])                # (p, C) -- negligible per penalty
        out = np.empty((len(F), C))
        for i in range(0, len(F), _GRAM_CHUNK):
            Z = (F[i:i + _GRAM_CHUNK].astype(np.float64) - mu) / sd
            out[i:i + _GRAM_CHUNK] = Z @ W + ym
        return out

    return predict


def _probe_r2(feat, target, fit_mask, sel_mask, test_mask, alphas):
    """Per-channel held-out R2 of a ridge probe feat -> target, with the penalty chosen on a
    held-out selection (validation) set -- never on train and never on test.

    feat (N,T',d), target (N,T',C). Fit on `fit_mask` windows for every penalty in `alphas`,
    score each on `sel_mask` with the CoST criterion (RMSE + MAE, pooled over channels), then
    report the R2 of the single winning penalty on `test_mask`.

    Returns (r2 (C,), lambda*).
    """
    def flat(a, m):
        s = a[m]
        return s.reshape(-1, s.shape[-1])
    Ftr, Ytr = flat(feat, fit_mask), flat(target, fit_mask)
    Fse, Yse = flat(feat, sel_mask), flat(target, sel_mask)
    Fte, Yte = flat(feat, test_mask), flat(target, test_mask)

    alphas = tuple(alphas)
    predict = _ridge_fit(Ftr, Ytr)
    err = []
    for a in alphas:                                         # CoST criterion: RMSE + MAE
        r = predict(a, Fse) - Yse
        err.append(np.sqrt((r ** 2).mean()) + np.abs(r).mean())
    k = int(np.argmin(err))                                  # lambda* = argmin validation error

    pred = predict(alphas[k], Fte)
    ss_res = ((Yte - pred) ** 2).sum(axis=0)
    ss_tot = ((Yte - Yte.mean(axis=0)) ** 2).sum(axis=0)
    r2 = np.where(ss_tot > 0, 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, 1.0), 0.0)
    return np.clip(r2, 0.0, 1.0), float(alphas[k])           # (C,), lambda*


def _selection_split(train_mask, val_mask, pids, seed):
    """Resolve the (fit, selection) window masks used for the penalty sweep.

    Prefers the caller's participant-disjoint validation mask. train_hrd.py falls back to
    `val_mask is train_mask` when no validation pids exist, and selecting a penalty on the
    same windows it was fit on always returns the smallest one -- so in that case carve a
    participant-disjoint quarter out of train instead. Returns (fit, sel, description).
    """
    if val_mask is not None:
        sel = val_mask & ~train_mask
        if sel.any():
            return train_mask, sel, "caller val_mask (participant-disjoint)"

    idx = np.flatnonzero(train_mask)
    if pids is not None:
        held = np.unique(pids[idx])
        if len(held) > 1:
            rng = np.random.RandomState(seed)
            rng.shuffle(held)
            n = max(1, int(round(0.25 * len(held))))
            sel = np.zeros_like(train_mask)
            sel[idx] = np.isin(pids[idx], held[:n])
            fit = train_mask & ~sel
            if fit.any() and sel.any():
                return fit, sel, f"carved from train ({n}/{len(held)} pids held out)"

    if len(idx) > 1:                                         # last resort: split the windows
        n = max(1, int(round(0.25 * len(idx))))
        sel = np.zeros_like(train_mask)
        sel[idx[:n]] = True
        return train_mask & ~sel, sel, "carved from train (windows, pids unavailable)"

    return train_mask, train_mask, "DEGENERATE: selected on train (too few train windows)"


def _var_weights(ref, mask):
    """Variance weights over channels: mean over windows of within-window variance."""
    v = ref[mask].var(axis=1).mean(axis=0)                   # (C,)
    s = v.sum()
    return v / s if s > 0 else np.full_like(v, 1.0 / len(v))


def _save_table(names, wT, rFullT, rT, rST, wS, rFullS, rS, rTS, agg, variant_dir, tag):
    cols = ["Channel", "w_T", "Full->tau", "V^T->tau", "leak V^S->tau",
            "w_S", "Full->sig", "V^S->sig", "leak V^T->sig"]
    rows = [[names[c], f"{wT[c]:.3f}", f"{rFullT[c]:.3f}", f"{rT[c]:.3f}", f"{rST[c]:.3f}",
             f"{wS[c]:.3f}", f"{rFullS[c]:.3f}", f"{rS[c]:.3f}", f"{rTS[c]:.3f}"]
            for c in range(len(names))]
    foot = (f"HEADLINE recovery from Full:  tau={agg['rec_full_trend']:.3f}  "
            f"sigma={agg['rec_full_rhythm']:.3f}   |   diagnostic (branch / leak):  "
            f"rec_T={agg['rec_trend_branch']:.3f} rec_S={agg['rec_rhythm_branch']:.3f}  "
            f"leak_T={agg['leak_into_trend']:.3f} leak_S={agg['leak_into_rhythm']:.3f}  "
            f"DIS={agg['DIS']:.3f}   ({tag})")
    lam = agg["ridge_alpha"]
    note = ("Ridge penalty selected on a held-out set -- "
            f"{agg['ridge_alpha_selection']}, {agg['n_alpha_select_windows']} windows -- "
            f"from the CoST grid {agg['ridge_alpha_grid']}. "
            f"lambda*: Full->tau={lam['rec_full_trend']:g} Full->sig={lam['rec_full_rhythm']:g} "
            f"V^T->tau={lam['rec_trend_branch']:g} V^S->sig={lam['rec_rhythm_branch']:g} "
            f"leak V^S->tau={lam['leak_into_trend']:g} leak V^T->sig={lam['leak_into_rhythm']:g}")

    csv = (",".join(cols) + "\n" + "\n".join(",".join(r) for r in rows)
           + "\n# " + foot + "\n# " + note)              # one comment line each, no newlines
    (variant_dir / "decomposition_recovery.csv").write_text(csv, encoding="utf-8")
    md = ("| " + " | ".join(cols) + " |\n| " + " | ".join("---" for _ in cols) + " |\n"
          + "\n".join("| " + " | ".join(r) + " |" for r in rows)
          + f"\n\n*{foot}*\n\n*{note}*\n")               # separate italic paragraphs
    (variant_dir / "decomposition_recovery.md").write_text(md, encoding="utf-8")


def _save_figure(names, rFullT, rT, rST, rFullS, rS, rTS, agg, variant_dir, tag):
    x = np.arange(len(names)); w = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(max(9, 0.75 * len(names) + 4), 4.7), sharey=True)
    panels = [(rFullT, rT, rST, "Trend  (target tau)"),
              (rFullS, rS, rTS, "Rhythm  (target sigma)")]
    for ax, (full, branch, leak, title) in zip(axes, panels):
        ax.bar(x - w, full,   w, color=REC_FULL_COLOR,   label="recovery R2 (Full)")
        ax.bar(x,     branch, w, color=REC_BRANCH_COLOR, label="recovery R2 (own branch)")
        ax.bar(x + w, leak,   w, color=LEAK_COLOR,       label="leak R2 (other branch)")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=11); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=7.5, framealpha=0.85)
    axes[0].set_ylabel("held-out R2")
    fig.suptitle(f"Decomposition recovery  -  {tag}\n"
                 f"HEADLINE (Full):  tau R2={agg['rec_full_trend']:.2f}   "
                 f"sigma R2={agg['rec_full_rhythm']:.2f}      "
                 f"diagnostic DIS={agg['DIS']:.2f} ({'entangled' if agg['DIS'] < 0 else 'split'})",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(variant_dir / "decomposition_recovery.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_decomposition_recovery(model, X, train_mask, test_mask, variant_dir,
                               seq_len, bin_minutes, sensor_cols=None, seed=42,
                               batch_size=128, period_hours=24.0, alpha=None,
                               alphas=RIDGE_ALPHAS, val_mask=None, pids=None,
                               time_stride=4):
    """Compute and save the DRS report for one trained variant. Returns the aggregates.

    The ridge penalty is swept over `alphas` and chosen per probe on a held-out selection
    set (see _selection_split), following the original CoST protocol. Passing a float as
    `alpha` pins the penalty to that single value instead (reproduces pre-sweep runs).
    """
    variant_dir = Path(variant_dir)
    tag = f"{model.net.backbone} / {model.net.pe}"
    n_sensors = len(sensor_cols) if sensor_cols is not None else X.shape[2]
    names = list(sensor_cols) if sensor_cols is not None else [f"ch{c}" for c in range(n_sensors)]
    grid = (float(alpha),) if alpha is not None else tuple(alphas)

    period_bins = period_hours * 60.0 / bin_minutes                 # 24h -> 96 bins
    tau, sig = harmonic_reference(X[:, :, :n_sensors], period_bins)
    VT, VS, idx_t = extract_components(model, X, batch_size, time_stride)
    tau, sig = tau[:, idx_t], sig[:, idx_t]                         # align to subsampled steps

    fit_mask, sel_mask, sel_src = _selection_split(train_mask, val_mask, pids, seed)
    print(f"[decomp] ridge lambda grid {list(grid)} | selection set: {sel_src} "
          f"({int(fit_mask.sum())} fit / {int(sel_mask.sum())} select / "
          f"{int(test_mask.sum())} test windows)")

    def probe(feat, target):
        return _probe_r2(feat, target, fit_mask, sel_mask, test_mask, grid)

    VF = np.concatenate([VT, VS], axis=-1)                          # Full representation
    rFullT, aFullT = probe(VF, tau)                                 # HEADLINE: Full -> trend
    rFullS, aFullS = probe(VF, sig)                                 # HEADLINE: Full -> rhythm
    rT, aT   = probe(VT, tau)                                       # branch trend recovery
    rS, aS   = probe(VS, sig)                                       # branch rhythm recovery
    rST, aST = probe(VS, tau)                                       # trend leak into SFD
    rTS, aTS = probe(VT, sig)                                       # rhythm leak into TFD

    wT, wS = _var_weights(tau, test_mask), _var_weights(sig, test_mask)
    DRS_T, DRS_S = float(wT @ rT), float(wS @ rS)
    leak_T, leak_S = float(wT @ rST), float(wS @ rTS)
    DIS = 0.5 * ((DRS_T - leak_T) + (DRS_S - leak_S))               # secondary diagnostic
    agg = {"backbone": model.net.backbone, "pe": model.net.pe,
           # headline: does the representation encode trend / 24h rhythm at all?
           "rec_full_trend": float(wT @ rFullT), "rec_full_rhythm": float(wS @ rFullS),
           # diagnostic: how the branches split it (DIS<0 = entangled)
           "rec_trend_branch": DRS_T, "rec_rhythm_branch": DRS_S,
           "leak_into_trend": leak_T, "leak_into_rhythm": leak_S, "DIS": DIS,
           # probe hyperparameter provenance: which penalty won, and where it was chosen
           "ridge_alpha_grid": list(grid), "ridge_alpha_selection": sel_src,
           "n_alpha_fit_windows": int(fit_mask.sum()),
           "n_alpha_select_windows": int(sel_mask.sum()),
           "n_test_windows": int(test_mask.sum()),
           "ridge_alpha": {"rec_full_trend": aFullT, "rec_full_rhythm": aFullS,
                           "rec_trend_branch": aT, "rec_rhythm_branch": aS,
                           "leak_into_trend": aST, "leak_into_rhythm": aTS},
           # True = at least one probe chose an ENDPOINT of the grid, i.e. it wanted a penalty
           # the grid does not offer and the R2 above is a truncated answer. Extend the grid
           # on that side before reading the number.
           "ridge_alpha_at_bound": bool(len(grid) > 1 and
                                        {aFullT, aFullS, aT, aS, aST, aTS}
                                        & {float(grid[0]), float(grid[-1])})}

    _save_table(names, wT, rFullT, rT, rST, wS, rFullS, rS, rTS, agg, variant_dir, tag)
    _save_figure(names, rFullT, rT, rST, rFullS, rS, rTS, agg, variant_dir, tag)
    (variant_dir / "decomposition_recovery.json").write_text(json.dumps(agg, indent=2),
                                                             encoding="utf-8")
    return agg
