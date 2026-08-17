"""Rhythm analysis of the learned CoST representations on the REAL HRD test set.

it analyses the encoder's trend / seasonal representations on the held-out HRD test 
set and asks:

    do the learned representations - and specifically the seasonal AMPLITUDE and
    PHASE features the model is built on - separate the depression-endpoint
    classes, and does the encoder actually disentangle trend from rhythm here?

Clinical link: in actigraphy/wearable depression research the circadian rhythm's
AMPLITUDE (strength; often blunted in depression) and PHASE (timing; often
shifted) are the established markers. CoST's seasonal branch encodes exactly
these two quantities (its pre-iFFT representation F, split into |F| and phi(F),
trained by L_amp and L_phase), so we can inspect them directly.

It produces, per variant, inside the variant's own results folder:
  * hrd_rhythm_separability.{csv,md,png} - AUC/F1/Acc of a logistic-regression
                                 probe on each representation (full, trend, season,
                                 amplitude, phase) vs the classical cosinor baseline
  * hrd_tsne_label.png           - t-SNE of TFD V^(T) & SFD V^(S), coloured by the
                                 depression label (which space separates the groups)
  * hrd_umap_label.png           - UMAP counterpart of the above (needs umap-learn)
  * hrd_rhythm.json              - machine-readable summary of all the above

Called from train_hrd.py with the in-memory model/data (no checkpoint reload).
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import normalize as l2normalize
from models.positional_encoding import position_matrix
from baselines.cosinor import N_PARAMS
from tasks.decomposition import RIDGE_ALPHAS, _ridge_fit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score, r2_score,
                             silhouette_score, balanced_accuracy_score, matthews_corrcoef)

# Okabe-Ito colour-blind-safe palette (widely used in reputable publications)
CLASS_COLORS = ["#0072B2", "#D55E00"]            # depression endpoint: 0 blue, 1 vermillion


# --------------------------------------------------------------------------- #
# 1. Representation extraction
# --------------------------------------------------------------------------- #
def extract_representations(model, X, batch_size=256, pool="mean"):
    """Return the learned representations of every window in X.

    full   (N, repr_dims)        : [V^(T); V^(S)] window-pooled rep (model.encode, `pool`) --
                                   the representation CoST actually uses downstream.
    V      (N, output_dims)      : the raw ENCODER/backbone output, mean-pooled over the
                                   window, taken BEFORE the trend (TFD) / seasonal (SFD) split.
    trend  (N, dS)               : V^(T)  (trend half of `full`)
    season (N, dS)               : V^(S)  (seasonal half of `full`, time domain)
    amp    (N, Ffreq*dS)         : |F|   -- seasonal AMPLITUDE
    phase  (N, Ffreq*dS)         : <F    -- seasonal PHASE

    `amp` / `phase` are computed EXACTLY as in the official CoST loss
    (cost.py: convert_coeff(rfft(F.normalize(season, dim=-1), dim=1))):
    the seasonal sequence is L2-normalised per channel, rFFT'd over time into the
    complex F in C^(Ffreq x dS), and split with amp = sqrt(Re^2+Im^2) and
    phase = atan2(Im, Re). Ffreq = floor(T/2)+1, dS = component_dims. They are
    flattened to (N, Ffreq*dS); CoST uses them only inside the pretraining loss, so
    here they are an analysis view (kept full-dim with strong L2 in the probe, see separability_table).
    """
    eps = 1e-6                                                   # same eps as cost.py
    full = model.encode(X, mode="forecasting", pool=pool).squeeze(1)   # (N, repr_dims)
    d = model.net.component_dims
    # `full` is [trend-part; season-part] with equal halves for every pool mode
    # (each half may be >d when pool='meanmax'), so split on the midpoint.
    half = full.shape[1] // 2
    trend, season = full[:, :half], full[:, half:]

    org = model.net.training
    model.net.eval()
    amp, phase, Vs = [], [], []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).float().to(model.device)
            v = model.net(xb, tcn_output=True)                   # (b, T, output_dims): V, pre-decomp
            Vs.append(v.mean(dim=1).cpu().numpy())               # window mean-pool -> (b, output_dims)
            _, season_seq = model.net(xb)                        # (b, T, dS) time domain
            season_seq = l2normalize(season_seq, dim=-1)         # exactly as cost.py
            Ff = torch.fft.rfft(season_seq.float(), dim=1)       # (b, Ffreq, dS) complex
            a = torch.sqrt((Ff.real + eps).pow(2) + (Ff.imag + eps).pow(2))
            p = torch.atan2(Ff.imag, Ff.real + eps)
            amp.append(a.reshape(a.size(0), -1).cpu().numpy())   # (b, Ffreq*dS)
            phase.append(p.reshape(p.size(0), -1).cpu().numpy())
    model.net.train(org)
    return {
        "full": full, "trend": trend, "season": season, "V": np.concatenate(Vs),
        "amp": np.concatenate(amp),
        "phase": np.concatenate(phase),
        # Same seasonal branch, read in the frequency domain at the chronobiological harmonics
        # instead of pooled over time. Directly comparable with the `season` row above: same
        # weights, same windows, only the readout differs -- and time pooling provably keeps
        # only the DC bin (see CoST._seasonal_spectral).
        "season_spec": model.encode(X, mode="forecasting", pool=pool,
                                    season_pool="spec").squeeze(1)[:, half:],
    }


def representation_views(rep):
    """Map the raw arrays to the named representations used for the probe / embeddings.
    Full / Trend / Season are CoST's downstream reps; the |F| and <F views are the
    paper's seasonal amplitude / phase (337*dS), kept here as an analysis extension."""
    return {
        "V (encoder pre-decomp)": rep["V"],
        "Full [V^(T);V^(S)]":  rep["full"],
        "Trend V^(T)":         rep["trend"],
        "Season V^(S)":        rep["season"],
        "Season V^(S) spectral": rep["season_spec"],
        "Seasonal amp":   rep["amp"],
        "Seasonal phase": rep["phase"],
    }


def cosinor_markers_per_channel(cf, n_channels, top_k=2):
    """Per-window (amplitude, acrophase) for EACH channel separately -- each (N, n_channels).

    There is deliberately no second cosinor implementation: the markers used to validate the
    latent space are read from the same `paper_cosinor_features` matrix that serves as the
    baseline, so the target and the competitor cannot disagree. Only the DOMINANT period of
    each channel (k=0) is used. Because the fit is clock-anchored and subject-aggregated, each
    acrophase is a wall-clock angle (0 = midnight) and is comparable across participants.

    This is the form every CLAIM should use. Heart rate, ambulatory activity and sleep have
    different acrophases by definition -- they are different chronobiological constructs, not
    three noisy measurements of one -- so they are carried separately and reported separately.
    """
    A, TH, M = [], [], []
    for c in range(n_channels):
        b = (c * top_k) * N_PARAMS                      # dominant-period block of this channel
        per = cf[:, b].astype(np.float64)
        M.append(cf[:, b + 1].astype(np.float64))       # MESOR (COSINOR_PARAM_COLS[1])
        A.append(cf[:, b + 2].astype(np.float64))       # Amplitude
        ph = cf[:, b + 4].astype(np.float64)            # Acrophase, in bins since midnight
        TH.append(np.where(per > 0, 2 * np.pi * ph / np.maximum(per, 1e-9), 0.0))
    return np.stack(A, 1), np.stack(TH, 1), np.stack(M, 1)


def cosinor_markers(cf, n_channels, top_k=2):
    """DISPLAY-ONLY pooled summary: one (amplitude, acrophase) per window. NOT a construct.

    Never use this as a probe target. Sleep runs roughly ANTIPHASE to activity and heart rate
    -- an `is_asleep` acrophase near 03:00 against a steps/HR acrophase near 15:00 -- so the
    amplitude-weighted circular mean below points somewhere between two different constructs,
    and where it lands is governed by the relative z-scored amplitudes of sleep vs activity,
    which is an artefact of preprocessing rather than physiology. Two antiphase channels cancel
    in proportion to their amplitudes however they are weighted; weighting only stops a WEAK
    rhythm from cancelling a STRONG one, which is a different problem. The amplitude half is
    likewise a mean over constructs that do not share a scale.

    It survives only because `clinical_marker_tsne` needs ONE angle per window to colour one
    scatter point. Everything that makes a claim -- E1.3's `rhythm_axis_probe` -- uses
    `cosinor_markers_per_channel` and reports each channel on its own.
    """
    A, TH, _ = cosinor_markers_per_channel(cf, n_channels, top_k)
    return A.mean(1), np.angle((A * np.exp(1j * TH)).sum(1))


def frequency_analysis(model, rep, variant_dir, seq_len, bin_minutes, top_k=8):
    """Persist WHAT THE SEASONAL (SFD / Fourier) BRANCH LEARNED, in the frequency domain.

    Frequency bin ``f`` of the length-``T`` window maps to a physical period
    ``period = window_hours / f`` (window_hours = T * bin_minutes / 60); e.g. for a 168 h
    window bin 7 = 24 h (circadian), 14 = 12 h, 1 = weekly. Two complementary views are saved:

      * ``weight_importance`` -- the learned per-frequency weight of the BandedFourierLayer,
        summarised as the Frobenius norm ||W_f|| of that frequency's complex weight matrix
        (how strongly the model *uses* frequency f). This is an importance proxy, not an
        amplitude.
      * ``repr_amplitude`` -- the mean amplitude of the seasonal representation at each
        frequency across all windows (what the latent actually *encodes*).

    Writes ``frequency_spectrum.{csv,json,png}`` into ``variant_dir``. Returns the summary."""
    variant_dir = Path(variant_dir)
    window_hours = seq_len * bin_minutes / 60.0

    # (1) learned weights of the seasonal Fourier layer. The weight is COMPLEX, so it has
    # both a MAGNITUDE (how strongly frequency f is used) AND a PHASE (how much it rotates
    # that frequency). Report both -- phase = magnitude-weighted circular mean = angle of the
    # summed complex weight (radians). Reporting magnitude alone would discard the phase,
    # which is half of what the SFD does and central to the circadian-timing hypothesis.
    W = model.net.sfd[0].weight.detach()                     # (num_freqs, in, out) complex
    importance = W.abs().pow(2).sum(dim=(1, 2)).sqrt().cpu().numpy()   # (num_freqs,) magnitude
    weight_phase = torch.angle(W.sum(dim=(1, 2))).cpu().numpy()        # (num_freqs,) radians

    # (2) data-driven: mean amplitude of the seasonal representation per frequency
    dS = int(model.net.component_dims)
    amp = np.asarray(rep["amp"])
    Ffreq = amp.shape[1] // dS if dS else importance.shape[0]
    spectrum = (amp.reshape(amp.shape[0], Ffreq, dS).mean(axis=(0, 2))
                if Ffreq else np.zeros_like(importance))

    n = min(importance.shape[0], len(spectrum))
    importance, spectrum, weight_phase = importance[:n], spectrum[:n], weight_phase[:n]
    f = np.arange(n)
    period = np.where(f == 0, np.inf, window_hours / np.maximum(f, 1))

    # CSV: every frequency bin
    lines = ["freq_bin,period_hours,weight_importance,weight_phase_rad,repr_amplitude"]
    for i in range(n):
        per = "inf" if not np.isfinite(period[i]) else f"{period[i]:.3f}"
        lines.append(f"{i},{per},{importance[i]:.6f},{weight_phase[i]:.6f},{spectrum[i]:.6f}")
    (variant_dir / "frequency_spectrum.csv").write_text("\n".join(lines), encoding="utf-8")

    # JSON: top-k periods (excluding DC f=0) + the key circadian / 12 h / weekly bins
    def _topk(vals):
        order = np.argsort(vals[1:])[::-1][:top_k] + 1       # skip DC bin 0
        return [{"period_hours": round(float(period[i]), 3), "freq_bin": int(i),
                 "value": round(float(vals[i]), 6)} for i in order]

    def _at(target_h):
        i = int(round(window_hours / target_h))
        if 1 <= i < n:
            return {"freq_bin": i, "period_hours": round(float(period[i]), 3),
                    "weight_importance": round(float(importance[i]), 6),
                    "weight_phase_rad": round(float(weight_phase[i]), 6),
                    "repr_amplitude": round(float(spectrum[i]), 6)}
        return None

    summary = {
        "backbone": model.net.backbone, "pe": model.net.pe,
        "window_hours": window_hours, "n_freq_bins": int(n),
        "top_by_weight_importance": _topk(importance),
        "top_by_repr_amplitude": _topk(spectrum),
        "key_periods": {"24h_circadian": _at(24.0), "12h": _at(12.0), "168h_weekly": _at(168.0)},
    }
    (variant_dir / "frequency_spectrum.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # NOTE: frequency_spectrum.png is intentionally NOT produced (removed on request).
    # The per-bin weight-importance / representation-amplitude data still live in
    # frequency_spectrum.{csv,json}; the depressed-vs-non spectrum is drawn separately
    # by group_spectrum_contrast -> frequency_contrast.png.
    return summary


def group_spectrum_contrast(rep, y, mask, pids, variant_dir, seq_len, bin_minutes, dS,
                            tag="", label_names=None, spectrum_title_noun="depression endpoint"):
    """Per-subject seasonal-representation spectra on the held-out test set, contrasted by
    depression endpoint, in ONE figure with two stacked panels that share the period x-axis:

        TOP -- seasonal AMPLITUDE |F|  vs period   (ALL test windows + bold group mean)
        MID -- seasonal PHASE      <F  vs period   (ALL test windows + bold circular mean)

    This answers the study's question directly: do the learned amplitude (rhythm STRENGTH) and
    phase (rhythm TIMING) look different between depressed and non-depressed? Amplitude/phase are
    read from the SEASONAL (SFD / weighted-Fourier) representation, reshaped to (N, Ffreq, dS) and
    reduced over the dS latent dims (mean for amplitude; AMPLITUDE-WEIGHTED circular mean for
    phase, so low-energy channels do not wash out the dominant phase). The full un-reduced
    (Ffreq x dS) field is drawn separately by group_spectrum_heatmap. EVERY test window is a faint
    curve (full distribution visible); the bold line is the group mean.

    Frequency bin f maps to the physical period ``window_hours / f``, so bin 1 = the whole
    window (168 h = weekly), bin 7 = 24 h (circadian), bin 14 = 12 h, ... -> the x-axis runs
    from weekly (left) to sub-daily (right). Writes the merged frequency_contrast.{csv,png};
    returns group sizes."""
    variant_dir = Path(variant_dir)
    label_names = label_names or {0: "non-depressed", 1: "depressed"}
    amp = np.asarray(rep["amp"]); phase = np.asarray(rep["phase"])
    Ffreq = amp.shape[1] // dS if dS else 0
    if not Ffreq:
        return None
    # reduce the dS latent channels -> one spectrum per window. Amplitude: mean |F|. Phase:
    # AMPLITUDE-WEIGHTED circular mean (= angle of the summed complex coefficients), so low-energy
    # channels -- whose phase is meaningless -- do not wash out the dominant rhythm's phase.
    amp_r = amp.reshape(amp.shape[0], Ffreq, dS)                                  # (N, Ffreq, dS)
    amp_wf = amp_r.mean(axis=2)                                                   # (N, Ffreq)
    z_wf = (amp_r * np.exp(1j * phase.reshape(amp.shape[0], Ffreq, dS))).mean(axis=2)   # (N, Ffreq)

    m = np.asarray(mask, bool)
    yy, pp = np.asarray(y)[m], np.asarray(pids)[m]
    amp_m, z_m = amp_wf[m], z_wf[m]

    # one curve per test SUBJECT (mean amplitude / circular-mean phase over that subject's windows)
    subj_amp = {0: [], 1: []}; subj_z = {0: [], 1: []}
    for pid in np.unique(pp):
        sel = pp == pid
        lbl = int(yy[sel][0])
        if lbl not in (0, 1):
            continue
        subj_amp[lbl].append(amp_m[sel].mean(axis=0))
        subj_z[lbl].append(z_m[sel].mean(axis=0))
    A = {lbl: (np.array(subj_amp[lbl]) if subj_amp[lbl] else np.empty((0, Ffreq))) for lbl in (0, 1)}
    Z = {lbl: (np.array(subj_z[lbl]) if subj_z[lbl] else np.empty((0, Ffreq), complex)) for lbl in (0, 1)}
    n_non, n_dep = len(A[0]), len(A[1])

    amp_mean = {lbl: (A[lbl].mean(axis=0) if len(A[lbl]) else np.full(Ffreq, np.nan)) for lbl in (0, 1)}
    ph_mean = {lbl: (np.angle(np.exp(1j * np.angle(Z[lbl])).mean(0)) if len(Z[lbl])
                     else np.full(Ffreq, np.nan)) for lbl in (0, 1)}      # circular mean of unit phases

    window_hours = seq_len * bin_minutes / 60.0
    f = np.arange(Ffreq)
    period = np.where(f == 0, np.inf, window_hours / np.maximum(f, 1))

    # ---- tidy CSV: per frequency -> group means and diffs ----
    dphi = np.angle(np.exp(1j * (ph_mean[1] - ph_mean[0])))              # circular dep-minus-non phase diff
    lines = ["freq_bin,period_hours,amp_nondepressed,amp_depressed,amp_diff,"
             "phase_nondepressed_rad,phase_depressed_rad,phase_diff_rad"]
    for i in range(Ffreq):
        per = "inf" if not np.isfinite(period[i]) else f"{period[i]:.4f}"
        lines.append(f"{i},{per},{amp_mean[0][i]:.6f},{amp_mean[1][i]:.6f},"
                     f"{amp_mean[1][i] - amp_mean[0][i]:.6f},"
                     f"{ph_mean[0][i]:.6f},{ph_mean[1][i]:.6f},{dphi[i]:.6f}")
    (variant_dir / "frequency_contrast.csv").write_text("\n".join(lines), encoding="utf-8")

    # ---- merged figure: amplitude (top) + phase (bottom) ----
    # every test-group window is a faint curve with the bold group mean over it (full distribution
    # visible).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        INK, GRID, MUTED = "#0b0b0b", "#d9d8d1", "#7a7873"
        COL = {0: "#E69F00", 1: "#0072B2"}       # non-depressed = orange, depressed = blue
        keep = (f >= 1) & (period >= 2.0)         # weekly (168 h) down to 2 h; drop DC + sub-2h noise
        o = np.argsort(period[keep])              # ascending period; inverted xlim puts weekly on the left
        xr = period[keep][o]
        sl = lambda v: np.asarray(v)[keep][o]     # reorder any per-frequency array onto the x-axis

        fig, (axA, axP) = plt.subplots(2, 1, figsize=(10.5, 7.6), sharex=True,
                                       gridspec_kw={"height_ratios": [1.0, 1.0]})
        # EVERY test-group window is drawn as its own faint curve (nothing is dropped), with the
        # bold group mean over it -> the full amplitude/phase distribution is visible per group.
        win_ph = np.angle(z_m)                          # (n_test_windows, Ffreq) per-window phase
        for lbl in (0, 1):
            idx = np.where(yy == lbl)[0]                 # ALL windows of this group's test subjects
            for i in idx:
                axA.plot(xr, sl(amp_m[i]),  color=COL[lbl], lw=0.6, alpha=0.10)
                axP.plot(xr, sl(win_ph[i]), color=COL[lbl], lw=0.6, alpha=0.10)
            if len(A[lbl]):                              # bold group mean (amplitude / circular phase)
                axA.plot(xr, sl(amp_mean[lbl]), color=COL[lbl], lw=2.8, solid_capstyle="round")
                axP.plot(xr, sl(ph_mean[lbl]),  color=COL[lbl], lw=2.8, solid_capstyle="round")

        for ax in (axA, axP):
            ax.set_xscale("log")
            ax.set_xlim(xr.max(), xr.min())      # weekly (168 h) on the LEFT -> sub-daily on the right
            ax.grid(alpha=0.3, color=GRID, lw=0.6)
            ax.xaxis.set_minor_locator(plt.NullLocator())
            for ph in (168.0, 24.0, 12.0):
                if xr.min() <= ph <= xr.max():
                    ax.axvline(ph, color=MUTED, ls=":", lw=1, zorder=0)
        for ph, lab in ((168.0, "168 h · weekly"), (24.0, "24 h · circadian"), (12.0, "12 h")):
            if xr.min() <= ph <= xr.max():
                axA.text(ph, 1.015, lab, transform=axA.get_xaxis_transform(),
                         ha="center", va="bottom", fontsize=8, color=MUTED)

        axA.set_ylabel("seasonal amplitude  |F|", fontsize=10.5, color=INK)
        axP.set_ylabel("seasonal phase  ∠F  (rad)", fontsize=10.5, color=INK)
        axP.set_ylim(-np.pi, np.pi)
        axP.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        axP.set_yticklabels(["−π", "−π/2", "0", "π/2", "π"])
        axP.set_xlabel("period  (hours, log scale)  —  weekly (left) → sub-daily (right)",
                       fontsize=10.5, color=INK)
        ticks = [t for t in (168, 72, 48, 24, 12, 6, 3, 2) if xr.min() <= t <= xr.max()]
        axP.set_xticks(ticks); axP.set_xticklabels([str(t) for t in ticks])

        handles = [
            Line2D([0], [0], color=COL[0], lw=2.8, label=f"{label_names.get(0)}  — group mean (n={n_non})"),
            Line2D([0], [0], color=COL[1], lw=2.8, label=f"{label_names.get(1)}  — group mean (n={n_dep})"),
            Line2D([0], [0], color=MUTED, lw=0.9, alpha=0.6, label="individual test window (faint)"),
        ]
        axA.legend(handles=handles, fontsize=9, frameon=False, loc="upper right")
        fig.suptitle(f"Seasonal-representation spectrum by {spectrum_title_noun}  —  {tag}",
                     fontsize=13, fontweight="bold", color=INK, x=0.012, ha="left")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(variant_dir / "frequency_contrast.png", dpi=200, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
    except Exception as e:
        print(f"[spectrum-contrast] plot skipped ({type(e).__name__}: {e})")
    return {"n_depressed": n_dep, "n_nondepressed": n_non}


def group_spectrum_heatmap(rep, y, mask, pids, variant_dir, seq_len, bin_minutes, dS,
                           tag="", label_names=None):
    """The FULL seasonal field (period x dS latent channels) WITHOUT collapsing the channels --
    the structure the line figure's mean-over-channels hides. One 2x3 grid: rows = amplitude /
    phase, cols = non-depressed / depressed / (depressed - non). Channels are ordered by their
    dominant period (shared across all panels) so the structure reads as a gradient. Writes
    frequency_heatmap.png; group maps are the mean amplitude / circular-mean phase over each
    group's test windows."""
    variant_dir = Path(variant_dir)
    names = label_names or {0: "non-depressed", 1: "depressed"}
    amp = np.asarray(rep["amp"]); phase = np.asarray(rep["phase"])
    Ffreq = amp.shape[1] // dS if dS else 0
    if not Ffreq:
        return None
    N = amp.shape[0]
    A = amp.reshape(N, Ffreq, dS); Zc = np.exp(1j * phase.reshape(N, Ffreq, dS))
    m = np.asarray(mask, bool); yy = np.asarray(y)[m]; Am, Zm = A[m], Zc[m]
    if not ((yy == 0).any() and (yy == 1).any()):
        return None
    G = {lbl: (Am[yy == lbl].mean(0), np.angle(Zm[yy == lbl].mean(0))) for lbl in (0, 1)}  # each (Ffreq, dS)

    window_hours = seq_len * bin_minutes / 60.0
    f = np.arange(Ffreq); period = np.where(f == 0, np.inf, window_hours / np.maximum(f, 1))
    idx = np.where((f >= 1) & (period >= 2.0))[0]                 # ascending f -> weekly (168 h) on the left
    per = period[idx]
    order = np.argsort(Am.mean(0)[idx].argmax(0))                # channels sorted by dominant period (shared)
    xt = [int(np.argmin(np.abs(per - p))) for p in (168, 24, 12, 6, 3) if per.min() <= p <= per.max()]
    xl = [str(int(round(per[t]))) for t in xt]
    sel = lambda M: M[idx].T[order]                              # (dS, n_freq) reordered for a panel

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        aN, aD = sel(G[0][0]), sel(G[1][0]); adiff = aD - aN
        pN, pD = sel(G[0][1]), sel(G[1][1]); pdiff = np.angle(np.exp(1j * (pD - pN)))
        amax = float(np.percentile(np.stack([aN, aD]), 99)); alim = float(np.percentile(np.abs(adiff), 99)) or 1e-6
        # phase is meaningless without energy -> grey out cells below the 90th amplitude percentile
        # AND below 20% of the peak (both groups must have energy for the diff), leaving only the
        # structured, high-energy phase.
        both = np.concatenate([aN.ravel(), aD.ravel()])
        athr = max(float(np.percentile(both, 90)), 0.20 * amax)
        cyc = plt.get_cmap("twilight").copy(); cyc.set_bad("0.85")
        div = plt.get_cmap("RdBu").copy();     div.set_bad("0.85")
        pN = np.ma.masked_where(aN < athr, pN); pD = np.ma.masked_where(aD < athr, pD)
        pdiff = np.ma.masked_where(np.minimum(aN, aD) < athr, pdiff)
        fig, ax = plt.subplots(2, 3, figsize=(13, 7.2), constrained_layout=True)
        def hm(a, M, cmap, vmin, vmax, ttl, cb):
            im = a.imshow(M, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            a.set_title(ttl, fontsize=10); a.set_xticks(xt); a.set_xticklabels(xl, fontsize=8)
            if cb:
                fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
        hm(ax[0, 0], aN, "magma", 0, amax, f"amplitude — {names[0]}", False)
        hm(ax[0, 1], aD, "magma", 0, amax, f"amplitude — {names[1]}", True)
        hm(ax[0, 2], adiff, "RdBu_r", -alim, alim, f"amplitude — {names[1]} − {names[0]}", True)
        hm(ax[1, 0], pN, cyc, -np.pi, np.pi, f"phase — {names[0]}", False)
        hm(ax[1, 1], pD, cyc, -np.pi, np.pi, f"phase — {names[1]}", True)
        hm(ax[1, 2], pdiff, div, -np.pi, np.pi, f"phase — {names[1]} − {names[0]} (circular)", True)
        for a in ax[:, 0]:
            a.set_ylabel(f"latent channel (of {dS})", fontsize=9)
        for a in ax[1, :]:
            a.set_xlabel("period (h)", fontsize=9)
        fig.suptitle(f"Seasonal amp/phase field — period × {dS} channels — {tag}\n"
                     "phase panels: grey = low amplitude (phase unreliable)",
                     fontsize=12, fontweight="bold")
        fig.savefig(variant_dir / "frequency_heatmap.png", dpi=200, facecolor="white")
        plt.close(fig)
    except Exception as e:
        print(f"[spectrum-heatmap] skipped ({type(e).__name__}: {e})")
    return {"Ffreq": Ffreq, "dS": dS}


def participant_trajectory_figures(model, X, y, pids, baseline_by_pid, rep, variant_dir,
                                   seq_len, bin_minutes, dS, seed=42, prefer_pids=None):
    """WITHIN-PERSON contrast (not group contrast): for up to 4 representative participants
    -- one per (baseline, endpoint) depression-status trajectory {dep->dep, non->non,
    dep->non, non->dep} -- plot that SAME participant's FIRST vs LAST window, four panels:
    seasonal amplitude and phase (vs period, reusing the already-computed `rep` spectra), the
    raw TREND sequence V^(T) and the raw SEASONAL sequence V^(S) (each mean over its dS
    channels) vs time-within-window. The two time-domain panels are the waveform view of what
    the two frequency-domain panels above describe as a spectrum; panel 4 is dropped in plain
    (--no-disentangle) mode, where the encoder has no seasonal branch. The bold
    colors are "first week" vs "last week" of ONE person, not two different people; every week
    IN BETWEEN is drawn as a faint grey line so the whole within-person trajectory is visible --
    does this individual's own rhythm/baseline drift across the study, and is the first->last
    change gradual or abrupt? Writes participant_trajectory_<pid>.png per selected participant
    (skipped if fewer than 2 windows exist for a group, or matplotlib fails).

    Which participant illustrates each group is drawn with `seed`, from `prefer_pids` (the
    held-out set) when that group has a member there. Different seeds therefore illustrate
    different people -- matching the fact that each seed holds out a different test set -- and
    a given seed always reproduces the same choice."""
    variant_dir = Path(variant_dir)
    endpoint_by_pid = {p: int(y[pids == p][0]) for p in np.unique(pids)}
    groups = {}                                          # (baseline, endpoint) -> [pids]
    for p, base in baseline_by_pid.items():
        end = endpoint_by_pid.get(p)
        if end is None or int((pids == p).sum()) < 2:    # need >=2 windows for first/last
            continue
        groups.setdefault((int(base), end), []).append(p)

    # Draw one participant per group with the run's seed, preferring the held-out set.
    # Previously this was a setdefault, i.e. the first pid encountered -- deterministic and
    # seed-independent, so all three seeds illustrated the SAME four people even though each
    # seed holds out a different test set. Seeding the draw makes the seeds show different
    # participants (more coverage, and the figures stop being redundant copies), while staying
    # reproducible: the same seed always picks the same person. `prefer_pids` keeps the
    # illustration on participants the model was actually evaluated on; it falls back to the
    # whole group when a group has no test member, rather than dropping the panel.
    rng = np.random.default_rng(seed)
    prefer = set(prefer_pids or ())
    chosen = {}
    for k, cands in groups.items():
        pool = sorted(p for p in cands if p in prefer) or sorted(cands)
        chosen[k] = str(rng.choice(np.array(pool, dtype=object)))

    wanted = [(1, 1, "depressed -> depressed"), (0, 0, "non-depressed -> non-depressed"),
              (1, 0, "depressed -> non-depressed"), (0, 1, "non-depressed -> depressed")]
    selected = [(chosen[k[:2]], k[2]) for k in wanted if k[:2] in chosen]
    if not selected:
        print("[trajectory] no participant with >=2 windows found for any (baseline,endpoint) group")
        return

    amp = np.asarray(rep["amp"]); phase = np.asarray(rep["phase"])
    Ffreq = amp.shape[1] // dS if dS else 0
    if not Ffreq:
        return
    window_hours = seq_len * bin_minutes / 60.0
    f = np.arange(Ffreq)
    period = np.where(f == 0, np.inf, window_hours / np.maximum(f, 1))
    keep = (f >= 1) & (period >= 2.0)                     # weekly (168h) down to 2h
    order = np.argsort(period[keep])
    xr = period[keep][order]
    t_axis = np.arange(seq_len) * bin_minutes / 60.0      # hours within window
    COL = {"first": "#0072B2", "last": "#D55E00"}
    MUTED = "#9a9a9a"                                      # in-between weeks (faint)

    for pid, tag in selected:
        idx = np.where(pids == pid)[0]                    # chronological windows of this pid
        n_win = len(idx)

        def _amp_phase(i):
            a = amp[i].reshape(Ffreq, dS); p_ = phase[i].reshape(Ffreq, dS)
            z = (a * np.exp(1j * p_)).mean(axis=1)                       # amplitude-weighted
            return a.mean(axis=1)[keep][order], np.angle(z)[keep][order]  # circular mean phase

        ap = [_amp_phase(int(i)) for i in idx]            # (amp_curve, phase_curve) per window

        try:                                    # raw trend + seasonal sequences, ALL windows of this pid
            with torch.no_grad():
                org = model.net.training; model.net.eval()
                xb = torch.from_numpy(X[idx]).float().to(model.device)
                trend_seq, season_seq = model.net(xb)      # (n_win, T, dS); season None if plain
                model.net.train(org)
            trend_mean = trend_seq.mean(dim=-1).cpu().numpy()           # (n_win, T)
            # V^(S) in the TIME domain -- the counterpart of the trend panel. The amp/phase
            # panels show the SAME seasonal branch in the FREQUENCY domain, so this fourth
            # box is what those spectra actually look like as a waveform across the week.
            season_mean = (season_seq.mean(dim=-1).cpu().numpy()
                           if season_seq is not None else None)         # (n_win, T) or None
        except Exception as e:
            print(f"[trajectory] {pid} trend sequence skipped ({type(e).__name__}: {e})")
            continue

        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
            n_panel = 3 if season_mean is None else 4   # 4th box = seasonal waveform V^(S)
            fig, axes = plt.subplots(n_panel, 1, figsize=(9, 3.2 * n_panel))
            axA, axP, axT = axes[0], axes[1], axes[2]
            axS = axes[3] if season_mean is not None else None
            # in-between weeks first (faint grey), so the bold first/last lines sit on top
            for w in range(1, n_win - 1):
                axA.plot(xr, ap[w][0], color=MUTED, lw=0.9, alpha=0.35)
                axP.plot(xr, ap[w][1], color=MUTED, lw=0.9, alpha=0.35)
                axT.plot(t_axis, trend_mean[w], color=MUTED, lw=0.9, alpha=0.35)
                if axS is not None:
                    axS.plot(t_axis, season_mean[w], color=MUTED, lw=0.9, alpha=0.35)
            # bold first + last week on top
            axA.plot(xr, ap[0][0], color=COL["first"], lw=2.2)
            axA.plot(xr, ap[-1][0], color=COL["last"], lw=2.2)
            axP.plot(xr, ap[0][1], color=COL["first"], lw=2.2)
            axP.plot(xr, ap[-1][1], color=COL["last"], lw=2.2)
            axT.plot(t_axis, trend_mean[0], color=COL["first"], lw=2.2)
            axT.plot(t_axis, trend_mean[-1], color=COL["last"], lw=2.2)
            if axS is not None:
                axS.plot(t_axis, season_mean[0], color=COL["first"], lw=2.2)
                axS.plot(t_axis, season_mean[-1], color=COL["last"], lw=2.2)
            panels = [
                (axA, "period (h)", "seasonal amplitude |F|", True),
                (axP, "period (h)", "seasonal phase ∠F (rad)", True),
                (axT, "time within window (h)", "trend V^(T)  (mean over channels)", False),
            ]
            if axS is not None:
                panels.append((axS, "time within window (h)",
                               "seasonal V^(S)  (mean over channels)", False))
            for ax, xlab, ylab, logx in panels:
                if logx:
                    ax.set_xscale("log"); ax.set_xlim(xr.max(), xr.min())
                else:
                    # day boundaries -- the circadian structure of V^(S) is only readable
                    # against them (and they make the trend panel comparable)
                    for dline in range(24, int(t_axis[-1]) + 1, 24):
                        ax.axvline(dline, color="#cccccc", lw=0.7, ls=":", zorder=0)
                ax.set_xlabel(xlab, fontsize=9.5); ax.set_ylabel(ylab, fontsize=9.5)
                ax.grid(alpha=0.3)
            handles = [
                Line2D([0], [0], color=COL["first"], lw=2.2, label="first week"),
                Line2D([0], [0], color=COL["last"], lw=2.2, label="last week"),
            ]
            if n_win > 2:
                handles.append(Line2D([0], [0], color=MUTED, lw=0.9, alpha=0.6,
                                      label=f"in-between weeks (n={n_win - 2}, faint)"))
            axA.legend(handles=handles, fontsize=9, frameon=False, loc="upper right")
            fig.suptitle(f"{pid}  ({tag})  --  first vs last week  ({n_win} weeks total)",
                         fontsize=12, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(variant_dir / f"participant_trajectory_{pid}.png", dpi=200,
                       bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"[trajectory] {pid} ({tag}, {n_win} weeks) -> participant_trajectory_{pid}.png")
        except Exception as e:
            print(f"[trajectory] {pid} plot skipped ({type(e).__name__}: {e})")


# --------------------------------------------------------------------------- #
# 2. Quantitative separability table
# --------------------------------------------------------------------------- #
def _agg_participant(pids, prob, y):
    uniq = np.unique(pids)
    p = np.array([prob[pids == u].mean() for u in uniq])
    l = np.array([int(y[pids == u][0]) for u in uniq])
    return p, l


def _metrics(y_true, prob, thr=0.5):
    """AUC (threshold-free) + threshold metrics at `thr`. Balanced accuracy and MCC are
    added alongside F1/Acc: on the (near-)balanced test set F1 saturates at ~0.667 for a
    degenerate all-positive classifier, while MCC is 0 there -- a more honest operating-point
    score. Sensitivity (recall/TPR) and specificity (TNR) are the clinical rates: sensitivity
    = TP/(TP+FN) = fraction of depressed caught, specificity = TN/(TN+FP) = fraction of
    non-depressed correctly cleared."""
    y_true = np.asarray(y_true)
    pred = (prob >= thr).astype(int)
    auc = roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else float("nan")
    tp = int(((pred == 1) & (y_true == 1)).sum()); fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum()); fp = int(((pred == 1) & (y_true == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")     # recall / true-positive rate
    spec = tn / (tn + fp) if (tn + fp) else float("nan")     # true-negative rate
    return {
        "auc": auc,
        "f1": f1_score(y_true, pred, zero_division=0),
        "acc": accuracy_score(y_true, pred),
        "bacc": balanced_accuracy_score(y_true, pred),
        "mcc": matthews_corrcoef(y_true, pred),
        "sensitivity": sens,
        "specificity": spec,
    }


def _macro_participant_metrics(pids, y_true, prob, thr):
    """Within-person metrics, averaged over participants (macro average).

    Used when the label varies WITHIN a participant (emotional energy: one label per day).
    There is then no participant-level label to pool to, so `_agg_participant` is meaningless
    -- it would pair a person's mean probability with one arbitrary day's label. The honest
    participant-level question is instead asked inside each person: does the probe rank THIS
    person's high-energy days above their own low-energy days? That is computed per
    participant and averaged, so every person counts once regardless of how many windows they
    contribute.

    Participants whose test days are all one class are skipped -- AUC is undefined there.
    If fewer than two participants survive that filter the average is not interpretable, so
    every metric is returned as NaN rather than as a number resting on one person.
    """
    pids = np.asarray(pids)
    y_true = np.asarray(y_true)
    keys = ("auc", "f1", "acc", "bacc", "mcc", "sensitivity", "specificity")
    per = {}
    n = 0
    for p in np.unique(pids):
        sel = pids == p
        if len(np.unique(y_true[sel])) < 2:
            continue
        for k, v in _metrics(y_true[sel], prob[sel], thr).items():
            if not np.isnan(v):
                per.setdefault(k, []).append(v)
        n += 1
    if n < 2:
        return {k: float("nan") for k in keys}
    return {k: float(np.mean(per[k])) if per.get(k) else float("nan") for k in keys}


def _best_threshold(y_true, prob):
    """Decision threshold in [0.05, 0.95] that maximises BALANCED ACCURACY (= Youden's J up
    to a constant). Mirrors train_hrd.best_threshold so the separability metrics use the same
    operating point as the downstream model. Balanced accuracy scores a degenerate
    all-one-class predictor at 0.5, so -- unlike F1-max -- it never selects the collapsed
    threshold when a genuinely discriminative one exists (fixes the MCC=0 failure)."""
    best_thr, best_ba = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 37):
        ba = balanced_accuracy_score(y_true, (prob >= thr).astype(int))
        if ba > best_ba:
            best_ba, best_thr = ba, float(thr)
    return best_thr


def separability_table(views, y, pids, train_mask, test_mask, val_mask=None, seed=42,
                       highdim_threshold=2000, highdim_C=0.01, dim_labels=None,
                       pca_views=None, lowdim_C=1.0, agg_views=None, macro_pids=None):
    """Fit a logistic-regression probe on each representation (train split) and score it
    on the held-out test split, at window and participant level.

    This is a SELF-CONTAINED cross-view comparison on a SINGLE train/val/test split: every
    view is probed under identical settings so the rows are comparable to EACH OTHER. It is
    NOT a reproduction of the headline metrics.json 'participant_level' number -- that model
    uses k-fold CV (refit on all pool participants) while this table uses the single split,
    so the 'Full' row here is CLOSE TO but not identical to the headline AUC. `lowdim_C`
    should be set to the downstream --probe-c so at least the regularisation matches.

    The high-dimensional FFT views (Seasonal amp/phase, Ffreq*dS >> #windows) are kept in
    FULL -- no PCA -- and regularised with a strong L2 penalty (small `highdim_C`); the
    small representations (Full/Trend/Season/Cosinor, p < n) use `lowdim_C`.

    `agg_views` (set of view names): for these views the feature of EACH participant is first
    averaged over ALL of that participant's windows (circular mean for phase views), then
    broadcast back, so the one-window-per-participant probe row carries a STABLE per-person
    estimate instead of a single noisy week. Applied identically to train / val / test, so the
    aggregation is consistent (and stays pseudo-replication-free -- still one sample per
    participant). Intended for the noisy high-dimensional Seasonal amp/phase views.

    `pca_views` {view_name: n_components}: for those views the probe first reduces the
    features with PCA **fit on the TRAIN split only** (PCA lives inside the pipeline, so
    `clf.fit(train)` fits it and val/test are merely transformed -> leakage-safe). This gives
    a fair, dimensionality-matched comparison for the p>>n FFT views. n_components is clamped
    to min(request, n_features, n_train-1) -- PCA cannot exceed n_train-1 components -- and the
    ACTUAL count used is shown in the 'Dim' column (e.g. 'PCA61'). After reduction the probe
    uses the standard C=1.0, matching the low-dimensional views.

    The F1/Accuracy threshold is tuned PER REPRESENTATION on the participant-aggregated
    validation split (matches the downstream model; 0.5 if none) and reported in 'Thr';
    AUC is threshold-free. `dim_labels` gives the readable 'Ffreq×dS' form for the FFT views."""
    dim_labels = dim_labels or {}
    pca_views = pca_views or {}
    agg_views = agg_views or set()
    use_val = val_mask is not None and int(np.sum(val_mask)) > 0
    n_train = int(np.sum(train_mask))
    rows = []
    yte, pte = y[test_mask], pids[test_mask]
    for name, R in views.items():
        agg_note = None
        if name in agg_views:
            # per-person feature: average this view over ALL of the person's windows (circular
            # mean for phase), then broadcast. The one-window-per-participant probe row then
            # carries a stable per-person estimate, identical for train / val / test.
            circ = "phase" in name.lower()
            R = R.copy()
            for pid in np.unique(pids):
                sel = pids == pid
                R[sel] = (np.angle(np.exp(1j * R[sel]).mean(axis=0)) if circ
                          else R[sel].mean(axis=0))
            agg_note = f"{name}: per-participant mean over all windows"
        if name in pca_views:
            # PCA fit on TRAIN only (in-pipeline, leakage-safe). At most n_train-1 comps.
            n_comp = max(2, min(int(pca_views[name]), R.shape[1], n_train - 1))
            clf = make_pipeline(
                StandardScaler(),
                PCA(n_components=n_comp, random_state=seed),
                LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=seed),
            )
            native = f"PCA{n_comp}"
            note = (f"{name} = PCA {n_comp} comps (fit on train) <- "
                    f"{dim_labels.get(name, R.shape[1])}")
        else:
            highdim = R.shape[1] > highdim_threshold    # amp/phase spectra: p >> n
            C = highdim_C if highdim else lowdim_C       # strong L2 for p>>n, else downstream --probe-c
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=seed),
            )
            native = dim_labels.get(name, str(R.shape[1]))
            note = (f"{name} = {native} dims kept in full, strong L2 (C={C:g})"
                    if highdim else None)

        clf.fit(R[train_mask], y[train_mask])
        prob = clf.predict_proba(R[test_mask])[:, 1]

        # per-representation decision threshold, tuned on the participant-aggregated
        # validation split (matches the downstream model); 0.5 if no validation split.
        if use_val:
            vp, vl = _agg_participant(pids[val_mask],
                                     clf.predict_proba(R[val_mask])[:, 1], y[val_mask])
            thr = _best_threshold(vl, vp)
        else:
            thr = 0.5

        w = _metrics(yte, prob, thr)
        if macro_pids is not None:
            # per-DAY label: average the metric computed inside each participant
            p = _macro_participant_metrics(np.asarray(macro_pids)[test_mask], yte, prob, thr)
        else:
            # participant-level label: pool each person's windows into one sample
            pp, pl = _agg_participant(pte, prob, yte)
            p = _metrics(pl, pp, thr)
        row = {
            "Representation": name,
            "Dim": native,
            "Thr": float(thr),
            "Win AUC": w["auc"], "Win F1": w["f1"], "Win Acc": w["acc"],
            "Win BAcc": w["bacc"], "Win MCC": w["mcc"],
            "Win Sens": w["sensitivity"], "Win Spec": w["specificity"],
            "Subj AUC": p["auc"], "Subj F1": p["f1"], "Subj Acc": p["acc"],
            "Subj BAcc": p["bacc"], "Subj MCC": p["mcc"],
            "Subj Sens": p["sensitivity"], "Subj Spec": p["specificity"],
        }
        full_note = "; ".join(n for n in (note, agg_note) if n)
        if full_note:
            row["_highdim_note"] = full_note
        rows.append(row)
    return rows


def save_table(rows, variant_dir, table_tag="", unit_note=""):
    """Write the separability table as .csv/.md/.png.

    `table_tag` ('depression' / 'energy') goes into the file name, so the two downstreams
    never write the same file even if their output trees are merged or mis-copied.
    A 'Unit' column appears automatically when the rows carry one (the probe-unit ablation).
    """
    stem = f"hrd_rhythm_separability_{table_tag}" if table_tag else "hrd_rhythm_separability"
    has_unit = any("Unit" in r for r in rows)
    cols = (["Unit"] if has_unit else []) + [
            "Representation", "Dim", "Thr",
            "Win AUC", "Win F1", "Win Acc", "Win BAcc", "Win MCC",
            "Subj AUC", "Subj F1", "Subj Acc", "Subj BAcc", "Subj MCC"]

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    def cell_of(r, c):
        return fmt(r[c]) if c in r else ""

    # footnote: the high-dim FFT views are kept in full (no PCA) with strong L2.
    notes, seen = [], set()
    for r in rows:                                   # dedupe: identical per unit block
        n = r.get("_highdim_note")
        if n and n not in seen:
            seen.add(n); notes.append(n)
    note_line = ("FFT views kept in full, no PCA -- " + ";  ".join(notes)) if notes else ""
    if unit_note:
        note_line = (note_line + "  |  " + unit_note) if note_line else unit_note

    # CSV
    csv = ",".join(cols) + "\n" + "\n".join(
        ",".join(cell_of(r, c) for c in cols) for r in rows)
    (variant_dir / f"{stem}.csv").write_text(csv, encoding="utf-8")

    # Markdown
    md = ("| " + " | ".join(cols) + " |\n"
          + "| " + " | ".join("---" for _ in cols) + " |\n"
          + "\n".join("| " + " | ".join(cell_of(r, c) for c in cols) + " |" for r in rows))
    if note_line:
        md += f"\n\n*{note_line}*\n"
    (variant_dir / f"{stem}.md").write_text(md, encoding="utf-8")

    # Rendered PNG (best subject AUC highlighted)
    def _auc(i):
        v = rows[i].get("Subj AUC")
        return v if isinstance(v, float) and v == v else -1        # NaN-safe
    best = max(range(len(rows)), key=_auc)
    fig, ax = plt.subplots(figsize=(1.0 + 0.85 * len(cols), 0.6 * len(rows) + 1.2))
    ax.axis("off")
    cell = [[cell_of(r, c) for c in cols] for r in rows]
    tbl = ax.table(cellText=cell, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.5)
    tbl.auto_set_column_width(col=list(range(len(cols))))   # size columns to content (no Dim overflow)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#40466e"); tbl[0, j].get_text().set_color("white")
        tbl[best + 1, j].set_facecolor("#FFF2CC")
    # Shade alternating probe-unit blocks so the ablation is readable at a glance.
    if has_unit:
        blocks, prev = [], None
        for i, r in enumerate(rows):
            u = r.get("Unit", "")
            if u != prev: blocks.append(i); prev = u
        for bi, start in enumerate(blocks):
            if bi % 2 == 0: continue
            end = blocks[bi + 1] if bi + 1 < len(blocks) else len(rows)
            for i in range(start, end):
                if i == best: continue
                for j in range(len(cols)):
                    tbl[i + 1, j].set_facecolor("#F2F2F2")
    title = "HRD test-set separability per representation (logistic-regression probe)"
    if table_tag:
        title += f"  -- {table_tag}"
    ax.set_title(title, fontsize=11, pad=12)
    if note_line:
        fig.text(0.5, 0.005, note_line, ha="center", va="bottom",
                 fontsize=7, style="italic", color="0.35")
    fig.tight_layout()
    fig.savefig(variant_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 3. 2-D embedding (t-SNE / UMAP) coloured by the depression label
# --------------------------------------------------------------------------- #
def _embed_prep(X, max_dim=50):
    """Standardise, then PCA-reduce very high-dimensional views before t-SNE/UMAP.

    The Seasonal amp/phase views are Ffreq*dS (~50k) dimensional; run directly, t-SNE/UMAP
    are dominated by distance concentration and show noise. Reducing to <=`max_dim` PCA
    components first lets the embedding reflect the actual class structure. Low-dim views
    (Trend/Season, 160-d) pass through unchanged."""
    X = StandardScaler().fit_transform(np.asarray(X, dtype=np.float64))
    if X.shape[1] > max_dim:
        from sklearn.decomposition import PCA
        n = min(max_dim, X.shape[0] - 1, X.shape[1])
        if n >= 2:
            X = PCA(n_components=n, random_state=0).fit_transform(X)
    return X


def _tsne(X, seed):
    z = _embed_prep(X)
    perp = max(5, min(30, (len(z) - 1) // 3))
    return TSNE(n_components=2, perplexity=perp, init="pca",
                learning_rate="auto", random_state=seed).fit_transform(z)


def _umap(X, seed):
    import umap                                          # optional dep (umap-learn)
    z = _embed_prep(X)
    n_neighbors = max(2, min(15, len(z) - 1))
    return umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1,
                     random_state=seed).fit_transform(z)


def _subsample(mask_idx, y, max_points, seed):
    if len(mask_idx) <= max_points:
        return mask_idx
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y[mask_idx]):
        idx_c = mask_idx[y[mask_idx] == c]
        n = max(1, int(round(max_points * len(idx_c) / len(mask_idx))))
        keep.append(rng.choice(idx_c, size=min(n, len(idx_c)), replace=False))
    return np.concatenate(keep)


def _subject_aggregate_views(views, y, pids, mask, keys):
    """Mean of each panel representation per subject (over the windows in `mask`).

    Produces one point per subject -- the embedding counterpart of the subject-level
    classifier, so the t-SNE/UMAP unit matches the depression label's unit (instead
    of ~29 correlated windows per subject dominating the layout). Returns
    (subject_views {key: (S, d)}, subject labels (S,))."""
    mp = pids[mask]
    ym = y[mask]
    uniq = np.unique(mp)
    sub_views = {k: np.stack([np.asarray(views[k])[mask][mp == u].mean(axis=0)
                              for u in uniq]) for k in keys}
    y_subj = np.array([int(ym[mp == u][0]) for u in uniq], dtype=int)
    return sub_views, y_subj


# The four rhythm views to embed. Trend/Season (time-domain) barely separate the classes;
# the predictive signal lives in the SFD's Seasonal amplitude |F| and phase <F, so those are
# shown too -- the contrast makes visible WHERE the depression signal is.
# Name of the classical-Cosinor view. Referenced in two places (where the view is built and
# where the probe-unit ablation decides what may be per-participant averaged), so it is a
# constant rather than a repeated literal.
COSINOR_VIEW = "Cosinor (paper)"

EMBED_PANELS = [("Trend V^(T)", "Trend  V^(T)"),
                ("Season V^(S)", "Season  V^(S)"),
                ("Seasonal amp", "Seasonal amplitude  |F|"),
                ("Seasonal phase", "Seasonal phase  <F")]
EMBED_KEYS = [k for k, _ in EMBED_PANELS]


def label_embedding_figure(views, y, idx, variant_dir, fname, method, embed, heading,
                           label_names, seed, panels=None):
    """One panel per representation, each reduced to 2-D by `embed` (t-SNE/UMAP) and coloured
    by the depression endpoint. Defaults to the four rhythm views (Trend, Season, and the
    predictive Seasonal amplitude/phase). The per-panel silhouette quantifies the separation."""
    variant_dir = Path(variant_dir)
    panels = panels or EMBED_PANELS
    ncols = 2
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.6 * nrows), squeeze=False)
    axes = axes.ravel()
    yi = y[idx]
    for ax, (key, title) in zip(axes, panels):
        emb = embed(np.asarray(views[key])[idx], seed)
        try:
            sil = silhouette_score(emb, yi) if len(np.unique(yi)) > 1 else float("nan")
        except Exception:
            sil = float("nan")
        for c in np.unique(yi):
            m = yi == c
            ax.scatter(emb[m, 0], emb[m, 1], s=10, alpha=0.7, linewidths=0,
                       color=CLASS_COLORS[int(c) % len(CLASS_COLORS)],
                       label=label_names.get(int(c), f"class {int(c)}"))
        ax.set_title(f"{title}   (silhouette={sil:.2f})", fontsize=10)
        ax.set_xlabel(f"{method} 1", fontsize=9); ax.set_ylabel(f"{method} 2", fontsize=9)
        ax.tick_params(labelsize=8); ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize=8, framealpha=0.85)
    for j in range(len(panels), len(axes)):              # hide any unused axes
        axes[j].axis("off")
    fig.suptitle(heading, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(variant_dir / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _interdaily_stability(Xs, bpd, per_channel=False):
    """Interdaily stability IS per window (Witting et al.): variance of the mean 24-h profile
    over total variance -> day-to-day regularity in [0, 1] (higher = more stable).
    IS = nd * SS_between_bins / SS_total, with nd = #days, bpd = bins/day.

    ``per_channel=True`` returns (N, C) -- the form E1.3 probes, because "the sleep rhythm is
    regular" and "the heart-rate rhythm is regular" are separate statements and a mean over
    channels can hide one collapsing while another holds. The default (N,) channel-mean is the
    display summary kept for the t-SNE colouring and for RQ2's raw markers."""
    N, T, C = Xs.shape
    nd = T // bpd
    if nd < 2:
        return np.full((N, C), np.nan) if per_channel else np.full(N, np.nan)
    m = Xs[:, :nd * bpd].reshape(N, nd, bpd, C)
    grand = m.mean(axis=(1, 2))                                  # (N, C)
    prof = m.mean(axis=1)                                        # (N, bpd, C) mean 24-h profile
    ss_between = ((prof - grand[:, None]) ** 2).sum(axis=1)      # (N, C)
    ss_total = ((m - grand[:, None, None]) ** 2).sum(axis=(1, 2))
    IS = np.where(ss_total > 0, nd * ss_between / ss_total, np.nan)
    return IS if per_channel else np.nanmean(IS, axis=1)         # (N, C) or (N,)


def clinical_marker_tsne(emb_view, Xs, idx, variant_dir, bin_minutes, seed, tag, cf,
                         n_sensors, top_k=2, fname="hrd_tsne_clinical.png"):
    """t-SNE of the latent embedding coloured by CLINICAL circadian markers (not the label):
    cosinor amplitude (rhythm strength), acrophase (peak timing), interdaily stability (day-to-day
    regularity). A smooth colour gradient => the latent space is organised by rhythm biology. One
    t-SNE, three colourings (amplitude/IS = viridis; acrophase = cyclic, coloured as hour-of-day)."""
    variant_dir = Path(variant_dir)
    amp, acro = cosinor_markers(cf, n_sensors, top_k)            # clock-anchored, subject-level
    acro_h = (acro % (2 * np.pi)) * 24 / (2 * np.pi)             # peak hour of day
    IS = _interdaily_stability(Xs, int(round(24 * 60 / bin_minutes)))
    emb = _tsne(np.asarray(emb_view)[idx], seed)
    panels = [("cosinor amplitude (rhythm strength)", amp[idx], "viridis", None),
              ("acrophase (hour of peak)", acro_h[idx], "twilight", (0, 24)),
              ("interdaily stability (regularity)", IS[idx], "viridis", (0, 1))]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (ttl, c, cmap, lim) in zip(axes, panels):
        vmin, vmax = lim or (np.nanpercentile(c, 5), np.nanpercentile(c, 95))
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=c, s=12, alpha=0.85, cmap=cmap,
                        vmin=vmin, vmax=vmax, linewidths=0)
        ax.set_title(ttl, fontsize=10); ax.set_xlabel("t-SNE 1", fontsize=9)
        ax.set_ylabel("t-SNE 2", fontsize=9); ax.tick_params(labelsize=8); ax.grid(alpha=0.2)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"Latent space coloured by clinical circadian markers  —  {tag}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(variant_dir / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _sim_vs_distance(S):
    """s(d) = mean over windows and t of cos(V_t, V_{t+d}).  S: (n, T, dim) -> (T,).

    Computed as the autocorrelation of the unit-normalised sequence via FFT: for unit
    vectors the inner product IS the cosine, and summing the per-channel autocorrelations
    gives every lag at once in O(dim * T log T) instead of the O(dim * T^2) of an explicit
    pair loop (451k pairs per window at T=672)."""
    u = S / (np.linalg.norm(S, axis=-1, keepdims=True) + 1e-8)
    T = u.shape[1]
    f = np.fft.rfft(u, n=2 * T, axis=1)
    ac = np.fft.irfft(f * np.conj(f), n=2 * T, axis=1)[:, :T].sum(-1)   # (n, T), unnormalised
    return (ac / np.arange(T, 0, -1)).mean(0)


def _bin_of_week(window_ids, T, bin_minutes):
    """Bin-of-week index (0 .. 7*bpd-1) of EVERY bin of every window, taken from the window's
    real start timestamp, which ``window_ids`` already carries as ``f"{pid}_{start.isoformat()}"``.

    This is what makes the anchored analysis honest on the depression windows too. Those start
    at each participant's first sample (``align_midnight=False``), so the sequence index says
    nothing about clock time -- but the calendar does, and it is right there in the id."""
    from datetime import datetime
    bpd = int(round(24 * 60 / bin_minutes))
    s0 = np.array([(lambda d: d.weekday() * bpd + (d.hour * 60 + d.minute) // bin_minutes)
                   (datetime.fromisoformat(str(w).rsplit("_", 1)[1])) for w in window_ids])
    return (s0[:, None] + np.arange(T)[None, :]) % (7 * bpd)


def circadian_similarity_figure(model, X, pids, mask, variant_dir, bin_minutes, tag,
                                table_tag="", batch_size=256, window_ids=None,
                                anchor_hour=9, max_windows=400, n_anchor=16, seed=42):
    """WavesFM Fig. 14, reproduced on this cohort: the DENSITY of pairwise cosine similarities,
    not just their mean. Two representations are compared as the paper compares its two
    (Stage I vs Stage II): CoST's trend branch V^(T) against its seasonal branch V^(S), the one
    that is supposed to carry rhythm.

    Three columns per branch:
      (a) Fig. 14a -- every sampled pair against the TIME DISTANCE between its two bins. A
          circadian representation ridges at 1, 2, 3 ... days; a merely smooth one decays.
      (b) Fig. 14b/c -- every bin against its position in the WEEK, referenced to each window's
          own ``anchor_hour`` bin on Monday. Peaks recurring each day at the anchor hour are
          the circadian cycle; a flattening over Sat/Sun is the circaseptan (weekday/weekend)
          shift the paper reports.
      (c) one line per PARTICIPANT: their own similarity profile folded onto a single 24 h
          clock. This is the individual-level view -- a group density can look rhythmic while
          no single person is, and this panel shows whether the cycle is really per-person.

    Everything is indexed by CALENDAR time via ``window_ids`` (see ``_bin_of_week``), so the
    x axes mean clock time whether or not the windows were midnight-aligned."""
    variant_dir = Path(variant_dir)
    bpd = int(round(24 * 60 / bin_minutes))
    idx = np.where(mask)[0]
    if len(idx) < 20 or X.shape[1] < 2 * bpd or window_ids is None:
        print(f"[circadian] skipped ({len(idx)} windows, window_ids="
              f"{'yes' if window_ids is not None else 'MISSING'})")
        return {}
    rng = np.random.default_rng(seed)
    if len(idx) > max_windows:                      # the density saturates long before this
        idx = np.sort(rng.choice(idx, max_windows, replace=False))
    Xs, pm = np.asarray(X)[idx], np.asarray(pids)[idx]
    bow = _bin_of_week([window_ids[i] for i in idx], Xs.shape[1], bin_minutes)

    org = model.net.training
    model.net.eval()
    seqs = {"Trend V^(T)": [], "Season V^(S)": []}
    with torch.no_grad():
        for i in range(0, len(Xs), batch_size):
            xb = torch.from_numpy(Xs[i:i + batch_size]).float().to(model.device)
            tr, se = model.net(xb)
            seqs["Trend V^(T)"].append(tr.float().cpu().numpy())
            if se is not None:
                seqs["Season V^(S)"].append(se.float().cpu().numpy())
    model.net.train(org)
    seqs = {k: np.concatenate(v, 0) for k, v in seqs.items() if v}

    T = Xs.shape[1]
    anchor_bow = anchor_hour * 60 // bin_minutes            # Monday, anchor_hour:00
    ai = np.argmax(bow == anchor_bow, axis=1)               # a 7-day window hits it exactly once
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    out = {}

    fig, axes = plt.subplots(len(seqs), 3, figsize=(16.5, 4.3 * len(seqs)), squeeze=False)
    for r, (name, S) in enumerate(seqs.items()):
        u = S / (np.linalg.norm(S, axis=-1, keepdims=True) + 1e-8)
        n = len(u)

        # ---- (a) density vs time distance -------------------------------------------
        a = rng.integers(0, T, size=(n, n_anchor))
        sim_a = np.einsum('nad,ntd->nat', u[np.arange(n)[:, None], a], u)
        lag_a = np.abs(a[:, :, None] - np.arange(T)[None, None, :]) * bin_minutes / 1440.0
        ax = axes[r][0]
        ax.hexbin(lag_a.ravel(), sim_a.ravel(), gridsize=90, cmap="Reds", mincnt=1, bins="log")
        for d in range(1, 8):
            ax.axvline(d, color="0.55", lw=0.6, alpha=0.7)
        ax.set_xlabel("time distance (days)", fontsize=9)
        ax.set_ylabel(f"{name}\ncosine similarity", fontsize=9)
        ax.set_title("(a) all intra-window pairs vs time distance", fontsize=9)

        # ---- (b) density across the week, anchored to Monday anchor_hour -------------
        # The anchor bin is kept, exactly as in the paper: every window is anchored to the SAME
        # bin-of-week, so masking it would empty that column and punch a hole at the very
        # reference point. Its cos = 1 IS the reference, and the daily peak at the anchor hour
        # is the finding, not an artefact.
        sim_b = np.einsum('nd,ntd->nt', u[np.arange(n), ai], u)
        ax = axes[r][1]
        ax.hexbin(bow.ravel(), sim_b.ravel(), gridsize=90, cmap="Reds", mincnt=1, bins="log")
        prof = (np.bincount(bow.ravel(), sim_b.ravel(), 7 * bpd)
                / np.maximum(np.bincount(bow.ravel(), minlength=7 * bpd), 1))
        ax.plot(np.arange(7 * bpd), prof, color="#3b9ad9", lw=1.4, label="weighted avg")
        for d in range(1, 7):
            ax.axvline(d * bpd, color="0.55", lw=0.6, alpha=0.7)
        ax.set_xticks(np.arange(7) * bpd + bpd // 2); ax.set_xticklabels(days, fontsize=8)
        ax.set_title(f"(b) referenced to each window's Mon {anchor_hour:02d}:00 bin", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")

        # ---- (c) per-participant 24 h profile ----------------------------------------
        ax, tod = axes[r][2], bow % bpd
        curves = []
        for p in np.unique(pm):
            m = pm == p
            c = (np.bincount(tod[m].ravel(), sim_b[m].ravel(), bpd)
                 / np.maximum(np.bincount(tod[m].ravel(), minlength=bpd), 1))
            curves.append(c); ax.plot(np.arange(bpd), c, color="0.55", lw=0.6, alpha=0.45)
        curves = np.array(curves)
        ax.plot(np.arange(bpd), np.median(curves, 0), color="#c0392b", lw=2.0, label="median")
        ax.axvline(anchor_bow, color="#3b9ad9", lw=1.0, ls="--", label=f"{anchor_hour:02d}:00")
        ax.set_xticks(np.arange(0, bpd, bpd // 4))
        ax.set_xticklabels([f"{h:02d}h" for h in range(0, 24, 6)], fontsize=8)
        ax.set_title(f"(c) per-participant 24 h profile  (n={len(curves)})", fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.2)

        s = _sim_vs_distance(S)
        # Peak-to-trough of the group 24 h profile: how much of the daily swing survives
        # averaging, i.e. how strongly the representation is organised by clock time.
        day_prof = prof.reshape(7, bpd).mean(0)
        out[name] = {
            "circadian_index": float(s[bpd] - 0.5 * (s[bpd // 2] + s[bpd + bpd // 2])),
            "diurnal_amplitude": float(day_prof.max() - day_prof.min()),
            # Fraction of participants whose OWN profile swings at least half as much as the
            # group's -- the group density can be rhythmic while few individuals are.
            "frac_participants_rhythmic": float(np.mean(
                (curves.max(1) - curves.min(1)) >= 0.5 * (day_prof.max() - day_prof.min()))),
            "mean_similarity": float(s.mean()), "n_windows": int(n),
            "s_curve": [round(float(x), 5) for x in s], "bins_per_day": int(bpd)}

    fig.suptitle(f"Circadian structure of the representation  -  {tag}"
                 + (f"  ({table_tag})" if table_tag else ""), fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    stem = f"circadian_similarity_{table_tag}" if table_tag else "circadian_similarity"
    fig.savefig(variant_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def _fold(M, idx, n):
    """Average M over all (i, j) sharing the same (idx[i], idx[j]) -> (n, n)."""
    key = (idx[:, None] * n + idx[None, :]).ravel()
    return (np.bincount(key, M.ravel(), n * n)
            / np.maximum(np.bincount(key, minlength=n * n), 1)).reshape(n, n)


def position_geometry_figure(model, variant_dir, seq_len, bin_minutes, tag, table_tag=""):
    """Pairwise geometry of the model's POSITION code (WavesFM Fig. 13).

    ``position_matrix`` returns a data-free ``seq_len x seq_len`` interaction matrix for
    whichever PE this variant uses. WavesFM can plot its time-of-day table directly because
    that table is indexed by clock time; ours is indexed by sequence position, so we
    factorise it after the fact -- position t is (day, time-of-day) = (t // bpd, t % bpd),
    and averaging the matrix over all pairs sharing a day pair, or a time-of-day pair,
    recovers the paper's two panels.

    The third panel subtracts, at every |i - j|, the mean value at that lag. Without it all
    three panels show only that neighbouring positions are alike, which is true of every
    encoding and discriminates nothing.

    With ``--pe factorized`` no folding is needed and none is done: that variant HAS the
    paper's two lookup tables, so the day-of-week and time-of-day panels are the exact
    7 x 7 and bpd x bpd cosine matrices of Fig. 13 rather than an after-the-fact estimate.
    """
    variant_dir = Path(variant_dir)
    bpd = int(round(24 * 60 / bin_minutes))
    cal = getattr(model.net, "cal_pe", None)
    if cal is not None:
        # Exact Fig. 13: cosine geometry of P_dow and P_tod themselves. Both calendar PEs
        # expose them through `tables()` -- learnable lookups for 'factorized', the linear
        # read-out of the fixed sin/cos basis for 'circular' -- so the two are directly
        # comparable panel for panel.
        cosm = lambda W: (lambda Q: (Q @ Q.T).numpy())(
            l2normalize(W.detach().cpu().float(), dim=-1))
        p_tod, p_dow = cal.tables()
        Mt = cosm(p_tod)
        lag = np.abs(np.subtract.outer(np.arange(bpd), np.arange(bpd)))
        lag = np.minimum(lag, bpd - lag)                    # time of day wraps at midnight
        prof_t = np.bincount(lag.ravel(), Mt.ravel()) / np.bincount(lag.ravel())
        panels = [("day-of-week  P_dow", cosm(p_dow)),
                  ("time-of-day  P_tod", Mt),
                  ("time-of-day, lag-standardised", Mt - prof_t[lag])]
        prof, seq_len = prof_t, bpd
    else:
        M = position_matrix(model.net, seq_len)
        if M is None:
            print("[position] no position code for this variant; figure skipped")
            return {}
        M = (M - M.mean()) / (M.std() + 1e-8)      # cosines and attention logits onto one scale
        lag = np.abs(np.subtract.outer(np.arange(seq_len), np.arange(seq_len)))
        prof = np.bincount(lag.ravel(), M.ravel()) / np.bincount(lag.ravel())
        day, tod = np.arange(seq_len) // bpd, np.arange(seq_len) % bpd
        panels = [("day pair (folded)", _fold(M, day, int(np.ceil(seq_len / bpd)))),
                  ("time-of-day pair (folded)", _fold(M, tod, bpd)),
                  ("time-of-day, lag-standardised", _fold(M - prof[lag], tod, bpd))]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (title, Z) in zip(axes, panels):
        v = float(np.abs(Z).max()) or 1.0
        im = ax.imshow(Z, cmap="coolwarm", vmin=-v, vmax=v, origin="lower")
        ax.set_title(title, fontsize=10)
        if len(Z) == bpd:                       # label the clock axes in hours, not bins
            ax.set_xticks(np.arange(0, bpd, bpd // 4))
            ax.set_yticks(np.arange(0, bpd, bpd // 4))
            ax.set_xticklabels([f"{h:02d}h" for h in range(0, 24, 6)], fontsize=8)
            ax.set_yticklabels([f"{h:02d}h" for h in range(0, 24, 6)], fontsize=8)
        elif len(Z) == 7:
            d7 = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            ax.set_xticks(range(7)); ax.set_yticks(range(7))
            ax.set_xticklabels(d7, fontsize=8, rotation=45); ax.set_yticklabels(d7, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"Positional-code geometry  -  {tag}"
                 + (f"  ({table_tag})" if table_tag else ""), fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    stem = f"position_geometry_{table_tag}" if table_tag else "position_geometry"
    fig.savefig(variant_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    # Two different lag profiles, so two different readings of the same question -- how much
    # the code separates times of day. Factorized: P_tod wraps at midnight, so the profile runs
    # 0..12 h and the contrast is same-hour vs opposite-hour. Folded: the profile runs over
    # sequence lag, so it is the prominence of the 24 h peak over the 12/36 h points.
    ci = (prof[0] - prof[bpd // 2] if cal is not None
          else prof[bpd] - 0.5 * (prof[bpd // 2] + prof[bpd + bpd // 2]))
    return {"circadian_index": float(ci), "exact_tables": cal is not None,
            "lag_profile": [round(float(x), 5) for x in prof], "bins_per_day": int(bpd)}


def _circ_corr(a, b):
    """Circular correlation (Jammalamadaka & SenGupta) between two angle vectors."""
    sa = np.sin(a - np.angle(np.mean(np.exp(1j * a))))
    sb = np.sin(b - np.angle(np.mean(np.exp(1j * b))))
    den = np.sqrt(np.sum(sa ** 2) * np.sum(sb ** 2))
    return float(np.sum(sa * sb) / den) if den > 0 else float("nan")


def _oof_ridge_selected(F, Y, groups, cv, alphas, seed):
    """Out-of-fold ridge predictions with the penalty selected INSIDE each fold.

    E1.3 used to pin ``Ridge(alpha=10.0)`` while E1.2 swept the CoST grid and chose lambda* on
    held-out participants -- two conventions for the same claim ("this marker is linearly
    readable from the latent"). A hand-set penalty also confounds a low R2 with a mis-set
    penalty, which is precisely the confound E1.2's sweep exists to remove
    (docs/RQ_Minimal_Experiment_Design.md, E1.2). This applies the E1.2 rule here: same grid
    (``RIDGE_ALPHAS``), same criterion (RMSE + MAE), same principle that the penalty is never
    chosen on the rows it is scored on.

    Nested, not flat: within each outer fold a participant-disjoint quarter of THAT fold's
    training participants is held out to pick lambda*, and the outer test fold is touched once,
    with lambda* already fixed. Choosing lambda* on the outer test fold would leak it.

    lambda* is then REFIT on the full outer-training fold, as GridSearchCV(refit=True) does --
    the inner split exists to choose the penalty, not to be thrown away. Fitting the returned
    model on the inner 75% instead would silently cost a quarter of the training data in every
    fold, which at ~36 participants is not affordable.

    Costs little more than a single fit: ``_ridge_fit`` pays the O(n p^2) Gram and its
    eigendecomposition once and reads the whole grid off that one decomposition.

    With a one-element ``alphas`` the selection is a no-op and this reduces EXACTLY to
    ``cross_val_predict(make_pipeline(StandardScaler(), Ridge(alpha)), ...)`` -- the invariant
    the unit check relies on.

    Returns ``(pred (n, C) aligned with F, [lambda* per outer fold])``.
    """
    F = np.asarray(F)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    pred = np.full((len(F), Y.shape[1]), np.nan)
    chosen, degenerate = [], False
    rng = np.random.RandomState(seed)
    for tr, te in cv.split(F, Y[:, 0], groups):
        g = np.unique(groups[tr])
        rng.shuffle(g)
        sel = np.isin(groups[tr], g[:max(1, int(round(0.25 * len(g))))])
        fit = ~sel
        if not fit.any() or not sel.any():       # too few participants to carve a selection
            fit = sel = np.ones(len(tr), bool)   # degenerate: selected on the fitted rows
            degenerate = True
        if len(alphas) > 1:
            inner = _ridge_fit(F[tr][fit], Y[tr][fit])
            err = [float(np.sqrt((r ** 2).mean()) + np.abs(r).mean())
                   for r in (inner(a, F[tr][sel]) - Y[tr][sel] for a in alphas)]
            a_star = float(alphas[int(np.argmin(err))])
        else:
            a_star = float(alphas[0])
        pred[te] = _ridge_fit(F[tr], Y[tr])(a_star, F[te])       # refit on the FULL train fold
        chosen.append(a_star)
    if degenerate:
        print("[rhythm] WARNING: too few participants to hold a selection set out inside a "
              "fold; lambda* was chosen on the rows it was fitted on for at least one fold.")
    return pred, chosen


def rhythm_axis_probe(emb, Xs, mask, pids, bin_minutes, variant_dir, seed, cf,
                      n_sensors, top_k=2, table_tag="", n_splits=5, sensor_cols=None):
    """Quantify what hrd_tsne_clinical.png shows by eye: is the latent space organised along
    known chronobiological axes?

    Ridge-regresses markers computed from the RAW signal (never from the model or the label)
    out of the embedding, out-of-fold with folds grouped by participant so a person's
    correlated windows never sit on both sides of a split. A mean predictor scores R^2 = 0, so
    R^2 > 0 means the marker is linearly readable from the latent.

    Every marker is PER CHANNEL. Heart rate, ambulatory activity and sleep have different
    acrophases by construction -- sleep is roughly antiphase to the other two -- so pooling
    them into one angle produces a quantity with no chronobiological referent whose value is
    driven by the relative z-scored amplitudes of sleep vs activity (see cosinor_markers).
    "HR acrophase is recoverable to within 1.4 h" is a claim; "the pooled acrophase is
    recoverable" is not. The same reasoning applies to amplitude and interdaily stability.

    This multiplies the probe count by n_sensors, so treat the per-channel table as a family
    and correct for multiplicity before calling any single channel significant.

    Acrophase is an angle, so it is fitted as sin/cos, recombined with arctan2, and scored with
    a circular correlation plus the median absolute error in hours (an R^2 on raw radians would
    be meaningless across the 0/2*pi wrap).

    The ridge penalty is NOT hand-set: it is swept over the same CoST grid as E1.2 and chosen
    inside each fold on held-out participants (see _oof_ridge_selected), so RQ1 states one
    convention for both probes. The lambda* actually used is reported per marker."""
    emb, pm = np.asarray(emb)[mask], pids[mask]
    Xm = Xs[mask]
    AMP, ACRO, MESOR = cosinor_markers_per_channel(np.asarray(cf)[mask], n_sensors, top_k)
    IS = _interdaily_stability(Xm, int(round(24 * 60 / bin_minutes)), per_channel=True)
    names = ([str(s) for s in sensor_cols][:n_sensors] if sensor_cols is not None
             else [f"ch{c}" for c in range(n_sensors)])

    n_groups = len(np.unique(pm))
    if n_groups < 3 or len(emb) < 20:
        return {}
    alphas = tuple(RIDGE_ALPHAS)

    def at_bound(lam):
        """True = some fold chose a grid ENDPOINT, i.e. it wanted a penalty the grid does not
        offer and the R2 beside it is a truncated answer. Same flag decomposition.py reports
        for E1.2; extend the grid on that side before reading the number."""
        return bool(len(alphas) > 1 and set(lam) & {float(alphas[0]), float(alphas[-1])})

    def oof(F, target, ok):
        """Nested-CV prediction over the FINITE rows only.

        Restricting to `ok` is not cosmetic: a marker is NaN for a window whose cosinor fit did
        not converge, and the previous version masked only when SCORING while still handing
        those NaNs to the fit. The fold count follows the participants that survive the mask.
        """
        g = pm[ok]
        cv = GroupKFold(n_splits=int(min(n_splits, len(np.unique(g)))))
        return _oof_ridge_selected(F[ok], target[ok], g, cv, alphas, seed)

    # The reference level. "R2 > 0" on its own says nothing here: these markers are computed
    # FROM the raw window, so the raw window predicts them trivially. What licenses a claim
    # about the ENCODER is the gain over the same probe run on a PCA of the raw window at the
    # latent's own width -- reported as `gain_over_raw` on every latent row.
    flat = Xm.reshape(len(Xm), -1)
    spaces = {"": emb, " | raw PCA": PCA(n_components=min(emb.shape[1], *flat.shape),
                                         random_state=seed).fit_transform(flat)}

    out = {}
    targets = [(f"{m} [{nm}]", kind, v[:, c])
               for c, nm in enumerate(names)
               for m, kind, v in (("cosinor amplitude", "linear", AMP),
                                  ("cosinor MESOR", "linear", MESOR),
                                  ("interdaily stability", "linear", IS),
                                  ("acrophase", "circular", ACRO))]
    for name, kind, t in targets:
        ok = np.isfinite(t)
        if ok.sum() < 20 or len(np.unique(pm[ok])) < 3:
            continue
        if kind == "linear" and np.std(t[ok]) == 0:            # dead channel: nothing to track
            continue
        if kind == "circular" and np.std(np.sin(t[ok])) == 0 and np.std(np.cos(t[ok])) == 0:
            continue                                           # constant angle
        common = {"n_windows": int(ok.sum()), "channel": name[name.index("[") + 1:-1],
                  "n_participants": int(len(np.unique(pm[ok])))}
        for tag, F in spaces.items():
            key = name + tag
            if kind == "linear":
                pred, lam = oof(F, t[:, None], ok)
                pred = pred[:, 0]
                # R2 is reported alongside Pearson r on purpose. Out-of-fold R2 goes NEGATIVE
                # whenever the fit is worse than the global mean, which is easy to hit with
                # grouped folds and a few dozen participants, and a negative number is hard to
                # read. r still says whether the marker is tracked, separately from whether the
                # scale/offset generalise.
                r = float(np.corrcoef(t[ok], pred)[0, 1]) if np.std(pred) > 0 else float("nan")
                out[key] = {"metric": "R2", "value": float(r2_score(t[ok], pred)),
                            "pearson_r": r, **common,
                            "ridge_alpha_per_fold": lam, "ridge_alpha_at_bound": at_bound(lam)}
            else:
                # sin and cos are two outputs of ONE circular model, so they share a single
                # lambda* on the joint criterion instead of drifting to two unrelated penalties.
                P, lam = oof(F, np.c_[np.sin(t), np.cos(t)], ok)
                pred = np.arctan2(P[:, 0], P[:, 1])
                err = np.angle(np.exp(1j * (pred - t[ok])))    # wrapped to (-pi, pi]
                out[key] = {"metric": "circular r", "value": _circ_corr(t[ok], pred),
                            "median_abs_err_hours": float(np.median(np.abs(err)) * 12 / np.pi),
                            **common,
                            "ridge_alpha_per_fold": lam, "ridge_alpha_at_bound": at_bound(lam)}
        # the number that actually answers RQ1: what the ENCODER adds over the raw window
        out[name]["gain_over_raw"] = out[name]["value"] - out[name + " | raw PCA"]["value"]

    # The one defensible cross-channel summary: aggregate the ERROR, never the phase. An error
    # of 1.4 h means the same thing whether the construct is sleep timing or HR timing, so a
    # median over channels is interpretable -- whereas a median over the phases themselves is
    # the quantity cosinor_markers exists to warn about.
    ph = [v["median_abs_err_hours"] for k, v in out.items()
          if k.startswith("acrophase [") and not k.endswith("raw PCA")]
    if ph:
        out["acrophase | median over channels"] = {
            "metric": "median abs err (h)", "value": float(np.median(ph)),
            "n_channels": len(ph)}

    if out:
        stem = f"rhythm_axis_probe_{table_tag}" if table_tag else "rhythm_axis_probe"
        lines = ["| Marker | Metric | Value | Extra | n windows | n participants |",
                 "| --- | --- | --- | --- | --- | --- |"]
        for k, v in out.items():
            extra = (f"median err {v['median_abs_err_hours']:.2f} h"
                     if "median_abs_err_hours" in v
                     else (f"Pearson r = {v['pearson_r']:.3f}" if "pearson_r" in v else ""))
            lines.append(f"| {k} | {v['metric']} | {v['value']:.3f} | {extra} | "
                         f"{v.get('n_windows', '')} | {v.get('n_participants', '')} |")
        lines.append("\n*Out-of-fold Ridge from the latent, folds grouped by participant. "
                     "Markers come from the raw signal, not from the model or the label; "
                     "a mean predictor would score R2 = 0.*")
        lines.append("\n*One row per (marker, CHANNEL): sleep, activity and heart rate are "
                     "different chronobiological constructs with different acrophases -- sleep "
                     "is roughly antiphase to the other two -- so pooling them into one angle "
                     "would report a quantity with no referent. Treat the rows as a family and "
                     "correct for multiplicity before calling one channel significant.*")
        used = sorted({a for v in out.values() for a in v.get("ridge_alpha_per_fold", ())})
        lines.append(f"\n*Ridge penalty selected INSIDE each fold on held-out participants -- "
                     f"the same rule as E1.2 (CoST grid {alphas[0]:g} ... {alphas[-1]:g}, "
                     f"minimising RMSE + MAE), never hand-set. lambda* used across folds and "
                     f"markers: {', '.join(f'{a:g}' for a in used)}.*")
        (Path(variant_dir) / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 5. Driver
# --------------------------------------------------------------------------- #
def run_hrd_rhythm_analysis(model, X, y, pids, train_mask, test_mask, variant_dir,
                            seq_len, bin_minutes, sensor_cols=None, seed=42,
                            label_names=None, max_tsne_points=3000, batch_size=256,
                            val_mask=None, baseline_rows=None, window_ids=None, pool="mean",
                            probe_sel=None, probe_c=1.0, paper_cosinor_topk=2,
                            baseline_by_pid=None, subject_aggregate=True,
                            label_noun="endpoint", table_tag="", headline_unit="last"):
    """Produce the HRD test-set rhythm figures + table + JSON inside `variant_dir`.

    ``subject_aggregate`` (default True, the depression behaviour): the label is CONSTANT
    per participant, so figures/metrics are aggregated to one point per subject. Set it
    False for a per-DAY label (emotional energy): each WINDOW is its own unit -- a synthetic
    unique id per window makes the per-subject aggregations collapse to a pooled per-day
    contrast, the per-person averaging (agg_views) and the subject-level embeddings are
    skipped. ``label_noun`` only relabels the figure headings (e.g. 'emotional energy')."""
    variant_dir = Path(variant_dir)
    label_names = label_names or {0: "non-depressed (0)", 1: "depressed (1)"}
    tag = f"{model.net.backbone} / {model.net.pe}"
    # per-DAY label -> one unit per window (synthetic unique ids collapse every per-subject
    # aggregation below to per-window, giving the pooled high-vs-low-day contrast).
    pids_agg = pids if subject_aggregate else np.arange(len(y))

    # X is [sensor | time]; keep ONLY the sensor channels for every rhythm target
    # (the appended clock channels are deterministic 24 h cosines -> would
    # contaminate the "true" markers).
    n_sensors = len(sensor_cols) if sensor_cols is not None else X.shape[2]
    sensor_cols = list(sensor_cols) if sensor_cols is not None else \
        [f"ch{c}" for c in range(n_sensors)]
    Xs = X[:, :, :n_sensors]

    rep = extract_representations(model, X, batch_size=batch_size, pool=pool)
    views = representation_views(rep)
    # Classical benchmark = EXACT clone of the paper's CosinorPy model (Yan et al. 2022),
    # probed by the SAME logistic-regression probe as every other view (LR-only, as asked).
    # Computed only for the probe rows (train|val|test), cached across the run's variants.
    # Lazy import so a missing CosinorPy disables ONLY this view, not the whole analysis.
    try:
        from baselines.cosinor import paper_cosinor_features
        _sep_test = (test_mask & probe_sel) if probe_sel is not None else test_mask
        _need = train_mask | _sep_test
        if val_mask is not None:
            _need = _need | val_mask
        _cache = Path(variant_dir).parent / f"paper_cosinor_topk{paper_cosinor_topk}_cache.npz"
        views[COSINOR_VIEW] = paper_cosinor_features(
            Xs, bin_minutes, need_mask=_need, top_k=paper_cosinor_topk,
            cache_path=str(_cache), window_ids=window_ids, pids=pids)
    except Exception as e:
        import traceback
        print(f"[paper_cosinor] view skipped ({type(e).__name__}: {e}); "
              f"install CosinorPy to enable the paper-cosinor baseline")
        traceback.print_exc()
        try:                          # surface the reason IN the results folder, not only the
            (variant_dir / "paper_cosinor.FAILED.txt").write_text(   # SLURM stdout log
                f"paper_cosinor_features failed: {type(e).__name__}: {e}\n\n"
                + traceback.format_exc(), encoding="utf-8")
        except Exception:
            pass

    # what the seasonal Fourier branch learned in the frequency domain (periods + weights)
    try:
        freq = frequency_analysis(model, rep, variant_dir, seq_len, bin_minutes)
        top = freq["top_by_weight_importance"][:3]
        print("[freq] top periods by weight importance: "
              + ", ".join(f"{t['period_hours']:.1f}h" for t in top)
              + f"  -> {variant_dir/'frequency_spectrum.csv'}")
    except Exception as e:
        import traceback
        print(f"[freq] frequency analysis skipped ({type(e).__name__}: {e})")
        traceback.print_exc()

    # depressed vs non-depressed contrast of the seasonal spectrum (held-out test set):
    # ONE merged figure -- per-subject phase + amplitude vs period, with bold group means.
    try:
        gc = group_spectrum_contrast(rep, y, test_mask, pids_agg, variant_dir, seq_len,
                                     bin_minutes, int(model.net.component_dims),
                                     tag=tag, label_names=label_names,
                                     spectrum_title_noun=label_noun)
        if gc:
            print(f"[freq] {label_noun} seasonal spectrum (phase+amplitude; "
                  f"n1={gc['n_depressed']}, n0={gc['n_nondepressed']}) -> "
                  f"{variant_dir/'frequency_contrast.png'}")
        # full (period x dS) field WITHOUT collapsing channels -- the complete 337x160 picture
        group_spectrum_heatmap(rep, y, test_mask, pids_agg, variant_dir, seq_len, bin_minutes,
                               int(model.net.component_dims), tag=tag, label_names=label_names)
    except Exception as e:
        import traceback
        print(f"[freq-contrast] skipped ({type(e).__name__}: {e})")
        traceback.print_exc()

    # WITHIN-PERSON first-week-vs-last-week contrast (not group contrast): one representative
    # participant per (baseline, endpoint) trajectory -- does THIS person's own rhythm/trend
    # shift between the start and end of the study? Uses ALL windows (not just the test set),
    # so it works even for a trajectory group that isn't in the held-out test split.
    if baseline_by_pid:
        try:
            participant_trajectory_figures(model, X, y, pids, baseline_by_pid, rep,
                                           variant_dir, seq_len, bin_minutes,
                                           int(model.net.component_dims),
                                           seed=seed, prefer_pids=set(pids[test_mask]))
        except Exception as e:
            import traceback
            print(f"[trajectory] skipped ({type(e).__name__}: {e})")
            traceback.print_exc()

    # readable factored dim 'Ffreq×dS' for the FFT-based amplitude / phase views
    # (instead of the opaque flattened Ffreq*dS product)
    dS = int(model.net.component_dims)
    Ffreq = rep["amp"].shape[1] // dS if dS else 0
    ffts = f"{Ffreq}×{dS}"
    dim_labels = {"Seasonal amp": ffts, "Seasonal phase": ffts} if Ffreq else {}

    # Fair, dimensionality-matched versions of the p>>n FFT views: PCA fit on the TRAIN
    # split ONLY (leakage-safe -- PCA lives inside the probe pipeline, so val/test are only
    # transformed). Auto-clamped to n_train-1. Answers "is amp/phase's edge real structure, or
    # just ambient dimensionality?".
    #
    # 20, not 120. The clamp made the old target meaningless for the participant-level units:
    # n_train there is ~58 participants, so PCA_TARGET=120 became PCA57, and PCA with
    # n_train-1 components is a near-identity transform -- it keeps 100% of the training
    # variance, so the "(PCA)" rows in run 18606330 matched their raw counterparts to within
    # 0.003 AUC while claiming to be a dimensionality-matched control. 20 components against
    # ~58 training participants is a real 3:1 reduction, and being below the clamp it also
    # applies identically to the 'all' unit, so the three units stay comparable.
    PCA_TARGET = 20
    pca_views = {}
    # The noisy Seasonal amp/phase views are averaged PER PARTICIPANT over all of that
    # person's windows (a stable per-person circadian estimate) instead of one noisy week --
    # applied identically to train / val / test, and still one sample per participant.
    agg_views = set()
    if Ffreq:
        views["Seasonal amp (PCA)"] = rep["amp"]
        views["Seasonal phase (PCA)"] = rep["phase"]
        dim_labels["Seasonal amp (PCA)"] = ffts
        dim_labels["Seasonal phase (PCA)"] = ffts
        pca_views.update({"Seasonal amp (PCA)": PCA_TARGET,
                          "Seasonal phase (PCA)": PCA_TARGET})
        agg_views = {"Seasonal amp", "Seasonal phase",
                     "Seasonal amp (PCA)", "Seasonal phase (PCA)"}
    # Dimensionality-matched counterparts of the LOW-dim views as well. Without them the table
    # compares representations that differ in WIDTH as much as in content (320 vs 160 vs 96
    # against only ~58 training participants), so "Full beats Cosinor" could just mean "320
    # parameters beat 96". These rows put every representation at the same PCA_TARGET with the
    # same penalty, which isolates content from capacity. The raw rows stay, so the headline
    # 'Full' number remains comparable to metrics.json.
    # Assignment is a REFERENCE to the existing array, not a copy: the PCA happens inside the
    # probe pipeline at fit time, so these rows cost no extra memory.
    for _base in ["V (encoder pre-decomp)", "Full [V^(T);V^(S)]", "Trend V^(T)",
                  "Season V^(S)", COSINOR_VIEW]:
        if _base in views:
            _pn = f"{_base} (PCA)"
            views[_pn] = views[_base]
            dim_labels[_pn] = dim_labels.get(_base, str(np.asarray(views[_base]).shape[1]))
            pca_views[_pn] = PCA_TARGET
    if not subject_aggregate:
        agg_views = set()          # per-DAY label: never average a person's windows together

    # rigorous metric 1: separability of each representation (learned vs cosinor).
    # Evaluate the CLASSIFICATION on the SAME unit as the downstream headline: one last-week
    # window per participant when --probe-last-window is on (train/val are already restricted
    # upstream). Without this the probe trained on last-week windows but was scored on ALL
    # windows -- a train/test mismatch that made the 'Full' row disagree with metrics.json.
    # The spectra / embeddings below intentionally still use ALL windows (richer, per-pid mean).
    sep_test_mask = (test_mask & probe_sel) if probe_sel is not None else test_mask

    # --- probe-unit ablation -------------------------------------------------------------
    # The incoming train/val masks are already restricted to whatever unit the headline probe
    # used, so we recover the participant SETS from them and rebuild the masks per unit. The
    # last-window mask is derived here from `pids` rather than taken from `probe_sel`, because
    # probe_sel is all-True when the caller ran --probe-unit all.
    #   all         every window is a probe sample.
    #   last        one window per participant (the most recent).
    #   persubject  same rows as 'last', but each view is first averaged over ALL of that
    #               participant's windows (circular mean for phase views) via agg_views, so
    #               the single row carries the person's whole record.
    # NOTE: 'persubject' here is a per-participant MEAN of each view. The headline probe in
    # train_hrd.py --probe-unit persubject uses [mean|std] of the encoder embedding; the std
    # half has no meaning for the circular phase views, so it is not replicated in this table.
    # For a per-DAY label (emotional energy) the label varies within a participant, so
    # averaging a person's windows would destroy it -- 'persubject' is skipped there.
    def _pid_set(m):
        return set(pids[m]) if m is not None and int(np.sum(m)) else set()

    last_sel = np.zeros(len(pids), bool)
    for _p in np.unique(pids):
        last_sel[np.where(pids == _p)[0][-1]] = True

    tr_pids, va_pids = _pid_set(train_mask), _pid_set(val_mask)
    units = ["all", "last", "persubject"] if subject_aggregate else ["all"]
    if subject_aggregate:
        unit_note = ("units: all = every window; last = one (most recent) window per "
                     f"participant; persubject = one row per participant, each view averaged "
                     f"over all their windows ('{COSINOR_VIEW}' excepted -- its vector mixes "
                     "angular and linear parameters, so it stays the last-window fit). "
                     "Subj* = each participant pooled to one sample (mean probability).")
    else:
        # A per-DAY label. 'last' would score one arbitrary day per person (36 rows out of
        # ~1,800 test windows) and measures nothing stable -- in run 18975686 its AUCs were
        # centred on chance (mean 0.488, 53% below 0.5, sd 0.100 against 0.027 for 'all'),
        # with 42% of thresholds driven to the [0.05, 0.95] rails. 'persubject' is excluded
        # for the same underlying reason: averaging a person's windows destroys the label.
        unit_note = ("units: all = every window (the only valid unit here -- the label is "
                     "per-day, so 'last'/'persubject' would collapse a within-person signal "
                     "onto one arbitrary day). Subj* = the metric computed INSIDE each "
                     "participant and averaged over participants (within-person macro "
                     "average); participants whose test days are all one class are skipped.")

    rows = []
    for unit in units:
        if unit == "all":
            u_tr = np.isin(pids, list(tr_pids))
            u_va = np.isin(pids, list(va_pids)) if va_pids else None
            u_te = test_mask
            u_agg = set()
        else:                                            # 'last' and 'persubject' share rows
            u_tr = np.isin(pids, list(tr_pids)) & last_sel
            u_va = (np.isin(pids, list(va_pids)) & last_sel) if va_pids else None
            u_te = test_mask & last_sel
            # Aggregate every view EXCEPT the Cosinor one. Its 96-dim vector interleaves
            # linear parameters (MESOR, Amplitude, p-value, ...) with three angular ones
            # (Acrophase, Orthophase, Bathyphase) inside each 12-parameter block, and the
            # circular branch in separability_table keys off the view NAME, so it would
            # arithmetic-average those angles (mean of 350 deg and 10 deg -> 180, not 0).
            # Leaving it out means its 'persubject' row is the same last-window fit as the
            # 'last' block, which is stated in the footnote, rather than a wrong number.
            # startswith, not set-difference: this must also exclude the "(PCA)" counterpart
            # of the cosinor view, whose vector mixes angular and linear parameters just the
            # same and so must not be arithmetically averaged either.
            u_agg = ({v for v in views if not v.startswith(COSINOR_VIEW)}
                     if unit == "persubject" else set())
        if int(u_tr.sum()) < 4 or int(u_te.sum()) < 2 or len(np.unique(y[u_tr])) < 2:
            print(f"[rhythm] probe-unit '{unit}' skipped (too few rows: "
                  f"train={int(u_tr.sum())}, test={int(u_te.sum())})")
            continue
        u_rows = separability_table(views, y, pids_agg, u_tr, u_te, val_mask=u_va,
                                    seed=seed, dim_labels=dim_labels, pca_views=pca_views,
                                    lowdim_C=probe_c, agg_views=u_agg,
                                    # per-day label: pids_agg is one id PER WINDOW, so the
                                    # Subj* columns must group by the REAL participant instead
                                    # (otherwise they silently duplicate the Win* columns --
                                    # they were identical in 462 of 462 rows of run 18975686).
                                    macro_pids=None if subject_aggregate else pids)
        for r in u_rows:
            r["Unit"] = unit
        rows += u_rows
        if baseline_rows and unit == "last":
            # The supervised baselines were trained upstream on ONE window per participant,
            # so they belong to the 'last' block and are not re-fit per unit.
            for b in baseline_rows:
                b = dict(b); b["Unit"] = "last"; rows.append(b)

    save_table(rows, variant_dir, table_tag=table_tag, unit_note=unit_note)

    test_all = np.where(test_mask)[0]
    idx = _subsample(test_all, y, max_tsne_points, seed)

    # TFD / SFD representations coloured by the depression endpoint, embedded with
    # t-SNE and (optionally) UMAP -- which disentangled space separates the groups?
    # (A) WINDOW level: one point per test window (the original view).
    label_embedding_figure(
        views, y, idx, variant_dir, "hrd_tsne_label.png", "t-SNE", _tsne,
        heading=f"HRD test set  -  {tag}  -  trend / seasonal / amplitude / phase coloured by "
                f"{label_noun} (t-SNE, window level)",
        label_names=label_names, seed=seed)

    # same latent space, coloured by CLINICAL circadian markers -> is it organised by rhythm biology?
    axis_probe = {}
    try:
        clinical_marker_tsne(views["Full [V^(T);V^(S)]"], Xs, idx, variant_dir, bin_minutes,
                             seed, tag, views[COSINOR_VIEW], Xs.shape[-1], paper_cosinor_topk)
        print(f"[rhythm] latent space by clinical markers -> {variant_dir/'hrd_tsne_clinical.png'}")
    except Exception as e:
        print(f"[rhythm] clinical-marker t-SNE skipped ({type(e).__name__}: {e})")
    try:
        # the numeric counterpart of that figure: read the same markers back out of the latent
        axis_probe = rhythm_axis_probe(views["Full [V^(T);V^(S)]"], Xs, test_mask, pids,
                                       bin_minutes, variant_dir, seed, views[COSINOR_VIEW],
                                       Xs.shape[-1], paper_cosinor_topk, table_tag=table_tag,
                                       sensor_cols=sensor_cols)
        for k, v in axis_probe.items():
            print(f"[rhythm] axis probe  {k:<22} {v['metric']} = {v['value']:.3f}")
    except Exception as e:
        print(f"[rhythm] rhythm-axis probe skipped ({type(e).__name__}: {e})")
    # Clock-referenced similarity figure. No longer gated on `windows_anchored`: the phase now
    # comes from each window's real start timestamp in `window_ids`, so unaligned depression
    # windows are placed on the calendar just as correctly as the aligned energy ones.
    circ = {}
    try:
        circ = circadian_similarity_figure(model, X, pids, test_mask, variant_dir,
                                           bin_minutes, tag, table_tag=table_tag,
                                           batch_size=batch_size, window_ids=window_ids,
                                           seed=seed)
        for k, v in circ.items():
            print(f"[circadian] {k:<28} index = {v['circadian_index']:+.3f}  "
                  f"diurnal amp = {v['diurnal_amplitude']:.3f}  "
                  f"rhythmic participants = {v['frac_participants_rhythmic']:.0%}")
    except Exception as e:
        print(f"[circadian] figure skipped ({type(e).__name__}: {e})")

    # The position code itself, independent of any data -- so unlike the figure above it is
    # drawn whether or not the windows are calendar-anchored (an unanchored window still
    # shows periodicity, only its phase is not clock time).
    posgeom = {}
    try:
        posgeom = position_geometry_figure(model, variant_dir, int(X.shape[1]), bin_minutes,
                                           tag, table_tag=table_tag)
        if posgeom:
            print(f"[position] circadian index = {posgeom['circadian_index']:+.3f}")
    except Exception as e:
        print(f"[position] figure skipped ({type(e).__name__}: {e})")
    try:
        label_embedding_figure(
            views, y, idx, variant_dir, "hrd_umap_label.png", "UMAP", _umap,
            heading=f"HRD test set  -  {tag}  -  trend / seasonal / amplitude / phase coloured "
                    f"by {label_noun} (UMAP, window level)",
            label_names=label_names, seed=seed)
    except Exception as e:
        print(f"[rhythm] UMAP figure skipped ({type(e).__name__}: {e}); "
              f"install umap-learn to enable it")

    # (B) SUBJECT level: one point per test subject (subject-mean of each space), so
    # the embedding unit matches the depression label and the subject-level
    # classifier. Kept ALONGSIDE the window-level figures above. Note: with only a
    # handful of subjects these embeddings are sparse -> interpret with care.
    # Skipped for a per-DAY label (subject_aggregate=False): a person spans both classes,
    # so a per-subject mean has no single label -- the window-level figures above are the unit.
    emb_keys = EMBED_KEYS                                # trend / season / amp / phase
    subj_views, y_subj = (_subject_aggregate_views(views, y, pids, test_mask, emb_keys)
                          if subject_aggregate else ({}, np.array([], dtype=int)))
    n_subj = len(y_subj)
    if subject_aggregate and n_subj >= 6 and len(np.unique(y_subj)) > 1:
        sidx = np.arange(n_subj)
        label_embedding_figure(
            subj_views, y_subj, sidx, variant_dir, "hrd_tsne_label_subject.png",
            "t-SNE", _tsne,
            heading=f"HRD test set  -  {tag}  -  SUBJECT-level trend / seasonal / amp / phase "
                    f"by endpoint (t-SNE, N={n_subj} subjects; interpret with care)",
            label_names=label_names, seed=seed)
        try:
            label_embedding_figure(
                subj_views, y_subj, sidx, variant_dir, "hrd_umap_label_subject.png",
                "UMAP", _umap,
                heading=f"HRD test set  -  {tag}  -  SUBJECT-level trend / seasonal / amp / phase "
                        f"by endpoint (UMAP, N={n_subj} subjects; interpret with care)",
                label_names=label_names, seed=seed)
        except Exception as e:
            print(f"[rhythm] subject-level UMAP figure skipped ({type(e).__name__}: {e})")
    elif not subject_aggregate:
        print("[rhythm] subject-level embedding skipped (per-day label -> window is the unit)")
    else:
        print(f"[rhythm] subject-level embedding skipped "
              f"(only {n_subj} test subjects / single class)")

    # machine-readable summary (read by scripts/collect_results.py for the
    # cross-variant rhythm <-> prediction link)
    def _entry(r):
        return {"win_auc": r["Win AUC"], "subj_auc": r["Subj AUC"],
                "subj_f1": r.get("Subj F1"),
                "subj_sensitivity": r.get("Subj Sens"), "subj_specificity": r.get("Subj Spec"),
                "subj_bacc": r.get("Subj BAcc"), "subj_mcc": r.get("Subj MCC")}

    # New: the full ablation, keyed probe-unit -> representation.
    sep_by_unit = {}
    for r in rows:
        sep_by_unit.setdefault(r.get("Unit", "default"), {})[r["Representation"]] = _entry(r)
    # Legacy: a FLAT "separability" block holding just the unit that matches the headline
    # probe. scripts/collect_results.py reads this key to pull the cosinor / seasonal-amp
    # baselines into summary.csv, and every result folder produced so far has this shape --
    # keeping it means old and new runs stay collectable by the same code path.
    legacy_unit = (headline_unit if headline_unit in sep_by_unit
                   else ("last" if "last" in sep_by_unit
                         else (next(iter(sep_by_unit)) if sep_by_unit else None)))
    summary = {
        "backbone": model.net.backbone, "pe": model.net.pe,
        "table_tag": table_tag or None,
        "separability": sep_by_unit.get(legacy_unit, {}),
        "separability_unit": legacy_unit,
        "separability_by_unit": sep_by_unit,
        "rhythm_axis_probe": axis_probe,
        "circadian_similarity": circ,
        "position_geometry": posgeom,
    }
    (variant_dir / "hrd_rhythm.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
