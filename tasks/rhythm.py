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
from tasks.rq_paths import rq_path
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import normalize as l2normalize
from models.positional_encoding import position_matrix
from baselines.cosinor import N_PARAMS
from tasks.decomposition import RIDGE_ALPHAS, _ridge_fit
from tasks._eval_protocols import (best_threshold, binary_metrics, make_probe,
                                   fit_persubject_probe, participant_aggregate,
                                   persubject_rows,
                                   within_person_macro_metrics)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, silhouette_score

# Okabe-Ito colour-blind-safe palette (widely used in reputable publications)
CLASS_COLORS = ["#0072B2", "#D55E00"]            # depression endpoint: 0 blue, 1 vermillion


# --------------------------------------------------------------------------- #
# 1. Representation extraction
# --------------------------------------------------------------------------- #
def extract_representations(model, X, batch_size=256, pool="mean", season_pool=None):
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
    # `season_pool` comes from the run's config and governs the seasonal half everywhere --
    # here, in model_build.encode_repr, and in the train_hrd headline probe. Passing it is not
    # optional: omitting it built a time-pooled seasonal half here while RQ3 built a spectral
    # one, and both were reported as "Full [V^(T);V^(S)]".
    full = model.encode(X, mode="forecasting", pool=pool, season_pool=season_pool,
                        batch_size=batch_size).squeeze(1)
    # The trend half is time-pooled whatever the seasonal readout is, so its width is
    # component_dims (doubled by 'meanmax', which concatenates mean and max). The halves are
    # NOT equal under a spectral seasonal readout, so the midpoint is the wrong boundary.
    half = model.net.component_dims * (2 if pool == "meanmax" else 1)
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
        # Explicit spectral readout, kept only when the configured one is NOT already
        # spectral -- otherwise it is the same array as `season` and would appear twice in the
        # table under two names.
        "season_spec": (None if season_pool == "spec" else
                        model.encode(X, mode="forecasting", pool=pool, season_pool="spec",
                                     batch_size=batch_size).squeeze(1)[:, half:]),
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
        **({} if rep.get("season_spec") is None
           else {"Season V^(S) spectral": rep["season_spec"]}),
        "Seasonal amp":   rep["amp"],
        "Seasonal phase": rep["phase"],
    }


def cosinor_markers_per_channel(cf, n_channels, top_k=2, bins_per_day=96, tol=0.05):
    """Per-window CIRCADIAN (amplitude, acrophase, MESOR) for EACH channel -- each (N, C).

    The fitted period closest to 24 h is used, and a window whose fits contain no circadian
    period is NaN. This is not a refinement, it is a correctness requirement. The period is
    chosen per window by a Fisher periodogram (baselines/cosinor.py::periodogram), so the
    dominant block is often NOT circadian: measured on the real cohort over 569 windows it is
    within 5% of 24 h for 95.8% of HR windows, 93.1% of Steps, 99.3% of is_asleep -- but only
    64.7% of screen windows, whose dominant period reaches 12 h at the 10th percentile and
    168 h at the 90th.

    Two things broke while the period was free. (1) The angle 2*pi*phi/P lives on the circle of
    THAT window's period, so a 06:00 peak is 1.571 rad under a 24 h fit and 0.224 rad under a
    168 h fit -- the same physical time, a different angle, pooled into one regression.
    (2) rhythm_axis_probe converts the angular error with 12/pi, i.e. 2*pi -> 24 h, which is
    wrong by a factor P/96 for any other period. Fixing the period fixes both at once.

    Windows dropped here need no extra bookkeeping: the probe filters on np.isfinite(target),
    so the surviving count already appears as `n_windows` for every marker.

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
        B = np.stack([cf[:, (c * top_k + k) * N_PARAMS:(c * top_k + k + 1) * N_PARAMS]
                      for k in range(top_k)], axis=1).astype(np.float64)   # (N, top_k, 12)
        per = B[:, :, 0]                                       # fitted period, in bins
        rel = np.where(per > 0, np.abs(per - bins_per_day) / bins_per_day, np.inf)
        k = np.argmin(rel, axis=1)                             # block closest to 24 h
        pick = np.take_along_axis(B, k[:, None, None], axis=1)[:, 0, :]
        good = np.take_along_axis(rel, k[:, None], axis=1)[:, 0] <= tol
        M.append(np.where(good, pick[:, 1], np.nan))           # MESOR
        A.append(np.where(good, pick[:, 2], np.nan))           # Amplitude
        # Acrophase is a peak time in bins since midnight. Dividing by bins_per_day -- never by
        # the fitted period -- puts every window on the SAME 24 h circle, which is what makes
        # the angles comparable and the hours conversion in rhythm_axis_probe correct.
        TH.append(np.where(good, 2 * np.pi * pick[:, 4] / bins_per_day, np.nan))
    return np.stack(A, 1), np.stack(TH, 1), np.stack(M, 1)


def cosinor_markers(cf, n_channels, top_k=2, bins_per_day=96):
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
    A, TH, _ = cosinor_markers_per_channel(cf, n_channels, top_k, bins_per_day)
    # A non-circadian channel is NaN above. It enters the amplitude-weighted circular mean
    # with weight 0, which is exactly right: no circadian rhythm means no circadian amplitude.
    A0, TH0 = np.nan_to_num(A), np.nan_to_num(TH)
    return A0.mean(1), np.angle((A0 * np.exp(1j * TH0)).sum(1))


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
    (rq_path(variant_dir, "frequency_spectrum.csv")).write_text("\n".join(lines), encoding="utf-8")

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
    (rq_path(variant_dir, "frequency_spectrum.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")

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
    (rq_path(variant_dir, "frequency_contrast.csv")).write_text("\n".join(lines), encoding="utf-8")

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
        fig.savefig(rq_path(variant_dir, "frequency_contrast.png"), dpi=200, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
    except Exception as e:
        print(f"[spectrum-contrast] plot skipped ({type(e).__name__}: {e})")
    return {"n_depressed": n_dep, "n_nondepressed": n_non}


def group_spectrum_heatmap(rep, y, mask, variant_dir, seq_len, bin_minutes, dS,
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
        fig.savefig(rq_path(variant_dir, "frequency_heatmap.png"), dpi=200, facecolor="white")
        plt.close(fig)
    except Exception as e:
        print(f"[spectrum-heatmap] skipped ({type(e).__name__}: {e})")
    return {"Ffreq": Ffreq, "dS": dS}


def _shared_pc1(seq):
    """(n_win, T, d) -> (n_win, T): every window projected on ONE direction, so the weeks stay
    comparable and the waveform keeps the branch's dominant variance.

    The previous summary was `seq.mean(-1)`, the mean ACROSS channels. Channels carry their own
    phase and sign, so that average cancels: measured on run 1239199, only 4.1-11.0% of the
    24 h amplitude survived it, i.e. the panel was drawing cancellation residue. The leading
    principal direction of the participant's own windows carries 70-79% of the variance
    instead, and one direction shared by all weeks is what makes first-vs-last a comparison of
    the person rather than of two arbitrary projections.

    The sign of a principal direction is arbitrary; it is fixed by forcing the largest-magnitude
    loading positive (the svd_flip convention), so re-running cannot mirror the figure.
    """
    A = seq.reshape(-1, seq.shape[-1])
    g = A.mean(0)
    v = np.linalg.svd(A - g, full_matrices=False)[2][0]
    v = v * np.sign(v[np.argmax(np.abs(v))])
    return (seq - g) @ v


PHASE_K = 2.079


def _phase_where_defined(a, ph, k=PHASE_K):
    """Phase, blanked wherever the amplitude that carries it is at the noise floor.

    A Fourier phase is undefined at zero amplitude, and `atan2` returns a value regardless, so
    drawing the whole curve gave half the panel to a dense zigzag between +/-pi that carries no
    information -- and, because the wrap is a branch cut rather than a real jump, invited
    reading a phase SHIFT off it.

    THRESHOLD, derived rather than chosen. Under circular-Gaussian noise the FFT magnitude is
    Rayleigh(sigma), whose median is sigma*sqrt(2*ln 2) = 1.1774*sigma and whose 95th
    percentile is sigma*sqrt(2*ln 20) = 2.4477*sigma. Their ratio, 2.079, is therefore the
    multiple of the OBSERVED median at which a bin is significant at the 5% per-bin level --
    and the median is the right reference because most bins in this spectrum are noise, so it
    estimates sigma robustly while the mean would be dragged up by the harmonics themselves.
    On run 1239199 the 24 h bin sits 19.5-61.3x above that floor.
    """
    a = np.asarray(a, float)
    return np.where(a >= k * float(np.median(a)), np.asarray(ph, float), np.nan)


def participant_trajectory_figures(model, X, y, pids, baseline_by_pid, rep, variant_dir,
                                   seq_len, bin_minutes, dS, seed=42, prefer_pids=None):
    """WITHIN-PERSON contrast (not group contrast): for up to 4 representative participants
    -- one per (baseline, endpoint) depression-status trajectory {dep->dep, non->non,
    dep->non, non->dep} -- plot that SAME participant's FIRST vs LAST window, four panels:
    seasonal amplitude and phase (vs period, reusing the already-computed `rep` spectra), the
    TREND sequence V^(T) and the SEASONAL sequence V^(S) vs time-within-window, each projected
    onto the leading principal direction of THAT participant's own windows. The two
    time-domain panels are the waveform view of what the two frequency-domain panels above
    describe as a spectrum; panel 4 is dropped in plain
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
            trend_mean = _shared_pc1(trend_seq.cpu().numpy())            # (n_win, T)
            # V^(S) in the TIME domain -- the counterpart of the trend panel. The amp/phase
            # panels show the SAME seasonal branch in the FREQUENCY domain, so this fourth
            # box is what those spectra actually look like as a waveform across the week.
            season_mean = (_shared_pc1(season_seq.cpu().numpy())
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
                axP.plot(xr, _phase_where_defined(*ap[w]), color=MUTED, lw=0.9, alpha=0.35,
                         marker=".", ms=2.5)
                axT.plot(t_axis, trend_mean[w], color=MUTED, lw=0.9, alpha=0.35)
                if axS is not None:
                    axS.plot(t_axis, season_mean[w], color=MUTED, lw=0.9, alpha=0.35)
            # bold first + last week on top
            axA.plot(xr, ap[0][0], color=COL["first"], lw=2.2)
            axA.plot(xr, ap[-1][0], color=COL["last"], lw=2.2)
            axP.plot(xr, _phase_where_defined(*ap[0]), color=COL["first"], lw=2.2,
                     marker="o", ms=4)
            axP.plot(xr, _phase_where_defined(*ap[-1]), color=COL["last"], lw=2.2,
                     marker="o", ms=4)
            axT.plot(t_axis, trend_mean[0], color=COL["first"], lw=2.2)
            axT.plot(t_axis, trend_mean[-1], color=COL["last"], lw=2.2)
            if axS is not None:
                axS.plot(t_axis, season_mean[0], color=COL["first"], lw=2.2)
                axS.plot(t_axis, season_mean[-1], color=COL["last"], lw=2.2)
            panels = [
                (axA, "period (h)", "seasonal amplitude |F|", True),
                (axP, "period (h)", "seasonal phase ∠F (rad)", True),
                (axT, "time within window (h)", "trend V^(T)  (PC1 of this participant)", False),
            ]
            if axS is not None:
                panels.append((axS, "time within window (h)",
                               "seasonal V^(S)  (PC1 of this participant)", False))
            for ax, xlab, ylab, logx in panels:
                if logx:
                    ax.set_xscale("log"); ax.set_xlim(xr.max(), xr.min())
                    # Named periods, not decades: 10^2 and 10^1 left the one line the figure
                    # exists to show -- 24 h -- unlabelled.
                    tk = [t for t in (168, 24, 12, 8, 6, 4, 2) if xr.min() <= t <= xr.max()]
                    ax.set_xticks(tk); ax.set_xticklabels([f"{t}h" for t in tk], fontsize=8)
                    ax.minorticks_off()
                    for h, cc in ((24, "#c0392b"), (12, "#3b9ad9")):
                        if xr.min() <= h <= xr.max():
                            ax.axvline(h, color=cc, lw=1.0, ls="--", alpha=.7, zorder=0)
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
            fig.savefig(rq_path(variant_dir, f"participant_trajectory_{pid}.png"), dpi=200,
                       bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"[trajectory] {pid} ({tag}, {n_win} weeks) -> participant_trajectory_{pid}.png")
        except Exception as e:
            print(f"[trajectory] {pid} plot skipped ({type(e).__name__}: {e})")


# --------------------------------------------------------------------------- #
# 2. Quantitative separability table
# --------------------------------------------------------------------------- #






def _persubject_row(name, native, note, clf, R, pids, y, train_mask, val_mask, test_mask, seed):
    """One table row from a fitted canonical per-participant probe. Rows are already people,
    so the Win* and Subj* columns are the same measurement and are reported as such."""
    Xte, yte, _ = persubject_rows(R, pids, y, test_mask, "phase" in name.lower())
    prob = clf.predict_proba(Xte)[:, 1]
    thr = 0.5
    if val_mask is not None and int(np.sum(val_mask)) > 0:
        val = val_mask & ~train_mask
        if val.any():
            Xva, yva, _ = persubject_rows(R, pids, y, val, "phase" in name.lower())
            if len(set(yva)) > 1:
                thr = best_threshold(yva, clf.predict_proba(Xva)[:, 1])
    m = binary_metrics(yte, prob, thr)
    return {"Representation": name, "Dim": native, "Thr": float(thr),
            "Win AUC": m["auc_roc"], "Win F1": m["f1"], "Win Acc": m["accuracy"],
            "Win BAcc": m["balanced_accuracy"], "Win MCC": m["mcc"],
            "Win Sens": m["sensitivity"], "Win Spec": m["specificity"],
            "Subj AUC": m["auc_roc"], "Subj F1": m["f1"], "Subj Acc": m["accuracy"],
            "Subj BAcc": m["balanced_accuracy"], "Subj MCC": m["mcc"],
            "Subj Sens": m["sensitivity"], "Subj Spec": m["specificity"],
            "_highdim_note": note}


def separability_table(views, y, pids, train_mask, test_mask, val_mask=None, seed=42,
                       highdim_threshold=2000, highdim_C=0.01, dim_labels=None,
                       pca_views=None, lowdim_C=1.0, persubject=False, macro_pids=None):
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

    `persubject=True` switches every view to the CANONICAL participant-level estimator in
    _eval_protocols (`persubject_rows` + `fit_persubject_probe`): one row per participant
    holding [mean | std] of their windows, with the penalty selected on the participant-
    disjoint validation split. That is the unit design doc 0.1 declares primary for the
    depression endpoint, and E1.2's rule that the penalty is never fixed by hand. RQ3 probes
    through the same two functions, so the two evaluations cannot disagree by protocol.
    `highdim_C` / `lowdim_C` and the mean-only aggregation they went with apply to the
    window-row units (`last`, `all`) only.

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
    use_val = val_mask is not None and int(np.sum(val_mask)) > 0
    n_train = int(np.sum(train_mask))
    rows = []
    yte, pte = y[test_mask], pids[test_mask]
    for name, R in views.items():
        agg_note = None
        if persubject:
            # Canonical participant-level estimator -- the single implementation RQ3 also uses.
            n_pca = int(pca_views.get(name, 0))
            circ = "phase" in name.lower()          # angular columns -> circular mean
            clf = fit_persubject_probe(R, pids, y, train_mask, val_mask, seed,
                                       n_pca=n_pca, circular=circ)
            native = f"PCA{n_pca}" if n_pca else f"2x{dim_labels.get(name, R.shape[1])}"
            note = f"{name} = [mean|std] per participant, C selected on val"
            rows.append(_persubject_row(name, native, note, clf, R, pids, y,
                                        train_mask, val_mask, test_mask, seed))
            continue
        if name in pca_views:
            # PCA fit on TRAIN only (in-pipeline, leakage-safe). At most n_train-1 comps.
            n_comp = max(2, min(int(pca_views[name]), R.shape[1], n_train - 1))
            clf = make_probe("supervised", 1.0, seed, n_comp)
            native = f"PCA{n_comp}"
            note = (f"{name} = PCA {n_comp} comps (fit on train) <- "
                    f"{dim_labels.get(name, R.shape[1])}")
        else:
            highdim = R.shape[1] > highdim_threshold    # amp/phase spectra: p >> n
            C = highdim_C if highdim else lowdim_C       # strong L2 for p>>n, else downstream --probe-c
            clf = make_probe("supervised", C, seed)
            native = dim_labels.get(name, str(R.shape[1]))
            note = (f"{name} = {native} dims kept in full, strong L2 (C={C:g})"
                    if highdim else None)

        clf.fit(R[train_mask], y[train_mask])
        prob = clf.predict_proba(R[test_mask])[:, 1]

        # per-representation decision threshold, tuned on the participant-aggregated
        # validation split (matches the downstream model); 0.5 if no validation split.
        if use_val:
            vp, vl = participant_aggregate(pids[val_mask],
                                           clf.predict_proba(R[val_mask])[:, 1], y[val_mask])
            thr = best_threshold(vl, vp)
        else:
            thr = 0.5

        w = binary_metrics(yte, prob, thr)
        if macro_pids is not None:
            # per-DAY label: average the metric computed inside each participant
            p = within_person_macro_metrics(np.asarray(macro_pids)[test_mask], yte, prob, thr)
        else:
            # participant-level label: pool each person's windows into one sample
            pp, pl = participant_aggregate(pte, prob, yte)
            p = binary_metrics(pl, pp, thr)
        row = {
            "Representation": name,
            "Dim": native,
            "Thr": float(thr),
            "Win AUC": w["auc_roc"], "Win F1": w["f1"], "Win Acc": w["accuracy"],
            "Win BAcc": w["balanced_accuracy"], "Win MCC": w["mcc"],
            "Win Sens": w["sensitivity"], "Win Spec": w["specificity"],
            "Subj AUC": p["auc_roc"], "Subj F1": p["f1"], "Subj Acc": p["accuracy"],
            "Subj BAcc": p["balanced_accuracy"], "Subj MCC": p["mcc"],
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
    (rq_path(variant_dir, f"{stem}.csv")).write_text(csv, encoding="utf-8")

    # Markdown
    md = ("| " + " | ".join(cols) + " |\n"
          + "| " + " | ".join("---" for _ in cols) + " |\n"
          + "\n".join("| " + " | ".join(cell_of(r, c) for c in cols) + " |" for r in rows))
    if note_line:
        md += f"\n\n*{note_line}*\n"
    (rq_path(variant_dir, f"{stem}.md")).write_text(md, encoding="utf-8")

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
    fig.savefig(rq_path(variant_dir, f"{stem}.png"), dpi=200, bbox_inches="tight")
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
    fig.savefig(rq_path(variant_dir, fname), dpi=200, bbox_inches="tight")
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
    amp, acro = cosinor_markers(cf, n_sensors, top_k,
                                int(round(24 * 60 / bin_minutes)))   # clock-anchored
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
    fig.savefig(rq_path(variant_dir, fname), dpi=200, bbox_inches="tight")
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


def _clock_variance(u, tod, pid_of_window):
    """How much of the per-bin embedding's variance is explained by CLOCK TIME, and how much
    by WHO the person is. BEYOND the paper (see docs/METHODOLOGY.md).

    WavesFM never needed this: it had one model that did not degenerate. Our runs produce two
    OPPOSITE degeneracies that a similarity plot cannot tell apart, because both draw a clean
    curve:
      * collapse to one direction -- every bin's embedding is nearly identical (cos ~ 1), so
        the "rhythm" is a numerical wobble that autoscaling inflates to full panel height;
      * collapse to one frequency -- the embedding is a perfect function of clock time and
        NOTHING else, identical for every participant, which draws a flawless sinusoid while
        carrying zero information about anyone.
    Only the second number separates them: a representation that is useful for RQ1-RQ3 must
    vary with the clock AND still differ between people at the same hour.

    Each fraction is the share of total variance explained by that factor ALONE (a one-way
    sum of squares). They are NOT orthogonal and do not sum to 1 -- clock and participant are
    correlated whenever people wear the device on different schedules -- so read each against
    zero, never as a partition.
    """
    flat = u.reshape(-1, u.shape[-1])
    g = flat.mean(0)
    ss_tot = float(((flat - g) ** 2).sum())
    if ss_tot <= 0:
        return {"clock_var_frac": None, "participant_var_frac": None}

    def explained(key):
        ss = 0.0
        for k in np.unique(key):
            m = key == k
            ss += m.sum() * float(((flat[m].mean(0) - g) ** 2).sum())
        return round(ss / ss_tot, 5)

    pw = np.repeat(np.asarray(pid_of_window), u.shape[1])
    return {"clock_var_frac": explained(tod.ravel()),
            "participant_var_frac": explained(pw)}


def _single_mode_r2(s, bpd):
    """R^2 of s(lag) against ONE cosine at exactly 1 cycle/day. ~1.0 means the representation
    is a single Fourier mode: a perfect clock, and nothing else."""
    if len(s) < 2 * bpd or bpd <= 0:
        return None
    lag = np.arange(len(s), dtype=float)
    A = np.stack([np.ones_like(lag), np.cos(2 * np.pi * lag / bpd),
                  np.sin(2 * np.pi * lag / bpd)], 1)
    fit = A @ np.linalg.lstsq(A, s, rcond=None)[0]
    ss = float(((s - s.mean()) ** 2).sum())
    return round(1 - float(((s - fit) ** 2).sum()) / ss, 5) if ss > 0 else None


def _period_power(s, bin_minutes):
    """Share of s(lag)'s periodic power sitting at each period, and the 24 h / 12 h shares.

    `s(lag)` is already the mean cosine between two bins `lag` apart, so its own spectrum says
    which periods the representation REPEATS at. The window is 168 h, so period 168/k hours
    lands on integer bin k: 24 h is bin 7 and 12 h is bin 14, both exact -- no interpolation.

    Bin k=1 (the 168 h "weekly" line) is deliberately NOT reported. One cycle inside a one-week
    window is a trend, not a rhythm: nothing distinguishes it from a slow drift, and reporting
    it as a circaseptan period would be an artefact of the window length. The paper reads
    circaseptan structure the other way, as a flattening of the Sat/Sun peaks, which stays
    visible in the anchored columns.

    Power is normalised by the total EXCLUDING the DC bin, so the numbers are shares of the
    part of s(lag) that actually oscillates, not of its mean level.
    """
    f = np.abs(np.fft.rfft(s - s.mean()))
    tot = float(f[1:].sum())
    if tot <= 0:
        return None, None, None, None
    per = np.array([float("inf")] + [len(s) * bin_minutes / 60.0 / k
                                     for k in range(1, len(f))])
    frac = f / tot
    at = lambda h: int(round(len(s) * bin_minutes / 60.0 / h))
    k24, k12 = at(24.0), at(12.0)
    p24 = round(float(frac[k24]), 5) if 1 <= k24 < len(f) else None
    p12 = round(float(frac[k12]), 5) if 1 <= k12 < len(f) else None
    return per, frac, p24, p12


def circadian_similarity_figure(model, X, pids, mask, variant_dir, bin_minutes, tag,
                                table_tag="", batch_size=256, window_ids=None,
                                n_sensors=None, anchors=(0, 9), max_windows=400,
                                n_anchor=16, seed=42):
    """WavesFM Fig. 14 reproduced on this cohort, with its layout and its FIXED axes.

    The paper compares the INPUT of its temporal encoder (per-bin Stage I embeddings) against
    that encoder's OUTPUT, on one shared y axis, to show what the temporal model adds. The
    same three rows here: the raw sensor bins that enter the encoder, then the two branches
    that come out of it.

    Columns are the paper's Fig. 14 exactly:
      (a) every intra-window pair against the TIME DISTANCE between its two bins;
      (b), (c) every bin against its position in the WEEK, referenced to that window's own
          Monday `anchors[0]`:00 and `anchors[1]`:00 bin. Two anchors, not one, because that
          is the paper's own control: a rhythm visible under only one reference hour is a
          property of the reference, not of the representation.

    Vectors are compared after their per-window temporal mean is removed, and every panel is
    drawn on a FIXED [-1, 1] cosine axis. This is the single most important
    difference from the previous version, which let matplotlib autoscale each panel: a
    diurnal swing of 0.006 and one of 1.99 were then drawn at identical visual height, so
    seeds differing by a factor of 500 looked alike and a fully collapsed representation
    looked like a textbook circadian rhythm. The paper fixes its own axis at 0.2-1.0 for
    exactly this reason.

    `n_sensors` selects the raw row's channels; the appended clock features are excluded
    because they are deterministic 24 h cosines and would draw a perfect rhythm by
    construction. Without it the raw row is skipped rather than silently drawn from them.
    """
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
    seqs = {}
    if n_sensors:
        seqs["Raw sensor bins\n(model INPUT)"] = Xs[:, :, :int(n_sensors)].astype(np.float32)
    # The backbone output, i.e. the representation BEFORE the trend/seasonal split. This is
    # what makes the figure answer the paper's actual question on our architecture: WavesFM
    # contrasts the input of its temporal encoder with that encoder's output, and the step
    # this project contributes is the DISENTANGLING, not the backbone. Without this row the
    # figure collapses two stages into one, so a degenerate V^(S) cannot be attributed --
    # a backbone that already lost the structure and an SFD that destroyed it look identical.
    # Free: `tcn_output=True` is the encoder's own early exit, no second forward pass of the
    # disentanglers and no change to the model.
    hb_l, tr_l, se_l = [], [], []
    split = bool(getattr(model.net, "disentangle", True))
    with torch.no_grad():
        for i in range(0, len(Xs), batch_size):
            xb = torch.from_numpy(Xs[i:i + batch_size]).float().to(model.device)
            if split:
                hb_l.append(model.net(xb, tcn_output=True).float().cpu().numpy())
            tr, se = model.net(xb)
            tr_l.append(tr.float().cpu().numpy())
            if se is not None:
                se_l.append(se.float().cpu().numpy())
    model.net.train(org)
    # Only when there IS a split: with disentangle=False the backbone output already IS the
    # single representation (models/encoder.py), so the row would be an exact duplicate.
    if hb_l:
        seqs["Backbone h\n(BEFORE disentangling)"] = np.concatenate(hb_l, 0)
    seqs[("DSSL trend V^(T)\n(AFTER)" if split else "DSSL representation\n(no disentangling)")
         ] = np.concatenate(tr_l, 0)
    if se_l:
        seqs["DSSL season V^(S)\n(AFTER)"] = np.concatenate(se_l, 0)

    T = Xs.shape[1]
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    out = {}
    nrow, ncol = len(seqs), 2 + len(anchors)     # +1 for the period spectrum
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.9 * ncol, 3.5 * nrow), squeeze=False)

    for r, (name, S) in enumerate(seqs.items()):
        # The cosine is taken on the TIME-CENTRED sequence. For the seasonal branch the
        # temporal mean IS the k=0 coefficient of an irfft, and the seasonal loss and readout
        # both select f = (1, D, 2D, 3D, 4D) under the guard `0 < i` (cost.py:688), so bin 0
        # is never trained -- it is a free parameter. Measured on run 1239199 it is 7-20x
        # longer than the oscillating part, which pinned every seasonal cosine above 0.97,
        # drew the row as a flat line at the top of the panel, and raised COLLAPSED on a
        # branch that carries 20-27% of its power at 24 h. Centring removes the untrained
        # offset from the comparison; `dc_ac` keeps its size visible as a number, so nothing
        # is hidden -- the level is reported instead of silently dominating the geometry.
        mu = S.mean(1, keepdims=True)
        dc_ac = float(np.linalg.norm(mu, axis=-1).mean()
                      / (np.sqrt(((S - mu) ** 2).sum(-1)).mean() + 1e-12))
        Sc = S - mu
        u = Sc / (np.linalg.norm(Sc, axis=-1, keepdims=True) + 1e-8)
        n = len(u)

        # ---- (a) density vs time distance (paper Fig. 14a) ---------------------------
        a = rng.integers(0, T, size=(n, n_anchor))
        sim_a = np.einsum('nad,ntd->nat', u[np.arange(n)[:, None], a], u)
        lag_a = np.abs(a[:, :, None] - np.arange(T)[None, None, :]) * bin_minutes / 1440.0
        ax = axes[r][0]
        ax.hexbin(lag_a.ravel(), sim_a.ravel(), gridsize=90, cmap="Reds", mincnt=1,
                  bins="log", extent=(0, T * bin_minutes / 1440.0, -1, 1))
        for d in range(1, 8):
            ax.axvline(d, color="0.55", lw=0.6, alpha=0.7)
        ax.set_xlabel("time distance (days)", fontsize=9)
        if r == 0:
            ax.set_title("(a) all intra-window pairs", fontsize=9.5, pad=20)

        # ---- (b..) density across the week, one column per anchor (paper Fig. 14b/c) --
        prof0 = None
        for c, ah in enumerate(anchors):
            anchor_bow = ah * 60 // bin_minutes                 # Monday, ah:00
            ai = np.argmax(bow == anchor_bow, axis=1)           # a 7-day window hits it once
            # The anchor bin is KEPT, exactly as in the paper: every window is referenced to
            # the same bin-of-week, so masking it would punch a hole at the reference point.
            # Its cos = 1 IS the reference, and the daily peak there is the finding.
            sim_b = np.einsum('nd,ntd->nt', u[np.arange(n), ai], u)
            ax = axes[r][1 + c]
            ax.hexbin(bow.ravel(), sim_b.ravel(), gridsize=90, cmap="Reds", mincnt=1,
                      bins="log", extent=(0, 7 * bpd, -1, 1))
            prof = (np.bincount(bow.ravel(), sim_b.ravel(), 7 * bpd)
                    / np.maximum(np.bincount(bow.ravel(), minlength=7 * bpd), 1))
            ax.plot(np.arange(7 * bpd), prof, color="#3b9ad9", lw=1.4, label="weighted avg")
            for d in range(1, 7):
                ax.axvline(d * bpd, color="0.55", lw=0.6, alpha=0.7)
            ax.set_xticks(np.arange(7) * bpd + bpd // 2)
            ax.set_xticklabels(days, fontsize=8)
            if r == 0:
                ax.set_title(f"({chr(98 + c)}) referenced to Mon {ah:02d}:00", fontsize=9.5, pad=20)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="lower right")
            if prof0 is None:
                prof0 = prof

        s = _sim_vs_distance(Sc)

        # ---- (last) WHICH periods does it repeat at? --------------------------------
        # Panel (a) shows that something recurs daily; this says whether 24 h is actually
        # the dominant period or whether the recurrence sits somewhere else -- which is the
        # documented artefact signature (banding at the window length rather than at 24 h).
        per, frac, p24, p12 = _period_power(s, bin_minutes)
        ax = axes[r][ncol - 1]
        if per is not None:
            keep = slice(1, len(frac))
            ax.plot(per[keep], frac[keep], color="0.35", lw=1.0)
            ax.fill_between(per[keep], 0, frac[keep], color="0.75", alpha=.6)
            for h, cc in ((24.0, "#c0392b"), (12.0, "#3b9ad9")):
                ax.axvline(h, color=cc, lw=1.2, ls="--")
                ax.text(h, 1.0, f" {h:.0f}h", color=cc, fontsize=8, va="top", ha="left")
            ax.set_xscale("log")
            ax.set_xlim(4, 168)
            ax.set_xticks([6, 12, 24, 48, 168])
            ax.set_xticklabels(["6h", "12h", "24h", "48h", "1wk"], fontsize=8)
        ax.set_ylim(0, 1)                       # a SHARE -- fixed, like every other panel
        ax.set_xlabel("period", fontsize=9)
        if r == 0:
            ax.set_title(f"({chr(98 + len(anchors))}) periods present", fontsize=9.5, pad=20)
        ax.text(0.0, 1.012,
                ("24 h share " + ("n/a" if p24 is None else format(p24, ".1%"))
                 + "   |   12 h share " + ("n/a" if p12 is None else format(p12, ".1%"))),
                transform=ax.transAxes, fontsize=8.5, color="0.35", va="bottom")

        day_prof = prof0.reshape(7, bpd).mean(0)
        amp = float(day_prof.max() - day_prof.min())
        r2 = _single_mode_r2(s, bpd)
        stat = {"diurnal_amplitude": amp,
                "circadian_index": float(s[bpd] - 0.5 * (s[bpd // 2] + s[bpd + bpd // 2])),
                "mean_similarity": float(s.mean()),
                # ~1.0 => a perfect clock and nothing else (see _single_mode_r2)
                "single_mode_r2": r2,
                # Shares of s(lag)'s oscillating power at the two biological periods a 168 h
                # window can resolve. High 24 h share says the representation REPEATS daily;
                # it does NOT say the daily pattern is the participant's own -- read it next
                # to participant_var_frac, which is what separates biology from a clock.
                "power_24h": p24, "power_12h": p12,
                "n_windows": int(n),
                "s_curve": [round(float(x), 5) for x in s], "bins_per_day": int(bpd)}
        stat["dc_ac_ratio"] = round(dc_ac, 4)
        stat.update(_clock_variance(
            S / (np.linalg.norm(S, axis=-1, keepdims=True) + 1e-8), bow % bpd, pm))
        # Two opposite degeneracies, both of which used to be drawn as a healthy rhythm.
        # Every LEARNED row is judged -- the backbone's as well, since localising the collapse
        # to the backbone or to the disentangler is the reason that row exists. Only the raw
        # row is exempt: it is the model's INPUT, whatever the sensors happened to record,
        # which on a strongly diurnal channel is legitimately close to a single mode.
        #
        # The first test used to be `mean_similarity > 0.95`, i.e. "every vector points the
        # same way". Centring makes that quantity ~0 by construction, so it is now dead as a
        # test -- and it was never the right one: a seasonal branch with a DC/AC of 17.5x
        # tripped it while carrying 20-27% of its power at 24 h. A large offset is reported
        # (`dc_ac_ratio`), not judged. What actually makes a row useless is carrying no
        # rhythm once that offset is removed, which is what NO RHYTHM tests.
        #
        # THRESHOLD. `power_24h` is a share of the magnitude spectrum of s(lag) excluding DC.
        # With no structure the L/2 non-DC bins are exchangeable, so each has expected share
        # 1 / (L/2) -- 0.30% at L = 672. Three times that null is the floor here: the flat
        # control measures 0.3% and a real seasonal branch 13.7-27.2%, two orders apart, so
        # the exact multiple is not load-bearing; it only has to sit inside that gap.
        null_share = 2.0 / max(len(s), 2)
        flag = "" if name.startswith("Raw") else (
            "NO RHYTHM (flat once the offset is removed)"
            if (p24 is not None and p24 < 3 * null_share) else
            "COLLAPSED to one frequency" if (r2 or 0) > 0.95 and amp > 1.5 else "")
        stat["degenerate"] = flag
        out[name.replace("\n", " ")] = stat

        pv = stat.get("participant_var_frac")
        axes[r][0].set_ylabel(name + "\ncosine (mean-removed)", fontsize=9)
        # Above the row, not inside it: at the bottom of the axes this line sat on top of the
        # very density it describes, and on a collapsed row that density is exactly there.
        axes[r][0].text(0.0, 1.012,
                        "diurnal swing " + format(amp, ".3f")
                        + "   |   DC/AC " + format(dc_ac, ".1f") + "x"
                        + "   |   between-person var "
                        + ("n/a" if pv is None else format(pv, ".1%"))
                        + (("   |   " + flag) if flag else ""),
                        transform=axes[r][0].transAxes, fontsize=8.5,
                        color="#b30000" if flag else "0.35", va="bottom")
        for c in range(ncol):
            if c < ncol - 1:                           # the spectrum panel is a share, 0..1
                axes[r][c].set_ylim(-1, 1)             # FIXED axis -- the whole point
            axes[r][c].grid(alpha=0.15)

    fig.suptitle(f"Circadian structure of the representation  -  {tag}"
                 + (f"  ({table_tag})" if table_tag else "")
                 + "\ncosine axis fixed to [-1, 1] in every panel, so rows and runs are "
                   "directly comparable", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    stem = f"circadian_similarity_{table_tag}" if table_tag else "circadian_similarity"
    fig.savefig(rq_path(variant_dir, f"{stem}.png"), dpi=180, bbox_inches="tight")
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
    fig.savefig(rq_path(variant_dir, f"{stem}.png"), dpi=200, bbox_inches="tight")
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
    # `seed` is retained in the signature for call-site compatibility but no longer changes the
    # result: the penalty is now chosen on a deterministic inner GroupKFold rather than on a
    # random quarter, so this probe is reproducible across seeds by construction.
    for tr, te in cv.split(F, Y[:, 0], groups):
        gtr = groups[tr]
        n_inner = int(min(4, len(np.unique(gtr))))
        if len(alphas) > 1 and n_inner >= 2:
            # lambda* is chosen on an inner GroupKFold and the criterion is SUMMED over its
            # folds, not read off a single random quarter of the participants.
            #
            # A single draw was the previous rule, and at this cohort size it is unusable: the
            # quarter is ~9 participants, so lambda* is picked from ~9 people and swings wildly
            # between outer folds. Measured on run 1608369, one marker selected
            # [5.0, 1000.0, 50.0, 1000.0, 0.01] across its five folds -- five orders of
            # magnitude -- and the fold that drew 0.01 fits essentially unpenalised, which sent
            # that marker's pooled R2 to -27.8. Across the 32-dim runs 22% of marker-seed cells
            # landed below zero, the worst at -705. A correctly penalised ridge cannot do much
            # worse than predicting the mean (R2 ~ 0), so those values described the SELECTION,
            # not the representation, and they are what the RQ1 comparison between
            # configurations was resting on.
            #
            # Averaging over inner folds costs almost nothing -- `_ridge_fit` pays the Gram
            # eigendecomposition once per inner fold and reads the entire grid off it -- and it
            # is applied identically to the encoder, the raw-PCA reference and the random-init
            # control, so no rung gains an advantage from the change.
            err = np.zeros(len(alphas))
            for f_i, s_i in GroupKFold(n_splits=n_inner).split(F[tr], Y[tr][:, 0], gtr):
                inner = _ridge_fit(F[tr][f_i], Y[tr][f_i])
                for j, a in enumerate(alphas):
                    r = inner(a, F[tr][s_i]) - Y[tr][s_i]
                    err[j] += float(np.sqrt((r ** 2).mean()) + np.abs(r).mean())
            a_star = float(alphas[int(np.argmin(err))])
        elif len(alphas) > 1:
            # Fewer than two participants to split on: no honest selection is possible.
            degenerate = True
            inner = _ridge_fit(F[tr], Y[tr])
            err = [float(np.sqrt((r ** 2).mean()) + np.abs(r).mean())
                   for r in (inner(a, F[tr]) - Y[tr] for a in alphas)]
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
    AMP, ACRO, MESOR = cosinor_markers_per_channel(
        np.asarray(cf)[mask], n_sensors, top_k, int(round(24 * 60 / bin_minutes)))
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
                     f"the same rule as E1.2 (DSSL grid {alphas[0]:g} ... {alphas[-1]:g}, "
                     f"minimising RMSE + MAE), never hand-set. lambda* used across folds and "
                     f"markers: {', '.join(f'{a:g}' for a in used)}.*")
        rq_path(variant_dir, f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 5. Driver
# --------------------------------------------------------------------------- #
def run_hrd_rhythm_analysis(model, X, y, pids, train_mask, test_mask, variant_dir,
                            seq_len, bin_minutes, sensor_cols=None, seed=42,
                            label_names=None, max_tsne_points=3000, batch_size=256,
                            val_mask=None, baseline_rows=None, extra_views=None,
                            window_ids=None, pool="mean", season_pool=None,
                            probe_sel=None, probe_c=1.0, paper_cosinor_topk=2,
                            baseline_by_pid=None, subject_aggregate=True,
                            label_noun="endpoint", table_tag="", headline_unit="last"):
    """Produce the HRD test-set rhythm figures + table + JSON inside `variant_dir`.

    ``subject_aggregate`` (default True, the depression behaviour): the label is CONSTANT
    per participant, so figures/metrics are aggregated to one point per subject. Set it
    False for a per-DAY label (emotional energy): each WINDOW is its own unit -- a synthetic
    unique id per window makes the per-subject aggregations collapse to a pooled per-day
    contrast, the per-person averaging and the subject-level embeddings are skipped. ``label_noun`` only relabels the figure headings (e.g. 'emotional energy')."""
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

    rep = extract_representations(model, X, batch_size=batch_size, pool=pool,
                                  season_pool=season_pool)
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
            (rq_path(variant_dir, "paper_cosinor.FAILED.txt")).write_text(   # SLURM stdout log
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
        group_spectrum_heatmap(rep, y, test_mask, variant_dir, seq_len, bin_minutes,
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
    # The RQ3 utility ladder's two CONTROLS, scored on exactly the rows of this table so the
    # two objects answer the same question with the same probe. Without them the separability
    # table ranks only representations that were all produced by a trained, disentangled
    # encoder -- so it can say which branch is best, but not whether TRAINING or DISENTANGLING
    # bought anything, which is what RQ1 and RQ3 are actually asking.
    for _n, _v in (extra_views or {}).items():
        views[_n] = np.asarray(_v)
        dim_labels[_n] = str(views[_n].shape[1])

    PCA_TARGET = 20
    pca_views = {}
    # Per-participant averaging of the noisy Seasonal amp/phase views is decided PER UNIT in
    # the probe-unit loop below (`u_agg`), which is the only place it is read.
    if Ffreq:
        views["Seasonal amp (PCA)"] = rep["amp"]
        views["Seasonal phase (PCA)"] = rep["phase"]
        dim_labels["Seasonal amp (PCA)"] = ffts
        dim_labels["Seasonal phase (PCA)"] = ffts
        pca_views.update({"Seasonal amp (PCA)": PCA_TARGET,
                          "Seasonal phase (PCA)": PCA_TARGET})
    # Dimensionality-matched counterparts of the LOW-dim views as well. Without them the table
    # compares representations that differ in WIDTH as much as in content (320 vs 160 vs 96
    # against only ~58 training participants), so "Full beats Cosinor" could just mean "320
    # parameters beat 96". These rows put every representation at the same PCA_TARGET with the
    # same penalty, which isolates content from capacity. The raw rows stay, so the headline
    # 'Full' number remains comparable to metrics.json.
    # Assignment is a REFERENCE to the existing array, not a copy: the PCA happens inside the
    # probe pipeline at fit time, so these rows cost no extra memory.
    # Only views WIDER than the target get a twin. Below it the "reduction" is a no-op --
    # n_comp is clamped to the view's own width -- so the twin would be the same row printed
    # twice under a name promising a dimensionality-matched control. It also keeps the
    # 1-column Majority view away from PCA, which cannot run on it at all.
    for _base in ["V (encoder pre-decomp)", "Full [V^(T);V^(S)]", "Trend V^(T)",
                  "Season V^(S)", COSINOR_VIEW] + list(extra_views or {}):
        if _base in views and np.asarray(views[_base]).shape[1] > PCA_TARGET:
            _pn = f"{_base} (PCA)"
            views[_pn] = views[_base]
            dim_labels[_pn] = dim_labels.get(_base, str(np.asarray(views[_base]).shape[1]))
            pca_views[_pn] = PCA_TARGET
    # A per-DAY label must never have a person's windows averaged together (it would destroy
    # the label); the loop below enforces that by giving the 'all' unit an empty u_agg.
    #
    # Historical note kept because the bug is easy to reintroduce: the classification must be
    # SCORED on the same unit the probe was TRAINED on. It once trained on last-week windows
    # and scored on every window, which made the 'Full' row disagree with metrics.json. Each
    # unit below now derives its own train/test masks, so the mismatch cannot recur. The
    # spectra / embeddings further down intentionally still use ALL windows (richer per-pid
    # mean) -- that is a different computation, not the probe.

    # --- probe-unit ablation -------------------------------------------------------------
    # The incoming train/val masks are already restricted to whatever unit the headline probe
    # used, so we recover the participant SETS from them and rebuild the masks per unit. The
    # last-window mask is derived here from `pids` rather than taken from `probe_sel`, because
    # probe_sel is all-True when the caller ran --probe-unit all.
    #   all         every window is a probe sample.
    #   last        one window per participant (the most recent).
    #   persubject  one row per participant holding [mean | std] of that participant's
    #               windows -- the unit design doc 0.1 declares primary for this label, built
    #               by the single canonical implementation in _eval_protocols that the
    #               train_hrd.py headline probe and the RQ3 ladder also call. The penalty is
    #               selected on the participant-disjoint validation split (E1.2: never fixed
    #               by hand). For the angular phase views the mean half uses the circular
    #               mean; see persubject_rows for the caveat on their std half.
    # For a per-DAY label (emotional energy) the label varies within a participant, so
    # averaging a person's windows would destroy it -- 'persubject' is skipped there.
    def _pid_set(m):
        return set(pids[m]) if m is not None and int(np.sum(m)) else set()

    last_sel = np.zeros(len(pids), bool)
    for _p in np.unique(pids):
        last_sel[np.where(pids == _p)[0][-1]] = True

    tr_pids, va_pids = _pid_set(train_mask), _pid_set(val_mask)
    # 'all' is DROPPED for a participant-level label. Design doc 0.1: ~26 windows per person
    # all carry the same value, so it is pseudo-replication -- long-record participants
    # dominate the fit and the window-level intervals are optimistically tight. Only the two
    # units the design actually sanctions remain: 'persubject' (primary) and 'last'
    # (sensitivity check). It stays for a per-DAY label, where it is the ONLY valid unit --
    # see the else branch.
    units = ["last", "persubject"] if subject_aggregate else ["all"]
    if subject_aggregate:
        unit_note = ("units: last = one (most recent) window per participant (sensitivity "
                     f"check); persubject = one row per participant, each view averaged "
                     f"over all their windows -- the PRIMARY unit for this participant-level "
                     f"label. '{COSINOR_VIEW}' is the one view NOT averaged here, and its two "
                     "rows are therefore IDENTICAL across the two units: "
                     "paper_cosinor_features already collapsed it to one vector per "
                     "participant upstream (population-mean cosinor, phases averaged as "
                     "angles), so every window of a person carries the same vector and there "
                     "is nothing left to average. Read its 'last' and 'persubject' rows as one "
                     "number reported twice, NOT as evidence that the unit choice does not "
                     "matter. 'all' is omitted on purpose: every window of a person carries "
                     "the same label, so it is pseudo-replication (design doc 0.1). "
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
        if unit in ("all", "persubject"):
            # 'persubject' aggregates each person's windows itself, so it must see ALL of
            # them: restricting to `last_sel` would leave one window per person and collapse
            # the [mean|std] row's std half to zero.
            u_tr = np.isin(pids, list(tr_pids))
            u_va = np.isin(pids, list(va_pids)) if va_pids else None
            u_te = test_mask
        else:                                            # 'last': one window per participant
            u_tr = np.isin(pids, list(tr_pids)) & last_sel
            u_va = (np.isin(pids, list(va_pids)) & last_sel) if va_pids else None
            u_te = test_mask & last_sel
            # Aggregate every view EXCEPT the Cosinor one. Its 96-dim vector interleaves
            # linear parameters (MESOR, Amplitude, p-value, ...) with three angular ones
            # (Acrophase, Orthophase, Bathyphase) inside each 12-parameter block, and the
            # circular branch in separability_table keys off the view NAME, so it would
            # arithmetic-average those angles (mean of 350 deg and 10 deg -> 180, not 0).
            # It needs no averaging here anyway: paper_cosinor_features was called with
            # `pids`, so it ALREADY collapsed each participant to one vector upstream, with
            # the phase columns averaged as angles (baselines/cosinor.py::_aggregate_to_subject
            # -- the correct circular mean, which is exactly what this branch cannot do).
            # Consequence, verified on run 19937323 across all 12 variant-seeds: the Cosinor
            # 'last' and 'persubject' rows come out bit-identical, because every window of a
            # person already carries the same vector. That is one number printed twice, not a
            # robustness result -- the footnote now says so, because the earlier wording
            # ("stays the last-window fit") described something the code does not do.
            # startswith, not set-difference: this must also exclude the "(PCA)" counterpart
            # of the cosinor view, whose vector mixes angular and linear parameters just the
            # same and so must not be arithmetically averaged either.

        if int(u_tr.sum()) < 4 or int(u_te.sum()) < 2 or len(np.unique(y[u_tr])) < 2:
            print(f"[rhythm] probe-unit '{unit}' skipped (too few rows: "
                  f"train={int(u_tr.sum())}, test={int(u_te.sum())})")
            continue
        u_rows = separability_table(views, y, pids_agg, u_tr, u_te, val_mask=u_va,
                                    seed=seed, dim_labels=dim_labels, pca_views=pca_views,
                                    lowdim_C=probe_c, persubject=(unit == "persubject"),
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
                                           n_sensors=n_sensors, seed=seed)
        for k, v in circ.items():
            _pv = v.get('participant_var_frac')
            print(f"[circadian] {k:<26} swing = {v['diurnal_amplitude']:.4f}  "
                  f"mean cos = {v['mean_similarity']:+.4f}  "
                  f"1-mode R2 = {(v.get('single_mode_r2') or float('nan')):.3f}  "
                  f"between-person var = " + ('n/a' if _pv is None else format(_pv, '.1%'))
                  + (('  <<< ' + v['degenerate']) if v.get('degenerate') else ''))
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
    (rq_path(variant_dir, "hrd_rhythm.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
