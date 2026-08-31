"""A structured rhythm parameterisation: everything cosinor has, plus what it cannot express.

Cosinor is the strongest baseline in this project on both RQ2 (C=0.921) and RQ3 (AUC=0.649).
It wins because it is a least-squares FIT of the rhythm to the signal -- the efficient
estimator for its model -- at 96 dimensions, while the learned readout is the spectrum of a
non-linear transform of the signal: the same information, estimated with more variance, at
1760 dimensions.

So the feature set below is fitted by the SAME criterion, and cosinor is a strict special case
of it (one harmonic, one channel at a time, all days pooled). Three constructs are added, each
of which cosinor cannot express for a structural reason, not an incidental one:

  waveform shape      harmonics 2..H of the daily cycle. A rest-activity rhythm is not a
                      sinusoid; a single-component fit puts that structure in the residual.
  within-window
  dispersion          each DAY is fitted on its own, giving R (phase concentration) and CV
                      (amplitude stability) across days. Cosinor pools all D days into one
                      fit, and that pooling is exactly what discards this.
  internal phase      the acrophase DIFFERENCE between channels. Cosinor fits each channel
                      independently, so a difference of two of its fits is available only to
                      a probe that can compute a difference of angles -- which a linear probe
                      cannot, because it is not linear in the phases.

Measured facts this design answers to, all at n=24 seeds with architecture-matched controls:

  * contrastive pretraining DEGRADES rhythm structure -- trend tau 0.683 -> 0.529, seasonal
    sigma 0.931 -> 0.896, day-to-day phase concentration R 0.5636 -> 0.4766 (0/24 seeds,
    corrected p = 1.2e-5). The objective scores the window-level marginal spectrum and never
    sees within-window organisation, so that organisation is collateral damage.
  * adding R to the learned readout helps the RANDOM-INIT encoder more than the trained one,
    so R is an architectural property, not a learned one.
  * week-to-week rhythm CHANGE carries no structure a linear probe can extract beyond
    regression to the mean (C - B = +0.012, below the 0.02 threshold fixed in advance).

None of this requires a network. That is the point of this module: it is the GATE. If this
feature set does not beat cosinor on RQ1, RQ2 and RQ3, then no amount of amortising it in an
encoder will, and the design is rejected before any GPU time is spent. If it does beat
cosinor, the network's job is a separate and measurable one -- shrinking these estimates
toward a cohort prior learned from the unlabelled windows, which a per-window least-squares
fit cannot do.
"""
import numpy as np

EPS = 1e-8


def _harmonic_design(n_samples, period, harmonics, extra_period=None):
    """[1 | cos/sin at h=1..H of `period` | cos/sin of `extra_period`], shape (n_samples, p)."""
    t = np.arange(n_samples, dtype=float)
    cols = [np.ones(n_samples)]
    for h in range(1, harmonics + 1):
        w = 2 * np.pi * h * t / period
        cols += [np.cos(w), np.sin(w)]
    if extra_period is not None:
        w = 2 * np.pi * t / extra_period
        cols += [np.cos(w), np.sin(w)]
    return np.stack(cols, axis=1)


def _fit(design, Y):
    """Least-squares coefficients for every column of Y at once.

    The design is identical for every window and channel -- the sampling grid is regular -- so
    the pseudo-inverse is formed once and applied to all of them. That is the same estimator a
    per-series lstsq would give, at a fraction of the cost.
    """
    return np.linalg.pinv(design) @ Y


def _amp_phase(coef, h):
    """Amplitude and phase of harmonic `h` from its (cos, sin) coefficients.

    x ~ a*cos(wt) + b*sin(wt) = A*cos(wt - phi) with A = hypot(a, b), phi = atan2(b, a). The
    convention only has to be consistent, and it is used identically for the window-level and
    the per-day fits, so their phases are comparable.
    """
    a, b = coef[2 * h - 1], coef[2 * h]
    return np.hypot(a, b), np.arctan2(b, a)


def max_harmonics(bins_per_day):
    """Harmonics a day can carry. h = bins/2 is Nyquist, whose sin column is identically zero
    and would make the design rank-deficient, so the last usable harmonic is (bins-1)//2."""
    return max(1, (int(bins_per_day) - 1) // 2)


def structured_features(X, bins_per_day, n_sensors, harmonics=4):
    """(n_windows, p) structured rhythm parameters. See the module docstring for the blocks.

    `X` is (n, T, C_total); only the first `n_sensors` channels are read, so calendar features
    never enter -- they are not a signal whose rhythm means anything.
    """
    B = int(bins_per_day)
    H = min(int(harmonics), max_harmonics(B))
    S = np.asarray(X[:, :, :n_sensors], dtype=float)
    if not np.isfinite(S).all():
        raise ValueError("structured_features: X holds non-finite values; the fit assumes the "
                         "loader has already filled gaps")
    n, T, C = S.shape
    D = max(1, T // B)

    # ---- window-level fit: daily harmonics 1..H plus one circaseptan term -----------------
    win = _harmonic_design(T, B, H, extra_period=float(T))
    cw = _fit(win, S.transpose(1, 0, 2).reshape(T, n * C)).reshape(-1, n, C)
    mesor = cw[0]                                                          # (n, C)
    amp = np.stack([_amp_phase(cw, h)[0] for h in range(1, H + 1)], 1)     # (n, H, C)
    pha = np.stack([_amp_phase(cw, h)[1] for h in range(1, H + 1)], 1)     # (n, H, C)
    wk_a = np.hypot(cw[-2], cw[-1])
    wk_p = np.arctan2(cw[-1], cw[-2])

    # ---- per-day fit: the block cosinor structurally cannot have -------------------------
    day = _harmonic_design(B, B, H)
    Sd = S[:, :D * B].reshape(n, D, B, C).transpose(2, 0, 1, 3).reshape(B, n * D * C)
    cd = _fit(day, Sd).reshape(-1, n, D, C)
    amp_d = np.stack([_amp_phase(cd, h)[0] for h in range(1, H + 1)], 1)   # (n, H, D, C)
    pha_d = np.stack([_amp_phase(cd, h)[1] for h in range(1, H + 1)], 1)
    # A day whose amplitude is numerically zero has no phase to contribute; giving it an
    # arbitrary direction would bias R toward whatever atan2(0, 0) happens to return.
    live = amp_d > EPS
    u = np.where(live, np.exp(1j * pha_d), 0)
    R = np.abs(u.sum(2)) / np.maximum(live.sum(2), 1)                      # (n, H, C)
    cv = amp_d.std(2) / (amp_d.mean(2) + EPS)                              # (n, H, C)

    # ---- internal phase: the acrophase DIFFERENCE between channels -----------------------
    iu, ju = np.triu_indices(C, k=1)
    d1 = pha[:, 0, iu] - pha[:, 0, ju]                                     # (n, n_pairs)

    blocks = [mesor, amp.reshape(n, -1), np.cos(pha).reshape(n, -1), np.sin(pha).reshape(n, -1),
              R.reshape(n, -1), cv.reshape(n, -1), np.cos(d1), np.sin(d1),
              wk_a, np.cos(wk_p), np.sin(wk_p)]
    return np.nan_to_num(np.concatenate(blocks, axis=1).astype(np.float32),
                         nan=0.0, posinf=0.0, neginf=0.0)


def feature_names(n_sensors, bins_per_day, harmonics=4):
    """Column names in the exact order structured_features emits them."""
    H = min(int(harmonics), max_harmonics(bins_per_day))
    C = int(n_sensors)
    iu, ju = np.triu_indices(C, k=1)
    out = [f"mesor_c{c}" for c in range(C)]
    for tag in ("amp", "cos_phase", "sin_phase", "R", "cv"):
        out += [f"{tag}_h{h}_c{c}" for h in range(1, H + 1) for c in range(C)]
    out += [f"cos_dphi_c{i}c{j}" for i, j in zip(iu, ju)]
    out += [f"sin_dphi_c{i}c{j}" for i, j in zip(iu, ju)]
    out += [f"weekly_amp_c{c}" for c in range(C)]
    out += [f"weekly_cos_c{c}" for c in range(C)]
    out += [f"weekly_sin_c{c}" for c in range(C)]
    return out
