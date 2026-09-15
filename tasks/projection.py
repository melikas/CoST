"""The random-projection control: a fixed Gaussian map of the standardised raw window."""
from __future__ import annotations

import numpy as np


class RawProjection:
    """Fixed Gaussian projection with a scaler fitted on training windows only."""

    def __init__(self, n_sensors, seed, width=512):
        self.n_sensors, self.seed, self.width = n_sensors, seed, width

    def fit(self, X):
        F = np.asarray(X, dtype=float)[:, :, :self.n_sensors].reshape(len(X), -1)
        if len(F) == 0 or not np.isfinite(F).all():
            raise ValueError("projection fit requires nonempty finite training windows")
        self.mean_ = F.mean(0)
        self.scale_ = F.std(0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        self.weight_ = np.random.default_rng(self.seed).normal(
            0, 1.0 / np.sqrt(F.shape[1]), (F.shape[1], int(self.width)))
        return self

    def encode(self, X, parts=False, batch_size=None):
        if not hasattr(self, "weight_"):
            raise ValueError("fit the projection on training windows before encoding")
        F = np.asarray(X, dtype=float)[:, :, :self.n_sensors].reshape(len(X), -1)
        if not np.isfinite(F).all():
            raise ValueError("projection inputs must be finite")
        return (((F - self.mean_) / self.scale_) @ self.weight_).astype(np.float32)
