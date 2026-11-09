"""Decoding of salience maps (frames, n_bins) into F0 trajectories (in bins of the relative pitch grid)."""
import numpy as np


def weighted_argmax(salience, bps=10):
    """ argmax-local weighted averaging over +- one semitone (PESTO's 'alwa' reduction, from crepe) """
    center = salience.argmax(1)
    idx = np.clip(center[:, None] + np.arange(1 - bps, bps), 0, salience.shape[1] - 1)
    w = np.take_along_axis(salience, idx, 1)
    return (w * idx).sum(1) / np.maximum(w.sum(1), 1e-12)
