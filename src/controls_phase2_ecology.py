"""
controls_phase2_ecology.py
--------------------------

Adversarial controls for Phase II.3 — Ecological dynamics.

These controls perturb predator–prey time series to destroy
true ecological structure while preserving marginal statistics.

All controls operate on the continuous time-series matrix X
before discretization.
"""

import numpy as np


# ---------------------------------------------------------
# CONTROL 1 — SHUFFLE TIME
# ---------------------------------------------------------

def shuffle_time(X, rng):
    """
    Randomly permute time order.

    Destroys temporal structure completely.
    """

    idx = rng.permutation(len(X))
    return X[idx]


# ---------------------------------------------------------
# CONTROL 2 — CIRCULAR SHIFT
# ---------------------------------------------------------

def circular_shift(X, rng):
    """
    Shift time series by a random offset.
    """

    shift = rng.integers(1, len(X) - 1)
    return np.roll(X, shift, axis=0)


# ---------------------------------------------------------
# CONTROL 3 — BLOCK SHUFFLE
# ---------------------------------------------------------

def block_shuffle(X, rng, block_size=5):
    """
    Shuffle blocks of time series to partially destroy
    temporal dynamics while preserving local structure.
    """

    n = len(X)
    blocks = []

    for i in range(0, n, block_size):
        blocks.append(X[i:i + block_size])

    rng.shuffle(blocks)

    return np.vstack(blocks)


# ---------------------------------------------------------
# CONTROL 4 — SPECIES SWAP
# ---------------------------------------------------------

def species_swap(X, rng):
    """
    Swap predator and prey signals.

    Breaks ecological causality.
    """

    X_swapped = X.copy()
    X_swapped[:, 0], X_swapped[:, 1] = X[:, 1], X[:, 0]

    return X_swapped


# ---------------------------------------------------------
# CONTROL 5 — TRANSITION RANDOMIZATION
# ---------------------------------------------------------

def transition_randomization(X, rng):
    """
    Replace each state with a random draw from the
    empirical distribution.

    Preserves marginal distribution but destroys
    temporal dependence.
    """

    n = len(X)

    idx = rng.integers(0, n, size=n)

    return X[idx]


# ---------------------------------------------------------
# CONTROL 6 — NOISE INJECTION
# ---------------------------------------------------------

def noise_injection(X, rng, sigma=0.05):
    """
    Add small gaussian noise.

    Tests estimator robustness to small perturbations.
    """

    noise = rng.normal(0, sigma, X.shape)

    return X + noise


# ---------------------------------------------------------
# CONTROL REGISTRY
# ---------------------------------------------------------

CONTROL_REGISTRY = {
    "shuffle_time": shuffle_time,
    "circular_shift": circular_shift,
    "block_shuffle": block_shuffle,
    "species_swap": species_swap,
    "transition_randomization": transition_randomization,
    "noise_injection": noise_injection,
}