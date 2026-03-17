"""
controls_phase2_ecology.py
--------------------------

Adversarial controls for Phase II.3 — Ecological dynamics.

Updated for:
- multi-variable state (hare, lynx, year, soi)
- safer transformations
"""

import numpy as np


# ---------------------------------------------------------
# CONTROL 1 — SHUFFLE TIME
# ---------------------------------------------------------

def shuffle_time(X, rng):
    idx = rng.permutation(len(X))
    return X[idx]


# ---------------------------------------------------------
# CONTROL 2 — CIRCULAR SHIFT
# ---------------------------------------------------------

def circular_shift(X, rng):
    shift = rng.integers(1, len(X) - 1)
    return np.roll(X, shift, axis=0)


# ---------------------------------------------------------
# CONTROL 3 — BLOCK SHUFFLE (IMPROVED)
# ---------------------------------------------------------

def block_shuffle(X, rng, block_size=5):
    n = len(X)

    if block_size >= n:
        return shuffle_time(X, rng)

    blocks = [
        X[i:i + block_size]
        for i in range(0, n, block_size)
    ]

    rng.shuffle(blocks)

    return np.vstack(blocks)


# ---------------------------------------------------------
# CONTROL 4 — SPECIES SWAP (SAFE VERSION)
# ---------------------------------------------------------

def species_swap(X, rng):
    """
    Swap ONLY ecological variables (first two columns).

    Assumes:
    col 0 = hare_log_return
    col 1 = lynx_log_return
    """

    if X.shape[1] < 2:
        return X

    X_swapped = X.copy()

    X_swapped[:, 0], X_swapped[:, 1] = X[:, 1], X[:, 0]

    return X_swapped


# ---------------------------------------------------------
# CONTROL 5 — TRANSITION RANDOMIZATION
# ---------------------------------------------------------

def transition_randomization(X, rng):
    n = len(X)
    idx = rng.integers(0, n, size=n)
    return X[idx]


# ---------------------------------------------------------
# CONTROL 6 — NOISE INJECTION (CONTROLLED)
# ---------------------------------------------------------

def noise_injection(X, rng, sigma=0.05):
    """
    Add Gaussian noise, but protect temporal structure variable.

    - noise only on biological variables
    - reduces distortion of time axis (year_norm)
    """

    X_noisy = X.copy()

    noise = rng.normal(0, sigma, size=X_noisy.shape)

    # Apply noise only to first two columns (hare/lynx dynamics)
    if X_noisy.shape[1] >= 2:
        X_noisy[:, 0] += noise[:, 0]
        X_noisy[:, 1] += noise[:, 1]

    return X_noisy


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