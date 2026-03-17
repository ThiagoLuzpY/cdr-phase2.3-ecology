"""
phase2_config_ecology.py
------------------------

Configuration file for CDR Phase II.3 — Ecological systems
(predator–prey dynamics).

This configuration is intentionally lightweight so that the
initial validation run executes quickly.
"""

import os


# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "ecology",
    "lynxhare.csv"
)


# ---------------------------------------------------------
# SYSTEM VARIABLES
# ---------------------------------------------------------

STATE_VARIABLES = [
    "hare",
    "lynx"
]


# ---------------------------------------------------------
# DISCRETIZATION
# ---------------------------------------------------------

# Using small bins to avoid sparse states
BINS_PER_VARIABLE = 3

# Total states:
# 3 x 3 = 9 states
TOTAL_STATES = BINS_PER_VARIABLE ** len(STATE_VARIABLES)


# ---------------------------------------------------------
# EPSILON GRID
# ---------------------------------------------------------

EPS_GRID_SIZE = 21

EPS_MIN = 0.0
EPS_MAX = 0.5


# ---------------------------------------------------------
# INJECTION TEST
# ---------------------------------------------------------

INJECTION_EPS = 0.25

INJECTION_TOL = 0.05


# ---------------------------------------------------------
# CONTROLS
# ---------------------------------------------------------

N_CONTROLS = 6

CONTROL_TYPES = [
    "shuffle_time",
    "circular_shift",
    "block_shuffle",
    "species_swap",
    "transition_randomization",
    "noise_injection"
]


# ---------------------------------------------------------
# HOLDOUT TEST
# ---------------------------------------------------------

TRAIN_FRACTION = 0.7

MAX_GENERALIZATION_DELTA = 0.12


# ---------------------------------------------------------
# DISCRETIZATION SENSITIVITY
# ---------------------------------------------------------

ALT_BINS = 4

MAX_BIN_DELTA = 0.15


# ---------------------------------------------------------
# RESULTS PATH
# ---------------------------------------------------------

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "phase2_ecology"
)


# ---------------------------------------------------------
# RANDOM SEED
# ---------------------------------------------------------

RANDOM_SEED = 42