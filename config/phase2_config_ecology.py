"""
phase2_config_ecology.py
------------------------

Configuration file for CDR Phase II.3 — Ecological systems
(predator–prey dynamics).

Updated for:
- log-return features
- normalized time
- optional exogenous variables (SOI)
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
# SYSTEM VARIABLES (UPDATED)
# ---------------------------------------------------------

# MUST match columns created in ecology_loader_v2
STATE_VARIABLES = [
    "hare_log_return",
    "lynx_log_return",
    "year_norm",
    # "soi_clean",  # ativar automaticamente se quiser usar variável exógena
]


# ---------------------------------------------------------
# DISCRETIZATION
# ---------------------------------------------------------

# Mantido conservador para evitar sparsidade
BINS_PER_VARIABLE = 3

# Total states ajustado dinamicamente
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

# Novo: aumentar robustez do teste
INJECTION_LENGTH_MULTIPLIER = 10  # aumenta tamanho da trajetória sintética


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
# HOLDOUT TEST (UPDATED)
# ---------------------------------------------------------

# Mantido para compatibilidade, mas runner usará split intercalado
TRAIN_FRACTION = 0.7

MAX_GENERALIZATION_DELTA = 0.12


# ---------------------------------------------------------
# DISCRETIZATION SENSITIVITY (UPDATED)
# ---------------------------------------------------------

# NÃO vamos usar bins diferentes inicialmente (evita sparsidade)
ALT_BINS = 3

# Novo: quantis alternativos para F5 (mais robusto)
ALT_QUANTILES = (0.25, 0.75)

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