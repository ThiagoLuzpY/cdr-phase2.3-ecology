"""
ecology_loader.py
-----------------

Loader for ecological predator–prey datasets (enhanced version).

Enhancements:
- Uses maximum available data
- Adds log-returns (stationarity)
- Adds normalized year
- Optionally includes external variables (e.g., SOI)

Compatible with existing CDR pipeline.
"""

import os
import pandas as pd
import numpy as np


def load_lynx_hare_dataset(csv_path):
    """
    Load and enrich the lynx-hare predator–prey dataset.

    Returns
    -------
    dict with:
        years
        features_df (DataFrame ready for discretization)
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Lynx-Hare dataset not found at: {csv_path}"
        )

    # === READ CSV ===
    df = pd.read_csv(csv_path, sep=';')

    # === STRONG COLUMN NORMALIZATION ===
    df.columns = [
        str(c).strip()
        .lower()
        .replace(' ', '')
        .replace('(monthly)', '')
        .replace('(', '')
        .replace(')', '')
        .replace('_', '')
        .replace(';', '')
        for c in df.columns
    ]

    print("Colunas detectadas:", list(df.columns))

    # === DETECT CORE COLUMNS ===
    hare_col = next((c for c in df.columns if 'hare' in c), None)
    lynx_col = next((c for c in df.columns if 'lynx' in c), None)
    year_col = next((c for c in df.columns if 'year' in c), None)

    # Optional external variable
    soi_col = next((c for c in df.columns if 'soi' in c), None)

    if hare_col is None or lynx_col is None or year_col is None:
        raise ValueError(
            f"Missing required columns. Found: {list(df.columns)}"
        )

    # === SAFE NUMERIC CONVERSION ===
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
    df[hare_col] = pd.to_numeric(df[hare_col], errors='coerce')
    df[lynx_col] = pd.to_numeric(df[lynx_col], errors='coerce')

    if soi_col is not None:
        df[soi_col] = pd.to_numeric(df[soi_col], errors='coerce')

    # === SORT BY YEAR ===
    df = df.sort_values(by=year_col).reset_index(drop=True)

    # === LOG TRANSFORM (avoid log(0)) ===
    eps = 1e-6
    df["hare_log"] = np.log(df[hare_col] + eps)
    df["lynx_log"] = np.log(df[lynx_col] + eps)

    # === LOG RETURNS (CORE FEATURE) ===
    df["hare_log_return"] = df["hare_log"].diff()
    df["lynx_log_return"] = df["lynx_log"].diff()

    # === NORMALIZED YEAR ===
    year_min = df[year_col].min()
    year_max = df[year_col].max()

    df["year_norm"] = (df[year_col] - year_min) / (year_max - year_min)

    # === BUILD FEATURE SET ===
    feature_cols = [
        "hare_log_return",
        "lynx_log_return",
        "year_norm",
    ]

    # Add SOI only where available
    if soi_col is not None:
        df["soi_clean"] = df[soi_col]
        feature_cols.append("soi_clean")

    # === CLEAN ONLY NECESSARY COLUMNS ===
    df_features = df[feature_cols].copy()

    # Remove rows where core dynamics are missing
    df_features = df_features.dropna(subset=[
        "hare_log_return",
        "lynx_log_return"
    ])

    # If SOI exists, drop rows only where SOI is missing (optional strict mode)
    if "soi_clean" in df_features.columns:
        df_features = df_features.dropna()

    print(f"Total linhas após processamento: {len(df_features)}")

    return {
        "years": df.loc[df_features.index, year_col].values,
        "features_df": df_features
    }


def build_predator_prey_matrix(data):
    """
    Convert processed features into matrix for discretization.
    """

    df = data["features_df"]

    # mantém ordem consistente
    cols = list(df.columns)

    X = df[cols].to_numpy(dtype=float)

    print(f"Matriz final construída com shape: {X.shape}")

    return X