"""
ecology_loader.py
-----------------

Loader for ecological predator–prey datasets.

Currently supported dataset:
- Hudson Bay Company Lynx–Hare dataset

The loader returns raw time series ready for discretization
by the standard CDR pipeline.
"""

import os
import pandas as pd
import numpy as np


def load_lynx_hare_dataset(csv_path):
    """
    Load the lynx-hare predator–prey dataset.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the dataset.

    Returns
    -------
    data : dict
        Dictionary containing:
            years : np.ndarray
            hare : np.ndarray
            lynx : np.ndarray
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Lynx-Hare dataset not found at: {csv_path}"
        )

    # === LEITURA CORRETA DO CSV (sep=';' é obrigatório) ===
    df = pd.read_csv(csv_path, sep=';')

    # === NORMALIZAÇÃO FORTE DE COLUNAS ===
    df.columns = [
        str(c).strip()
         .lower()
         .replace(' ', '')
         .replace('(monthly)', '')
         .replace('(', '')
         .replace(')', '')
         .replace('_', '')
         .replace(';', '')          # remove qualquer ; que sobrou
        for c in df.columns
    ]

    print("Colunas após normalização:", list(df.columns))  # DEBUG

    # Detecção robusta
    hare_col = next((col for col in df.columns if 'hare' in col), None)
    lynx_col = next((col for col in df.columns if 'lynx' in col), None)
    year_col = next((col for col in df.columns if 'year' in col), None)

    if hare_col is None or lynx_col is None:
        raise ValueError(
            f"Não encontrou 'hare' e 'lynx'. Colunas disponíveis: {list(df.columns)}"
        )

    # Conversão segura para float (ignora linhas ruins)
    years = pd.to_numeric(df[year_col], errors='coerce').values
    hare  = pd.to_numeric(df[hare_col],  errors='coerce').values
    lynx  = pd.to_numeric(df[lynx_col],  errors='coerce').values

    # Remove linhas com NaN (caso existam)
    mask = ~np.isnan(hare) & ~np.isnan(lynx)
    years = years[mask]
    hare  = hare[mask]
    lynx  = lynx[mask]

    print(f"Dataset carregado com sucesso: {len(years)} anos")

    return {
        "years": years,
        "hare": hare,
        "lynx": lynx,
    }


def build_predator_prey_matrix(data):
    """
    Convert predator–prey time series into a matrix
    suitable for discretization.
    """
    hare = data["hare"]
    lynx = data["lynx"]

    X = np.column_stack([hare, lynx])

    return X