# src/phase2_runner_ecology.py

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import config.phase2_config_ecology as cfg

from src.ecology_loader import load_lynx_hare_dataset, build_predator_prey_matrix
from src.discretize import fit_and_discretize
from src.build_states import make_encoding, encode_states, build_transitions
from src.kernels.empirical_kernel import EmpiricalKernel
from src.validators_phase2 import (
    gate_F1_injection_recovery,
    gate_F2_controls_collapse,
    gate_F3_holdout_generalization,
    gate_F5_sensitivity,
    summarize,
)
from src.controls_phase2_ecology import CONTROL_REGISTRY
from src.phase2_runner import _estimate_epsilon_grid, _simulate_trajectory


def run_phase2_ecology():
    print("\n==============================")
    print("CDR Phase II.3 — Ecology Domain")
    print("==============================\n")

    rng = np.random.default_rng(cfg.RANDOM_SEED)

    # -------------------------------------------------
    # Config summary
    # -------------------------------------------------

    print("[Phase2-Ecology] Configuration loaded")
    print("Dataset path:", cfg.DATA_PATH)
    print("State variables:", tuple(cfg.STATE_VARIABLES))
    print("Bins per variable:", cfg.BINS_PER_VARIABLE)
    print("Alt bins:", cfg.ALT_BINS)

    # -------------------------------------------------
    # Load dataset
    # -------------------------------------------------

    print("\n[Phase2-Ecology] Loading lynx-hare dataset...")

    data = load_lynx_hare_dataset(cfg.DATA_PATH)
    X = build_predator_prey_matrix(data)

    df = pd.DataFrame(X, columns=list(cfg.STATE_VARIABLES))

    # Keep only the ecological variables and drop missing rows
    df = df[list(cfg.STATE_VARIABLES)].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    if len(df) < 12:
        raise ValueError(
            f"Dataset too small after cleaning: {len(df)} rows. Need at least 12 rows."
        )

    print("[Phase2-Ecology] Rows after cleaning:", len(df))
    print("[Phase2-Ecology] Columns:", list(df.columns))

    # -------------------------------------------------
    # Train/Test split
    # -------------------------------------------------

    n = len(df)
    split = int(cfg.TRAIN_FRACTION * n)

    if split < 6 or (n - split) < 4:
        raise ValueError(
            f"Invalid split for ecology dataset: n={n}, split={split}. "
            "Need enough rows for both train and test."
        )

    idx_train = np.arange(0, split, dtype=int)
    idx_test = np.arange(split, n, dtype=int)

    print("\n[Phase2-Ecology] Train/Test split")
    print("Train rows:", len(idx_train))
    print("Test rows:", len(idx_test))

    # -------------------------------------------------
    # Discretization
    # -------------------------------------------------

    print("\n[Phase2-Ecology] Discretizing ecological variables...")

    if cfg.BINS_PER_VARIABLE == 2:
        quantiles_main = (0.5,)
    elif cfg.BINS_PER_VARIABLE == 3:
        quantiles_main = (0.33, 0.66)
    else:
        # for >=4, your discretize.py ignores provided quantiles and computes evenly spaced cuts
        quantiles_main = (0.25, 0.5, 0.75)

    df_disc, specs = fit_and_discretize(
        df,
        n_bins=cfg.BINS_PER_VARIABLE,
        quantiles=quantiles_main,
        fit_on_index=idx_train,
    )

    print("[Phase2-Ecology] Discretization completed")

    # -------------------------------------------------
    # Build states and transitions
    # -------------------------------------------------

    n_components = df_disc.shape[1]
    enc = make_encoding(
        n_components=n_components,
        n_bins=cfg.BINS_PER_VARIABLE,
    )

    n_states = cfg.BINS_PER_VARIABLE ** n_components

    comps = df_disc.to_numpy(dtype=int)
    state_ids = encode_states(comps, enc)

    curr_all, nxt_all = build_transitions(state_ids)

    curr_train = curr_all[: split - 1]
    nxt_train = nxt_all[: split - 1]

    curr_test = curr_all[split - 1 :]
    nxt_test = nxt_all[split - 1 :]

    print(f"[Phase2-Ecology] State space: {n_components} variables x {cfg.BINS_PER_VARIABLE} bins = {n_states} states")
    print(f"[Phase2-Ecology] Transitions: train={len(curr_train)} | test={len(curr_test)}")

    # -------------------------------------------------
    # Build empirical kernel P0
    # -------------------------------------------------

    print("\n[Phase2-Ecology] Building empirical baseline kernel P0...")

    # Domain default consistent with previous Phase II runs
    dirichlet_alpha = 0.1
    min_prob = 1e-12

    P0 = EmpiricalKernel.from_transitions(
        curr_train,
        nxt_train,
        n_states=n_states,
        enc=enc,
        alpha=dirichlet_alpha,
    )

    print("[Phase2-Ecology] P0 built.")

    eps_grid = np.linspace(cfg.EPS_MIN, cfg.EPS_MAX, cfg.EPS_GRID_SIZE, dtype=float)

    # -------------------------------------------------
    # Estimate epsilon (train)
    # -------------------------------------------------

    print("\n[Phase2-Ecology] Estimating epsilon (train)...")

    eps_train, ll_train = _estimate_epsilon_grid(
        curr_train,
        nxt_train,
        P0,
        eps_grid,
        min_prob,
        label="ecology_train",
        progress_every=5,
    )

    print(f"[Phase2-Ecology] eps_hat_train = {eps_train:.4f}")

    # -------------------------------------------------
    # Estimate epsilon (test)
    # -------------------------------------------------

    print("\n[Phase2-Ecology] Estimating epsilon (test)...")

    eps_test, ll_test = _estimate_epsilon_grid(
        curr_test,
        nxt_test,
        P0,
        eps_grid,
        min_prob,
        label="ecology_test",
        progress_every=5,
    )

    print(f"[Phase2-Ecology] eps_hat_test = {eps_test:.4f}")

    # -------------------------------------------------
    # Gate F1 — Injection recovery
    # -------------------------------------------------

    print("\n[Phase2-Ecology] Running Gate F1 (injection recovery)...")

    eps_true = float(cfg.INJECTION_EPS)

    sim_traj = _simulate_trajectory(
        P0,
        eps=eps_true,
        n_steps=len(curr_train),
        seed=cfg.RANDOM_SEED,
    )

    sim_curr, sim_nxt = build_transitions(sim_traj)

    eps_injected, ll_inj = _estimate_epsilon_grid(
        sim_curr,
        sim_nxt,
        P0,
        eps_grid,
        min_prob,
        label="injection",
        progress_every=5,
    )

    gate1 = gate_F1_injection_recovery(
        eps_hat=eps_injected,
        eps_true=eps_true,
        tol_abs=float(cfg.INJECTION_TOL),
    )

    print(f"[Phase2-Ecology] F1: eps_injected={eps_injected:.4f} vs eps_true={eps_true:.4f}")

    # -------------------------------------------------
    # Gate F2 — Controls collapse
    # -------------------------------------------------

    print("\n[Phase2-Ecology] Running Gate F2 (controls collapse)...")

    eps_controls = []

    for i, control_name in enumerate(cfg.CONTROL_TYPES, start=1):
        if control_name not in CONTROL_REGISTRY:
            raise KeyError(f"Unknown ecology control: {control_name}")

        print(f"[Phase2-Ecology] Control {i}/{len(cfg.CONTROL_TYPES)}: {control_name}")

        control_fn = CONTROL_REGISTRY[control_name]
        X_ctrl = control_fn(df.to_numpy(dtype=float).copy(), rng)

        df_ctrl = pd.DataFrame(X_ctrl, columns=list(cfg.STATE_VARIABLES))
        df_ctrl = df_ctrl.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

        # Keep same train window size if possible
        if len(df_ctrl) < len(df):
            m = len(df_ctrl)
            split_ctrl = int(cfg.TRAIN_FRACTION * m)
            idx_train_ctrl = np.arange(0, split_ctrl, dtype=int)
        else:
            split_ctrl = split
            idx_train_ctrl = idx_train
            df_ctrl = df_ctrl.iloc[: len(df)].reset_index(drop=True)

        if split_ctrl < 6:
            raise ValueError(f"Control '{control_name}' became too short after cleaning.")

        df_ctrl_disc, _ = fit_and_discretize(
            df_ctrl,
            n_bins=cfg.BINS_PER_VARIABLE,
            quantiles=quantiles_main,
            fit_on_index=idx_train_ctrl,
        )

        comps_ctrl = df_ctrl_disc.to_numpy(dtype=int)
        ids_ctrl = encode_states(comps_ctrl, enc)
        c_ctrl, n_ctrl = build_transitions(ids_ctrl)

        c_ctrl_train = c_ctrl[: split_ctrl - 1]
        n_ctrl_train = n_ctrl[: split_ctrl - 1]

        eps_c, _ = _estimate_epsilon_grid(
            c_ctrl_train,
            n_ctrl_train,
            P0,
            eps_grid,
            min_prob,
            label=f"control_{i}_{control_name}",
            progress_every=5,
        )

        eps_controls.append(eps_c)
        print(f"  -> eps={eps_c:.4f}")

    gate2 = gate_F2_controls_collapse(
        eps_controls=eps_controls,
        tol=0.05,
        required_fraction=2 / 3,
    )

    # -------------------------------------------------
    # Gate F3 — Holdout generalization
    # -------------------------------------------------

    print("\n[Phase2-Ecology] Running Gate F3 (holdout generalization)...")

    gate3 = gate_F3_holdout_generalization(
        eps_train=eps_train,
        eps_test=eps_test,
        max_delta=float(cfg.MAX_GENERALIZATION_DELTA),
    )

    # -------------------------------------------------
    # Gate F5 — Sensitivity
    # -------------------------------------------------

    print("\n[Phase2-Ecology] Running Gate F5 (sensitivity)...")

    # Sensitivity test keeps same variables but changes discretization resolution
    if cfg.ALT_BINS == 2:
        quantiles_alt = (0.5,)
    elif cfg.ALT_BINS == 3:
        quantiles_alt = (0.33, 0.66)
    else:
        quantiles_alt = (0.25, 0.5, 0.75)

    df_disc_alt, _ = fit_and_discretize(
        df,
        n_bins=cfg.ALT_BINS,
        quantiles=quantiles_alt,
        fit_on_index=idx_train,
    )

    enc_alt = make_encoding(
        n_components=n_components,
        n_bins=cfg.ALT_BINS,
    )

    n_states_alt = cfg.ALT_BINS ** n_components

    comps_alt = df_disc_alt.to_numpy(dtype=int)
    ids_alt = encode_states(comps_alt, enc_alt)
    c_alt, n_alt = build_transitions(ids_alt)

    c_alt_train = c_alt[: split - 1]
    n_alt_train = n_alt[: split - 1]

    P0_alt = EmpiricalKernel.from_transitions(
        c_alt_train,
        n_alt_train,
        n_states=n_states_alt,
        enc=enc_alt,
        alpha=dirichlet_alpha,
    )

    eps_alt, ll_alt = _estimate_epsilon_grid(
        c_alt_train,
        n_alt_train,
        P0_alt,
        eps_grid,
        min_prob,
        label=f"bins{cfg.ALT_BINS}",
        progress_every=5,
    )

    gate5 = gate_F5_sensitivity(
        eps_binsA=eps_train,
        eps_binsB=eps_alt,
        max_delta=float(cfg.MAX_BIN_DELTA),
    )

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    gates = [gate1, gate2, gate3, gate5]
    summary = summarize(gates)

    print("\n==============================")
    print("CDR Phase II.3 (Ecology) — Gates")
    print("==============================")

    for g in gates:
        status = "PASS" if g.passed else "FAIL"
        print(f"{g.name}: {status} | metrics={g.metrics}")

    print("==============================")
    final_status = "PASS ✅" if summary["passed_all"] else "FAIL ❌"
    print(f"FINAL: {final_status}")

    # -------------------------------------------------
    # Save results
    # -------------------------------------------------

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    results = {
        "dataset_path": cfg.DATA_PATH,
        "state_variables": list(cfg.STATE_VARIABLES),
        "n_rows_clean": int(len(df)),
        "n_states_main": int(n_states),
        "n_states_alt": int(n_states_alt),
        "eps_hat_train": float(eps_train),
        "eps_hat_test": float(eps_test),
        "eps_hat_injection": float(eps_injected),
        "eps_controls": [float(x) for x in eps_controls],
        "eps_hat_bins_alt": float(eps_alt),
        "gates": summary,
    }

    out_file = os.path.join(cfg.RESULTS_DIR, "phase2_ecology_results.json")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    run_phase2_ecology()