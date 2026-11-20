#!/usr/bin/env python3
# =============================================================================
# plot_spaghetti_sparsity_se_cdfcurve_bootstrap_v7f.py
# =============================================================================
# Publication-style 4x2 figure for contrasting high- and low-savings models.
#
# Rows:
#   1. Full submitted trajectories (spaghetti)
#   2. Full-sample quantile sparsity vs horizon
#   3. Full-sample bootstrap SE vs horizon
#   4. Quantile-function / CDF-lens crosshair plot at one representative horizon
#
# The fourth row shows the empirical quantile function Q(q) at a representative
# horizon. For selected quantiles q in Q_LEVELS, the vertical half-length is the
# bootstrap SE on the outcome scale and the horizontal half-length is the bootstrap SD of F*(Qhat(q)),
# holding Qhat(q) fixed across bootstrap replicates.
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd()
STOPPING_TABLE = ROOT / "htc_iid_20k" / "analysis" / "replay_stop_results_joint_tau_quantiles.csv"
RAW_DIR = ROOT / "data/processed"
FULL_N_SE_TABLE = ROOT / "results_raw_quantile_heterogeneity" / "full_n_model_bootstrap_unified103" / "full_n_model_quantile_bootstrap_se_UNIFIED_103Q.csv"

TAU = 7500.0
STOPPING_SPEC = "through_0.95"
SCENARIO_SHORT = "A"
HIGH_MODEL: Optional[str] = None
LOW_MODEL: Optional[str] = None

Q_LEVELS = [0.50, 0.80, 0.90, 0.95]
Q_COLOR_MAP = {0.50: "tab:blue", 0.80: "tab:purple", 0.90: "tab:orange", 0.95: "tab:green"}
DELTA = 0.04
BOOTSTRAP_REPS = 20000
BOOTSTRAP_SEED = 20260917
CROSSHAIR_Q_GRID = np.array([0.01, 0.025] + list(np.arange(0.05, 0.951, 0.05)) + [0.975, 0.99])


def crosshair_colors(n: int):
    cmap = plt.get_cmap("cividis")
    vals = np.linspace(0.18, 0.78, n)
    return [cmap(v) for v in vals]

HORIZON_START = 1
DPI = 300
MAX_TRAJECTORIES_TO_DRAW: Optional[int] = None
TRAJECTORY_SUBSAMPLE_SEED = 20260917

OUTPUT_DIR = ROOT / "paper_spaghetti_sparsity_se"
OUTPUT_FIGURE = OUTPUT_DIR / "FIG_spaghetti_sparsity_se_cdfcurve_bootstrap_23q_v7f.png"
OUTPUT_SELECTION = OUTPUT_DIR / "TABLE_spaghetti_sparsity_se_cdfcurve_bootstrap_23q_v7f_selected_models.csv"
OUTPUT_CROSSHAIR = OUTPUT_DIR / "TABLE_spaghetti_sparsity_se_cdfcurve_bootstrap_23q_v7f.csv"


# =============================================================================
# Stop-table loading and model selection
# =============================================================================

def load_selected_operating_point() -> pd.DataFrame:
    if not STOPPING_TABLE.exists():
        raise FileNotFoundError(f"Missing stopping table: {STOPPING_TABLE}")
    d = pd.read_csv(STOPPING_TABLE)
    required = {
        "scenario", "scenario_short", "model", "replay", "stopping_spec",
        "tau", "stop_n", "n_available", "fraction_saved",
    }
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"Stopping table missing required columns: {sorted(missing)}")

    x = d[(d["stopping_spec"].astype(str) == STOPPING_SPEC) &
          np.isclose(pd.to_numeric(d["tau"], errors="coerce"), TAU)].copy()
    if x.empty:
        raise RuntimeError(f"No rows found for stopping_spec={STOPPING_SPEC}, tau={TAU:g}")

    dup = x.duplicated(["scenario", "model", "replay"]).sum()
    if dup:
        raise RuntimeError(f"Found {dup} duplicate scenario/model/replay rows")
    replay_counts = x.groupby(["scenario", "model"])["replay"].nunique()
    if not (replay_counts == 100).all():
        raise RuntimeError("Expected 100 replays per model-scenario pair")
    return x


def select_high_low_models(x: pd.DataFrame) -> pd.DataFrame:
    z = x[x["scenario_short"].astype(str) == str(SCENARIO_SHORT)].copy()
    if z.empty:
        raise RuntimeError(f"No stopping rows found for scenario_short={SCENARIO_SHORT}")

    summary = (
        z.groupby(["scenario", "scenario_short", "model"], as_index=False)
        .agg(
            n_replays=("replay", "nunique"),
            n_available=("n_available", "first"),
            median_stop_n=("stop_n", "median"),
            median_fraction_saved=("fraction_saved", "median"),
            min_fraction_saved=("fraction_saved", "min"),
            max_fraction_saved=("fraction_saved", "max"),
        )
        .sort_values(["median_fraction_saved", "model"])
        .reset_index(drop=True)
    )

    if HIGH_MODEL is None and LOW_MODEL is None:
        low = summary.iloc[0].copy()
        high = summary.iloc[-1].copy()
    elif HIGH_MODEL is not None and LOW_MODEL is not None:
        high_rows = summary[summary["model"].astype(str) == str(HIGH_MODEL)]
        low_rows = summary[summary["model"].astype(str) == str(LOW_MODEL)]
        if high_rows.empty:
            raise ValueError(f"HIGH_MODEL={HIGH_MODEL!r} not found")
        if low_rows.empty:
            raise ValueError(f"LOW_MODEL={LOW_MODEL!r} not found")
        high = high_rows.iloc[0].copy()
        low = low_rows.iloc[0].copy()
    else:
        raise ValueError("Set both HIGH_MODEL and LOW_MODEL, or leave both as None.")

    high["selection"] = "high savings"
    low["selection"] = "low savings"
    return pd.DataFrame([high, low]).reset_index(drop=True)


# =============================================================================
# Raw trajectory loading
# =============================================================================

_SCENARIO_CACHE: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}


def load_raw_scenario(scenario: str) -> tuple[np.ndarray, pd.DataFrame]:
    matrix_path = RAW_DIR / f"Global_{scenario}_raw_matrix.npy"
    metadata_path = RAW_DIR / f"Global_{scenario}_metadata.csv"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing raw matrix: {matrix_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {metadata_path}")

    X = np.load(matrix_path)
    meta = pd.read_csv(metadata_path)
    if X.ndim != 2:
        raise ValueError(f"Expected raw matrix to be 2D, got {X.shape}")
    if len(meta) != X.shape[1] and len(meta) == X.shape[0]:
        X = X.T
    if len(meta) != X.shape[1]:
        raise ValueError("Metadata rows do not match raw matrix columns")

    meta = meta.copy()
    meta["model"] = meta["model"].astype(str)
    return X, meta


def get_model_full_trajectories(scenario: str, model: str) -> np.ndarray:
    if scenario not in _SCENARIO_CACHE:
        _SCENARIO_CACHE[scenario] = load_raw_scenario(scenario)
    X_h_by_t, meta = _SCENARIO_CACHE[scenario]
    idx = np.where(meta["model"].to_numpy().astype(str) == str(model))[0]
    if idx.size == 0:
        raise ValueError(f"No raw trajectories found for model={model}, scenario={scenario}")
    return X_h_by_t[:, idx].T  # trajectories x horizons


# =============================================================================
# Derived quantities
# =============================================================================

def estimated_sparsity(X: np.ndarray, q: float, delta: float = DELTA) -> np.ndarray:
    q_lo = q - delta
    q_hi = q + delta
    if q_lo < 0 or q_hi > 1:
        raise ValueError(f"q={q:g}, delta={delta:g} requires quantiles outside [0,1].")
    Q_lo = np.quantile(X, q_lo, axis=0)
    Q_hi = np.quantile(X, q_hi, axis=0)
    return (Q_hi - Q_lo) / (2.0 * delta)


def bootstrap_quantile_store(X: np.ndarray, q_levels: list[float] | np.ndarray,
                             n_boot: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED) -> dict[float, np.ndarray]:
    n_traj, n_h = X.shape
    rng = np.random.default_rng(seed)
    q_levels = [float(q) for q in q_levels]
    store = {q: np.empty((n_boot, n_h), dtype=float) for q in q_levels}
    for b in range(n_boot):
        idx = rng.integers(0, n_traj, size=n_traj)
        xb = X[idx, :]
        for q in q_levels:
            store[q][b, :] = np.quantile(xb, q, axis=0)
    return store


def bootstrap_quantile_se_curves(X: np.ndarray, q_levels: list[float],
                                 n_boot: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED) -> dict[float, np.ndarray]:
    store = bootstrap_quantile_store(X, q_levels=q_levels, n_boot=n_boot, seed=seed)
    return {q: np.std(store[q], axis=0, ddof=1) for q in q_levels}



def load_full_sample_se_curves(scenario: str, model: str, n_horizons: int) -> dict[float, np.ndarray]:
    """Use precomputed full-sample 20,000-replicate SE; never recompute the main curves."""
    if not FULL_N_SE_TABLE.is_file():
        raise FileNotFoundError(f"Missing unified 20k model bootstrap file: {FULL_N_SE_TABLE}")
    df = pd.read_csv(FULL_N_SE_TABLE)
    required = {"scenario", "model", "horizon", "quantile", "bootstrap_se", "bootstrap_reps"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Unified bootstrap CSV missing columns: {sorted(missing)}")
    df = df[(df["scenario"].astype(str) == scenario) & (df["model"].astype(str) == model)].copy()
    if df.empty:
        raise ValueError(f"No 20k bootstrap results for {scenario} / {model}")
    curves = {}
    for q in Q_LEVELS:
        sub = df[np.isclose(pd.to_numeric(df["quantile"], errors="coerce"), q)].sort_values("horizon")
        if (len(sub) != n_horizons or
            not np.array_equal(sub["horizon"].to_numpy(dtype=int), np.arange(1, n_horizons+1)) or
            not (pd.to_numeric(sub["bootstrap_reps"]) == 20000).all()):
            raise ValueError(f"Incomplete 20k bootstrap grid for {scenario}/{model}, q={q}")
        curves[q] = sub["bootstrap_se"].to_numpy(dtype=float)
    return curves

def choose_crosshair_horizon(se_curves: dict[float, np.ndarray], target_q: float = 0.95) -> int:
    return int(np.argmax(se_curves[target_q]))


def probability_scale_halfwidth(se_outcome: float, sparsity: float) -> float:
    if not np.isfinite(sparsity) or sparsity <= 0:
        return np.nan
    return float(se_outcome / sparsity)


# =============================================================================
# Plotting helpers
# =============================================================================

def maybe_subsample_trajectories(X: np.ndarray) -> np.ndarray:
    if MAX_TRAJECTORIES_TO_DRAW is None or X.shape[0] <= MAX_TRAJECTORIES_TO_DRAW:
        return X
    rng = np.random.default_rng(TRAJECTORY_SUBSAMPLE_SEED)
    idx = np.sort(rng.choice(X.shape[0], size=MAX_TRAJECTORIES_TO_DRAW, replace=False))
    return X[idx]


def add_panel_label(ax, letter: str) -> None:
    ax.text(0.02, 0.97, letter, transform=ax.transAxes,
            ha="left", va="top", fontsize=11, fontweight="bold")


def draw_spaghetti(ax, X: np.ndarray, panel_letter: str) -> None:
    horizons = np.arange(HORIZON_START, HORIZON_START + X.shape[1])
    X_draw = maybe_subsample_trajectories(X)
    for tr in X_draw:
        ax.plot(horizons, tr, linewidth=0.5, alpha=0.10)
    for q in Q_LEVELS:
        ax.plot(horizons, np.quantile(X, q, axis=0), linewidth=1.6, color=Q_COLOR_MAP[q], label=f"q={q:g}")
    add_panel_label(ax, panel_letter)
    ax.set_ylabel("Incident hospitalizations")
    ax.grid(alpha=0.18, linewidth=0.5)


def draw_sparsity(ax, X: np.ndarray, panel_letter: str) -> None:
    horizons = np.arange(HORIZON_START, HORIZON_START + X.shape[1])
    for q in Q_LEVELS:
        ax.plot(horizons, estimated_sparsity(X, q), linewidth=1.7, color=Q_COLOR_MAP[q], label=f"q={q:g}")
    add_panel_label(ax, panel_letter)
    ax.set_ylabel(rf"Estimated sparsity $\widehat{{s}}_{{{DELTA:g}}}(q)$")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.grid(alpha=0.18, linewidth=0.5)


def draw_se(ax, se_curves: dict[float, np.ndarray], n_horizons: int, panel_letter: str) -> None:
    horizons = np.arange(HORIZON_START, HORIZON_START + n_horizons)
    for q in Q_LEVELS:
        ax.plot(horizons, se_curves[q], linewidth=1.7, color=Q_COLOR_MAP[q], label=f"q={q:g}")
    add_panel_label(ax, panel_letter)
    ax.set_ylabel("Bootstrap SE")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.grid(alpha=0.18, linewidth=0.5)


def bootstrap_dual_se_at_horizon(values: np.ndarray, q_grid: np.ndarray,
                                 n_boot: int, seed: int):
    """Bootstrap Q*(q) and F*(Qhat(q)) using paired trajectory resamples.

    The CDF thresholds Qhat(q) are fixed from the ORIGINAL sample, not
    re-estimated on each resample. Counting <= includes ties, matching the
    right-continuous empirical CDF convention.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    original_quantiles = np.quantile(values, q_grid)
    rng = np.random.default_rng(seed)
    boot_quantiles = np.empty((n_boot, len(q_grid)), dtype=float)
    boot_probabilities = np.empty_like(boot_quantiles)
    for b in range(n_boot):
        sample = values[rng.integers(0, n, n)]
        boot_quantiles[b] = np.quantile(sample, q_grid)
        boot_probabilities[b] = np.mean(sample[:, None] <= original_quantiles[None, :], axis=0)
    return (original_quantiles,
            boot_quantiles.std(axis=0, ddof=1),
            boot_probabilities.std(axis=0, ddof=1),
            np.mean(values[:, None] <= original_quantiles[None, :], axis=0))


def draw_crosshair(ax, X: np.ndarray, se_curves: dict[float, np.ndarray], panel_letter: str,
                   representative_horizon: int) -> dict[str, float]:
    q_grid = CROSSHAIR_Q_GRID
    horizon_label = representative_horizon + 1
    values = np.sort(X[:, representative_horizon])
    qfun, se_x, se_y, empirical_prob = bootstrap_dual_se_at_horizon(
        values, q_grid, n_boot=BOOTSTRAP_REPS,
        seed=BOOTSTRAP_SEED + representative_horizon
    )
    ecdf_y = np.arange(1, len(values) + 1) / len(values)
    colors = crosshair_colors(len(q_grid))
    ax.step(values, ecdf_y, where='post', linewidth=1.2, color='0.45', zorder=1)
    for j, q in enumerate(q_grid):
        ax.errorbar([qfun[j]], [empirical_prob[j]], xerr=[[se_x[j]], [se_x[j]]],
                    yerr=[[se_y[j]], [se_y[j]]], fmt='o',
                    color=colors[j], capsize=1.6,
                    markersize=3.0, elinewidth=0.75, alpha=0.95, zorder=3)
    add_panel_label(ax, panel_letter)
    ax.text(0.10, 0.97, f"h={horizon_label}", transform=ax.transAxes,
            ha="left", va="top", fontsize=9)
    ax.set_xlabel("Incident hospitalizations")
    ax.set_ylabel("Probability level q")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 11))
    ax.grid(alpha=0.18, linewidth=0.5)
    return {
        "representative_horizon": horizon_label,
        **{f"se_probability_q{str(float(q)).replace('.', '_')}": float(se_y[j]) for j, q in enumerate(q_grid)},
        **{f"se_outcome_q{str(float(q)).replace('.', '_')}": float(se_x[j]) for j, q in enumerate(q_grid)},
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stopping = load_selected_operating_point()
    selected = select_high_low_models(stopping)
    selected.to_csv(OUTPUT_SELECTION, index=False)

    print("\nSelected models from canonical joint bootstrap-SE stopping results:")
    print(selected[[
        "selection", "scenario", "model", "n_available", "median_stop_n",
        "median_fraction_saved", "min_fraction_saved", "max_fraction_saved",
    ]].to_string(index=False))

    fig, axes = plt.subplots(4, 2, figsize=(12.6, 14.6), sharex=False, squeeze=False)
    # Share y within each row, preserving independent y-scales between rows.
    # Do NOT share x with row 4: it is probability rather than forecast horizon.
    for row in range(4):
        axes[row, 1].sharey(axes[row, 0])
    for col in range(2):
        for row in (1, 2):
            axes[row, col].sharex(axes[0, col])
    panel_letters = list("ABCDEFGH")
    crosshair_rows = []

    column_headers = []

    for col_i, (_, row) in enumerate(selected.iterrows()):
        scenario = str(row["scenario"])
        model = str(row["model"])
        median_saved = float(row["median_fraction_saved"])
        X = get_model_full_trajectories(scenario=scenario, model=model)
        se_curves = load_full_sample_se_curves(scenario, model, X.shape[1])
        h_star = choose_crosshair_horizon(se_curves, target_q=0.95)

        draw_spaghetti(axes[0, col_i], X, panel_letters[col_i])
        draw_sparsity(axes[1, col_i], X, panel_letters[2 + col_i])
        draw_se(axes[2, col_i], se_curves, X.shape[1], panel_letters[4 + col_i])
        crosshair_info = draw_crosshair(axes[3, col_i], X, se_curves, panel_letters[6 + col_i], representative_horizon=h_star)

        crosshair_rows.append({
            "selection": row["selection"],
            "scenario": scenario,
            "model": model,
            "median_fraction_saved": median_saved,
            **crosshair_info,
        })
        column_headers.append((col_i, f"{row['selection'].title()} model: {model} (Scenario {row['scenario_short']}; median saved {100*median_saved:.1f}%)"))


    # The right column shares the y-axis scale with the left column.
    # Keep right-side y tick labels for readability, but remove redundant y-axis labels.
    for row in range(4):
        axes[row, 1].set_ylabel("")
        axes[row, 1].tick_params(axis="y", labelleft=True, pad=2)

    # The bottom-row x-axis is independent from the first three rows.
    for j in range(2):
        axes[0, j].set_xlabel("Forecast horizon")
        axes[1, j].set_xlabel("Forecast horizon")
        axes[2, j].set_xlabel("Forecast horizon")

    # Minimal column labels; no figure-level title.
    for col_i, text in column_headers:
        axes[0, col_i].text(0.5, 1.035, text, transform=axes[0, col_i].transAxes,
                            ha="center", va="bottom", fontsize=9)

    fig.subplots_adjust(left=0.08, right=0.985, top=0.95, bottom=0.06, hspace=0.28, wspace=0.12)
    fig.savefig(OUTPUT_FIGURE, dpi=DPI, bbox_inches='tight')
    fig.savefig(OUTPUT_FIGURE.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)

    pd.DataFrame(crosshair_rows).to_csv(OUTPUT_CROSSHAIR, index=False)
    print(f"\nWrote figure:\n  {OUTPUT_FIGURE}")
    print(f"Wrote selected-model table:\n  {OUTPUT_SELECTION}")
    print(f"Wrote crosshair-horizon table:\n  {OUTPUT_CROSSHAIR}")


if __name__ == "__main__":
    main()
