#!/usr/bin/env python3
# =============================================================================
# 14_unified_quantile_se_explanation_ALL23Q.py
# =============================================================================
# Final unified explanatory analysis using ALL 23 monitored quantiles.
#
# Scientific organizing principle:
#
#   SE(Qhat_q) ~ s(q) * sqrt(q(1-q) / n)
#
# This version does NOT rely on the older 4-quantile
# 00_quantile_se_explanatory_dataset.csv.
#
# Instead it rebuilds the explanatory dataset directly from:
#
#   results_raw_quantile_heterogeneity/full_n_model_ensemble_bootstrap/
#       full_n_model_quantile_bootstrap_se_FULL_GRID_long.csv
#
# and the raw trajectory matrices in:
#
#   data/processed/
#
# No bootstrap is rerun.
# =============================================================================

from pathlib import Path
from typing import Dict, Sequence
import glob
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

try:
    import statsmodels.formula.api as smf
except ImportError as e:
    raise ImportError(
        "This script requires statsmodels. Install with: pip install statsmodels"
    ) from e


# =============================================================================
# SETTINGS
# =============================================================================

repo_root = Path.cwd()

raw_input_dir = repo_root / "data/processed"

full_n_se_path = (
    repo_root
    / "results_raw_quantile_heterogeneity"
    / "full_n_model_bootstrap_unified103"
    / "full_n_model_quantile_bootstrap_se_UNIFIED_103Q.csv"
)

output_dir = (
    repo_root
    / "results_raw_quantile_heterogeneity"
    / "quantile_se_explanation"
)

QUANTILES = [
    0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
    0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
    0.975, 0.99,
]

PRIMARY_SPARSITY_DELTA = 0.04
SPARSITY_DELTAS = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
BOUNDARY_MODE = "adaptive"

UPPER_TAIL_SENSITIVITY_QUANTILE = 0.99

DPI = 300

# Publication-facing quantile grouping for all 23 quantiles.
QUANTILE_GROUP_COLORS = {
    "Extreme lower": "#4C78A8",
    "Lower/interior": "#72B7B2",
    "Median": "#54A24B",
    "Upper/interior": "#ECA82C",
    "Upper tail": "#E45756",
}

QUANTILE_GROUP_ORDER = [
    "Extreme lower",
    "Lower/interior",
    "Median",
    "Upper/interior",
    "Upper tail",
]

GROUP_ALPHA = {
    "Extreme lower": 0.48,
    "Lower/interior": 0.42,
    "Median": 0.70,
    "Upper/interior": 0.42,
    "Upper tail": 0.52,
}

GROUP_SIZE = {
    "Extreme lower": 12,
    "Lower/interior": 11,
    "Median": 20,
    "Upper/interior": 11,
    "Upper tail": 13,
}


# =============================================================================
# RAW DATA HELPERS
# =============================================================================

def discover_scenarios(input_dir: Path):
    files = sorted(glob.glob(str(input_dir / "Global_*_raw_matrix.npy")))
    return [
        Path(f).name.replace("Global_", "").replace("_raw_matrix.npy", "")
        for f in files
    ]


def ordered_scenarios(found: Sequence[str]):
    preferred = ["A-2025-07-29", "B-2025-07-29", "C-2025-07-29"]
    found = [str(x) for x in found]
    out = [x for x in preferred if x in found]
    out += sorted([x for x in found if x not in out])
    return out


def short_scenario(x):
    return str(x).split("-")[0]


def load_raw_scenario(input_dir: Path, scenario: str):
    matrix_path = input_dir / f"Global_{scenario}_raw_matrix.npy"
    meta_path = input_dir / f"Global_{scenario}_metadata.csv"

    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing raw matrix: {matrix_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")

    X = np.load(matrix_path)
    meta = pd.read_csv(meta_path)

    if X.ndim != 2:
        raise ValueError(
            f"Expected 2D raw matrix for {scenario}; got shape {X.shape}"
        )

    if "model" not in meta.columns:
        raise ValueError(
            f"Metadata file missing required column 'model': {meta_path}"
        )

    # Normal orientation is horizons x trajectories.
    if X.shape[1] != len(meta) and X.shape[0] == len(meta):
        X = X.T

    if X.shape[1] != len(meta):
        raise ValueError(
            f"Metadata rows ({len(meta)}) do not match trajectory columns "
            f"({X.shape[1]}) for {scenario}; matrix shape={X.shape}"
        )

    meta = meta.copy()
    meta["model"] = meta["model"].astype(str)
    return X, meta


def model_groups(meta: pd.DataFrame) -> Dict[str, np.ndarray]:
    arr = meta["model"].to_numpy()
    return {
        model: np.where(arr == model)[0]
        for model in sorted(meta["model"].unique())
    }


def effective_delta(q: float, requested_delta: float):
    max_sym = min(q, 1.0 - q)

    if requested_delta <= max_sym:
        return requested_delta

    if BOUNDARY_MODE == "drop":
        return None

    if BOUNDARY_MODE == "adaptive":
        return max_sym if max_sym > 0 else None

    raise ValueError("BOUNDARY_MODE must be 'adaptive' or 'drop'")


def finite_quantile(x, p):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.quantile(x, p))


def local_sparsity_metrics(x, q, requested_delta):
    """
    Estimate quantile sparsity

        s(q) ~= [Q(q+d) - Q(q-d)] / (2d)

    directly from the completed empirical predictive distribution.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    d = effective_delta(float(q), requested_delta)

    if x.size == 0 or d is None or d <= 0:
        return {
            "q_lo": np.nan,
            "q_hi": np.nan,
            "effective_delta": np.nan,
            "quantile_value": np.nan,
            "local_spacing": np.nan,
            "local_sparsity": np.nan,
            "local_density": np.nan,
        }

    p_lo = float(q) - d
    p_hi = float(q) + d

    Q_lo = finite_quantile(x, p_lo)
    Q_mid = finite_quantile(x, float(q))
    Q_hi = finite_quantile(x, p_hi)

    spacing = Q_hi - Q_lo
    denom = p_hi - p_lo

    sparsity = (
        spacing / denom
        if np.isfinite(spacing) and denom > 0
        else np.nan
    )

    if np.isfinite(sparsity) and sparsity > 0:
        density = 1.0 / sparsity
    elif np.isfinite(sparsity) and sparsity == 0:
        density = np.inf
    else:
        density = np.nan

    return {
        "q_lo": p_lo,
        "q_hi": p_hi,
        "effective_delta": d,
        "quantile_value": Q_mid,
        "local_spacing": spacing,
        "local_sparsity": sparsity,
        "local_density": density,
    }


# =============================================================================
# BUILD ALL-23Q EXPLANATORY DATASET
# =============================================================================

def build_explanatory_dataset(sparsity_delta=PRIMARY_SPARSITY_DELTA):
    if not full_n_se_path.exists():
        raise FileNotFoundError(
            "Missing final full-N 23Q model SE file:\n"
            f"  {full_n_se_path}\n\n"
            "Run scripts/bootstrap_model_quantiles.py first."
        )

    if not raw_input_dir.exists():
        raise FileNotFoundError(
            f"Missing raw trajectory directory: {raw_input_dir}"
        )

    se = pd.read_csv(full_n_se_path)

    required = {
        "scenario",
        "model",
        "n_available",
        "horizon",
        "quantile",
        "bootstrap_se",
    }
    missing = required - set(se.columns)
    if missing:
        raise ValueError(
            f"Full-N SE CSV is missing required columns: {sorted(missing)}"
        )

    for c in ["n_available", "horizon", "quantile", "bootstrap_se"]:
        se[c] = pd.to_numeric(se[c], errors="coerce")

    # Require the full 23Q result rather than silently running a four-Q analysis.
    present_q = sorted(se["quantile"].dropna().unique())
    missing_q = [
        q for q in QUANTILES
        if not np.any(np.isclose(np.asarray(present_q, dtype=float), q))
    ]

    if missing_q:
        raise ValueError(
            "The supplied full-N SE file does not contain all 23 monitored "
            f"quantiles. Missing: {missing_q}\n"
            f"Observed quantiles: {present_q}\n\n"
            "Make sure you are reading the output from "
            "scripts/bootstrap_model_quantiles.py."
        )

    # Restrict exactly to the frozen 23Q analysis grid.
    q_arr = se["quantile"].to_numpy(dtype=float)
    keep = np.zeros(len(se), dtype=bool)
    for q in QUANTILES:
        keep |= np.isclose(q_arr, q)
    se = se.loc[keep].copy()

    records = []
    scenarios = ordered_scenarios(discover_scenarios(raw_input_dir))

    for scenario in scenarios:
        print(f"Building explanatory cells: {scenario}")

        X, meta = load_raw_scenario(raw_input_dir, scenario)
        groups = model_groups(meta)

        for model, cols in groups.items():
            X_m = X[:, cols]
            n_total = X_m.shape[1]

            se_sub = se[
                (se["scenario"].astype(str) == str(scenario))
                & (se["model"].astype(str) == str(model))
            ].copy()

            if se_sub.empty:
                warnings.warn(
                    f"No full-N SE rows found for {scenario} | {model}; skipping"
                )
                continue

            for h_idx in range(X_m.shape[0]):
                x_h = X_m[h_idx, :]

                for q in QUANTILES:
                    se_match = se_sub[
                        (se_sub["horizon"].astype(int) == h_idx + 1)
                        & np.isclose(se_sub["quantile"].astype(float), q)
                    ]

                    if se_match.empty:
                        continue

                    # Full-N file should contain one row/cell.
                    row = se_match.iloc[-1]

                    metrics = local_sparsity_metrics(
                        x_h,
                        q=q,
                        requested_delta=sparsity_delta,
                    )

                    q_value = metrics["quantile_value"]
                    sparsity = metrics["local_sparsity"]

                    theory_proxy = (
                        math.sqrt(q * (1.0 - q) / n_total) * sparsity
                        if np.isfinite(sparsity)
                        else np.nan
                    )

                    records.append({
                        "scenario": scenario,
                        "scenario_short": short_scenario(scenario),
                        "model": model,
                        "n_available": int(n_total),
                        "horizon": h_idx + 1,
                        "quantile": float(q),
                        "bootstrap_se": float(row["bootstrap_se"]),
                        "full_quantile_value": q_value,
                        "abs_full_quantile_value":
                            abs(q_value) if np.isfinite(q_value) else np.nan,
                        "q_lo": metrics["q_lo"],
                        "q_hi": metrics["q_hi"],
                        "requested_delta": sparsity_delta,
                        "effective_delta": metrics["effective_delta"],
                        "local_spacing": metrics["local_spacing"],
                        "local_sparsity": sparsity,
                        "local_density": metrics["local_density"],
                        "theory_se_proxy": theory_proxy,
                    })

    out = pd.DataFrame.from_records(records)

    if out.empty:
        raise RuntimeError("No explanatory rows were produced.")

    return out



# =============================================================================
# DELTA SENSITIVITY
# =============================================================================

def run_delta_sensitivity():
    """
    Rebuild sparsity/theory-proxy values from the raw trajectories for each
    candidate delta, without rerunning any bootstrap, and summarize how the
    theory-proxy calibration changes.

    Outputs:
      - 14_theory_proxy_delta_sensitivity_ALL23Q.csv
      - 14_theory_proxy_delta_sensitivity_R2.png
      - 14_theory_proxy_delta_sensitivity_slope.png
    """
    rows = []

    for delta in SPARSITY_DELTAS:
        print(f"\nDelta sensitivity: delta={delta:g}")

        ddf = build_explanatory_dataset(sparsity_delta=delta)

        needed = [
            "bootstrap_se",
            "abs_full_quantile_value",
            "local_sparsity",
            "theory_se_proxy",
            "quantile",
            "scenario",
            "model",
            "n_available",
        ]

        reg_d = (
            ddf[needed]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .copy()
        )

        fit = smf.ols(
            "bootstrap_se ~ theory_se_proxy",
            data=reg_d,
        ).fit()

        fit_plus_mag = smf.ols(
            "bootstrap_se ~ theory_se_proxy + abs_full_quantile_value",
            data=reg_d,
        ).fit()

        rows.append({
            "delta": float(delta),
            "n_obs": int(fit.nobs),
            "r_squared_theory_proxy_only": float(fit.rsquared),
            "adj_r_squared_theory_proxy_only": float(fit.rsquared_adj),
            "intercept": float(fit.params["Intercept"]),
            "slope_theory_proxy": float(fit.params["theory_se_proxy"]),
            "r_squared_theory_plus_magnitude": float(fit_plus_mag.rsquared),
            "delta_r2_magnitude_beyond_theory":
                float(fit_plus_mag.rsquared - fit.rsquared),
            "zero_sparsity_rows":
                int((reg_d["local_sparsity"] == 0).sum()),
            "zero_theory_proxy_rows":
                int((reg_d["theory_se_proxy"] == 0).sum()),
        })

    sens = pd.DataFrame(rows).sort_values("delta").reset_index(drop=True)

    out_csv = (
        output_dir / "14_theory_proxy_delta_sensitivity_ALL23Q.csv"
    )
    sens.to_csv(out_csv, index=False)

    # R^2 vs delta
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(
        sens["delta"],
        sens["r_squared_theory_proxy_only"],
        marker="o",
        linewidth=1.8,
    )
    ax.axvline(
        PRIMARY_SPARSITY_DELTA,
        linestyle="--",
        linewidth=1.0,
        color="black",
        alpha=0.65,
    )
    ax.set_xlabel(r"Sparsity finite-difference half-width, $\delta$")
    ax.set_ylabel(r"$R^2$: observed bootstrap SE vs theory predictor")
    ax.set_title("Sensitivity of theory-proxy fit to sparsity window")
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(
        output_dir / "14_theory_proxy_delta_sensitivity_R2.png",
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Calibration slope vs delta
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(
        sens["delta"],
        sens["slope_theory_proxy"],
        marker="o",
        linewidth=1.8,
    )
    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
        color="black",
        alpha=0.65,
    )
    ax.axvline(
        PRIMARY_SPARSITY_DELTA,
        linestyle="--",
        linewidth=1.0,
        color="black",
        alpha=0.65,
    )
    ax.set_xlabel(r"Sparsity finite-difference half-width, $\delta$")
    ax.set_ylabel("Fitted slope on theory predictor")
    ax.set_title("Sensitivity of theory-proxy calibration slope")
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(
        output_dir / "14_theory_proxy_delta_sensitivity_slope.png",
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\nDelta sensitivity summary:")
    print(sens.to_string(index=False))

    return sens


# =============================================================================
# MODEL HELPERS
# =============================================================================

def fit_model(formula, data):
    return smf.ols(formula, data=data).fit()


def model_row(name, formula, fit, analysis):
    return {
        "analysis": analysis,
        "model_name": name,
        "formula": formula,
        "n_obs": int(fit.nobs),
        "r_squared": float(fit.rsquared),
        "adj_r_squared": float(fit.rsquared_adj),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "intercept": float(fit.params.get("Intercept", np.nan)),
        "coef_magnitude":
            float(fit.params.get("abs_full_quantile_value", np.nan)),
        "coef_sparsity":
            float(fit.params.get("local_sparsity", np.nan)),
        "coef_theory_proxy":
            float(fit.params.get("theory_se_proxy", np.nan)),
    }


def run_models(df, analysis_label):
    formulas = {
        "U0_raw_magnitude_only":
            "bootstrap_se ~ abs_full_quantile_value",

        "U1_raw_sparsity_only":
            "bootstrap_se ~ local_sparsity",

        "U2_theory_proxy_only":
            "bootstrap_se ~ theory_se_proxy",

        "U3_theory_proxy_plus_magnitude":
            "bootstrap_se ~ theory_se_proxy + abs_full_quantile_value",

        "U4_theory_proxy_plus_scenario_model":
            "bootstrap_se ~ theory_se_proxy + C(scenario) + C(model)",

        "U5_theory_proxy_magnitude_plus_scenario_model":
            "bootstrap_se ~ theory_se_proxy + abs_full_quantile_value + "
            "C(scenario) + C(model)",

        # Sensitivity: residual quantile offsets after q has already entered
        # mechanistically through sqrt(q(1-q)).
        "U6_theory_proxy_plus_quantile_scenario_model":
            "bootstrap_se ~ theory_se_proxy + C(quantile) + "
            "C(scenario) + C(model)",

        "U7_theory_proxy_magnitude_plus_quantile_scenario_model":
            "bootstrap_se ~ theory_se_proxy + abs_full_quantile_value + "
            "C(quantile) + C(scenario) + C(model)",
    }

    rows = []
    summaries = []
    fits = {}

    for name, formula in formulas.items():
        fit = fit_model(formula, df)
        fits[name] = fit
        rows.append(model_row(name, formula, fit, analysis_label))

        summaries.extend([
            "=" * 96,
            f"{analysis_label} | {name}",
            formula,
            "=" * 96,
            fit.summary().as_text(),
            "",
        ])

    return pd.DataFrame(rows), fits, summaries


# =============================================================================
# PLOTS
# =============================================================================

def add_fit(ax, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)

    if ok.sum() < 3 or np.unique(x[ok]).size < 2:
        return

    slope, intercept = np.polyfit(x[ok], y[ok], 1)
    grid = np.linspace(x[ok].min(), x[ok].max(), 200)
    ax.plot(grid, intercept + slope * grid, linewidth=1.5)


def quantile_group(q):
    q = float(q)

    if (
        np.isclose(q, 0.01)
        or np.isclose(q, 0.025)
        or np.isclose(q, 0.05)
        or np.isclose(q, 0.10)
    ):
        return "Extreme lower"

    if 0.15 <= q <= 0.45:
        return "Lower/interior"

    if np.isclose(q, 0.50):
        return "Median"

    if 0.55 <= q <= 0.85:
        return "Upper/interior"

    if (
        np.isclose(q, 0.90)
        or np.isclose(q, 0.95)
        or np.isclose(q, 0.975)
        or np.isclose(q, 0.99)
    ):
        return "Upper tail"

    raise ValueError(f"Quantile {q} is not assigned to a plotting group.")


def scatter_all_quantiles(ax, sub, xcol, ycol, show_legend=False):
    """
    Plot all 23 quantiles using five discrete, interpretable quantile groups.
    """
    tmp = sub.copy()
    tmp["quantile_group"] = tmp["quantile"].apply(quantile_group)

    for group in QUANTILE_GROUP_ORDER:
        g = tmp[tmp["quantile_group"] == group]

        if g.empty:
            continue

        ax.scatter(
            g[xcol].to_numpy(dtype=float),
            g[ycol].to_numpy(dtype=float),
            s=GROUP_SIZE[group],
            alpha=GROUP_ALPHA[group],
            color=QUANTILE_GROUP_COLORS[group],
            edgecolors="none",
            zorder=2,
        )

    if show_legend:
        handles = []
        for group in QUANTILE_GROUP_ORDER:
            handles.append(
                plt.Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="None",
                    markersize=7,
                    markerfacecolor=QUANTILE_GROUP_COLORS[group],
                    markeredgecolor="none",
                    alpha=GROUP_ALPHA[group],
                    label=group,
                )
            )

        ax.legend(
            handles=handles,
            title="Quantile group",
            frameon=False,
            loc="upper left",
        )


def plot_main_two_panel(df, output_path):
    scenarios = ordered_scenarios(df["scenario"].unique())

    fig, axes = plt.subplots(
        len(scenarios),
        2,
        figsize=(12.5, 4.2 * len(scenarios)),
        squeeze=False,
    )

    for r, scenario in enumerate(scenarios):
        sub = df[df["scenario"] == scenario].copy()

        # ---------------------------------------------------------------------
        # Column 1: raw SE vs local sparsity
        # ---------------------------------------------------------------------
        ax = axes[r, 0]
        scatter_all_quantiles(
            ax,
            sub,
            "local_sparsity",
            "bootstrap_se",
            show_legend=False,
        )
        add_fit(ax, sub["local_sparsity"], sub["bootstrap_se"])

        ax.set_title(
            f"Scenario {short_scenario(scenario)}: local sparsity"
        )
        ax.set_xlabel(
            rf"Local quantile sparsity, "
            rf"$[\hat Q(q+{PRIMARY_SPARSITY_DELTA:g})-"
            rf"\hat Q(q-{PRIMARY_SPARSITY_DELTA:g})]/"
            rf"{2 * PRIMARY_SPARSITY_DELTA:g}$"
        )
        ax.set_ylabel("Bootstrap SE (hospitalizations)")
        ax.grid(alpha=0.20)

        # ---------------------------------------------------------------------
        # Column 2: observed SE vs theory-derived predictor
        # ---------------------------------------------------------------------
        ax = axes[r, 1]
        scatter_all_quantiles(
            ax,
            sub,
            "theory_se_proxy",
            "bootstrap_se",
            show_legend=(r == 0),
        )
        add_fit(ax, sub["theory_se_proxy"], sub["bootstrap_se"])

        both = np.concatenate([
            sub["theory_se_proxy"].to_numpy(dtype=float),
            sub["bootstrap_se"].to_numpy(dtype=float),
        ])
        both = both[np.isfinite(both)]

        if both.size:
            lo = max(0.0, float(np.nanmin(both)))
            hi = float(np.nanmax(both))
            if hi > lo:
                ax.plot(
                    [lo, hi],
                    [lo, hi],
                    linestyle="--",
                    linewidth=1.0,
                    color="black",
                    alpha=0.65,
                    label=None,
                )

        ax.set_title(
            f"Scenario {short_scenario(scenario)}: theory-derived predictor"
        )
        ax.set_xlabel(
            r"$\sqrt{q(1-q)/n}\times$ local sparsity"
        )
        ax.set_ylabel("Observed bootstrap SE (hospitalizations)")
        ax.grid(alpha=0.20)

    fig.suptitle(
        "Mechanistic explanation of finite-trajectory quantile uncertainty",
        fontsize=15,
        fontweight="bold",
    )

    fig.subplots_adjust(
        top=0.94,
        right=0.98,
        hspace=0.28,
        wspace=0.22,
    )

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_theory_three_scenarios(df, output_path):
    scenarios = ordered_scenarios(df["scenario"].unique())

    fig, axes = plt.subplots(
        1,
        len(scenarios),
        figsize=(5.2 * len(scenarios), 5.0),
        squeeze=False,
    )

    for c, scenario in enumerate(scenarios):
        ax = axes[0, c]
        sub = df[df["scenario"] == scenario].copy()

        scatter_all_quantiles(
            ax,
            sub,
            "theory_se_proxy",
            "bootstrap_se",
            show_legend=(c == len(scenarios) - 1),
        )
        add_fit(ax, sub["theory_se_proxy"], sub["bootstrap_se"])

        both = np.concatenate([
            sub["theory_se_proxy"].to_numpy(dtype=float),
            sub["bootstrap_se"].to_numpy(dtype=float),
        ])
        both = both[np.isfinite(both)]

        if both.size:
            lo = max(0.0, float(np.nanmin(both)))
            hi = float(np.nanmax(both))
            if hi > lo:
                ax.plot(
                    [lo, hi],
                    [lo, hi],
                    linestyle="--",
                    linewidth=1.0,
                    color="black",
                    alpha=0.65,
                )

        ax.set_title(f"Scenario {short_scenario(scenario)}")
        ax.set_xlabel(
            r"Theory predictor: $\sqrt{q(1-q)/n}\times$ sparsity"
        )
        ax.set_ylabel("Observed bootstrap SE (hospitalizations)")
        ax.grid(alpha=0.20)

    fig.suptitle(
        "Observed bootstrap SE versus quantile-theory predictor",
        fontsize=14,
        fontweight="bold",
    )

    fig.subplots_adjust(
        top=0.88,
        right=0.98,
        wspace=0.28,
    )

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close(fig)



# =============================================================================
# MAIN
# =============================================================================

def main():
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running sparsity-window sensitivity across candidate deltas...")
    delta_sensitivity = run_delta_sensitivity()

    print(
        f"\nBuilding primary 23-quantile explanatory dataset at "
        f"delta={PRIMARY_SPARSITY_DELTA:g}..."
    )
    df = build_explanatory_dataset(
        sparsity_delta=PRIMARY_SPARSITY_DELTA
    )

    needed = [
        "bootstrap_se",
        "abs_full_quantile_value",
        "local_sparsity",
        "theory_se_proxy",
        "quantile",
        "scenario",
        "model",
        "n_available",
    ]

    reg = (
        df[needed]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    if (reg["bootstrap_se"] < 0).any():
        raise ValueError("bootstrap_se contains negative values.")
    if (reg["local_sparsity"] < 0).any():
        raise ValueError("local_sparsity contains negative values.")
    if (reg["n_available"] <= 0).any():
        raise ValueError("n_available contains non-positive values.")

    observed_q = sorted(reg["quantile"].unique())
    print(f"Quantiles retained ({len(observed_q)}): {observed_q}")
    print(f"Unified raw-scale rows: {len(reg):,}")
    print(f"Zero-SE rows retained: {(reg['bootstrap_se'] == 0).sum():,}")
    print(f"Zero-sparsity rows retained: {(reg['local_sparsity'] == 0).sum():,}")
    print(
        "Zero-theory-proxy rows retained: "
        f"{(reg['theory_se_proxy'] == 0).sum():,}"
    )

    if len(observed_q) != 23:
        raise RuntimeError(
            f"Expected 23 quantiles after complete-case filtering; got "
            f"{len(observed_q)}: {observed_q}"
        )

    dataset_path = (
        output_dir / "14_unified_explanatory_dataset_ALL23Q_delta004.csv"
    )
    reg.to_csv(dataset_path, index=False)

    # -------------------------------------------------------------------------
    # PRIMARY
    # -------------------------------------------------------------------------
    primary_table, primary_fits, primary_text = run_models(
        reg,
        "PRIMARY_ALL_FINITE_RAW_SCALE_ROWS_ALL23Q",
    )

    # -------------------------------------------------------------------------
    # Upper-tail leverage sensitivity
    # -------------------------------------------------------------------------
    se_cut = reg["bootstrap_se"].quantile(
        UPPER_TAIL_SENSITIVITY_QUANTILE
    )
    mag_cut = reg["abs_full_quantile_value"].quantile(
        UPPER_TAIL_SENSITIVITY_QUANTILE
    )
    sparse_cut = reg["local_sparsity"].quantile(
        UPPER_TAIL_SENSITIVITY_QUANTILE
    )
    proxy_cut = reg["theory_se_proxy"].quantile(
        UPPER_TAIL_SENSITIVITY_QUANTILE
    )

    sens = reg[
        (reg["bootstrap_se"] <= se_cut)
        & (reg["abs_full_quantile_value"] <= mag_cut)
        & (reg["local_sparsity"] <= sparse_cut)
        & (reg["theory_se_proxy"] <= proxy_cut)
    ].copy()

    sensitivity_table, _, sensitivity_text = run_models(
        sens,
        "SENSITIVITY_EXCLUDING_VALUES_ABOVE_ANY_99TH_PERCENTILE_ALL23Q",
    )

    model_table = pd.concat(
        [primary_table, sensitivity_table],
        ignore_index=True,
    )

    model_table.to_csv(
        output_dir / "14_unified_model_comparison_ALL23Q_delta004.csv",
        index=False,
    )

    (
        output_dir / "14_unified_regression_summaries_ALL23Q_delta004.txt"
    ).write_text(
        "\n".join(primary_text + sensitivity_text),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Incremental R2
    # -------------------------------------------------------------------------
    def r2(tbl, name):
        return float(
            tbl.loc[tbl["model_name"] == name, "r_squared"].iloc[0]
        )

    increment_rows = []

    for label, tbl in [
        ("PRIMARY_ALL_FINITE_RAW_SCALE_ROWS_ALL23Q", primary_table),
        (
            "SENSITIVITY_EXCLUDING_VALUES_ABOVE_ANY_99TH_PERCENTILE_ALL23Q",
            sensitivity_table,
        ),
    ]:
        r_mag = r2(tbl, "U0_raw_magnitude_only")
        r_sparse = r2(tbl, "U1_raw_sparsity_only")
        r_theory = r2(tbl, "U2_theory_proxy_only")
        r_theory_mag = r2(tbl, "U3_theory_proxy_plus_magnitude")
        r_theory_adj = r2(
            tbl, "U4_theory_proxy_plus_scenario_model"
        )
        r_theory_mag_adj = r2(
            tbl, "U5_theory_proxy_magnitude_plus_scenario_model"
        )
        r_theory_q_adj = r2(
            tbl, "U6_theory_proxy_plus_quantile_scenario_model"
        )
        r_theory_mag_q_adj = r2(
            tbl,
            "U7_theory_proxy_magnitude_plus_quantile_scenario_model",
        )

        increment_rows.append({
            "analysis": label,
            "r2_raw_magnitude_only": r_mag,
            "r2_raw_sparsity_only": r_sparse,
            "r2_theory_proxy_only": r_theory,
            "r2_theory_proxy_plus_magnitude": r_theory_mag,
            "delta_r2_magnitude_beyond_theory_proxy":
                r_theory_mag - r_theory,
            "r2_theory_proxy_plus_scenario_model": r_theory_adj,
            "r2_theory_proxy_magnitude_plus_scenario_model":
                r_theory_mag_adj,
            "delta_r2_magnitude_beyond_theory_proxy_adjusted":
                r_theory_mag_adj - r_theory_adj,
            "r2_theory_proxy_plus_quantile_scenario_model":
                r_theory_q_adj,
            "r2_theory_proxy_magnitude_plus_quantile_scenario_model":
                r_theory_mag_q_adj,
            "delta_r2_magnitude_beyond_theory_proxy_quantile_adjusted":
                r_theory_mag_q_adj - r_theory_q_adj,
        })

    pd.DataFrame(increment_rows).to_csv(
        output_dir / "14_unified_incremental_r2_ALL23Q_delta004.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Theory calibration
    # -------------------------------------------------------------------------
    theory_fit = primary_fits["U2_theory_proxy_only"]

    pd.DataFrame([{
        "n_obs": int(theory_fit.nobs),
        "intercept": float(theory_fit.params["Intercept"]),
        "slope_theory_proxy":
            float(theory_fit.params["theory_se_proxy"]),
        "r_squared": float(theory_fit.rsquared),
        "ideal_intercept_reference": 0.0,
        "ideal_slope_reference": 1.0,
    }]).to_csv(
        output_dir / "14_theory_proxy_calibration_ALL23Q_delta004.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------
    plot_main_two_panel(
        reg,
        output_dir / "14_MAIN_raw_sparsity_and_theory_proxy_ALL23Q_delta004_GROUPED.png",
    )

    plot_theory_three_scenarios(
        reg,
        output_dir / "14_observed_se_vs_theory_proxy_ALL23Q_delta004_GROUPED.png",
    )

    # -------------------------------------------------------------------------
    # Audit
    # -------------------------------------------------------------------------
    audit = [
        "UNIFIED FINITE-TRAJECTORY QUANTILE-SE EXPLANATION -- ALL 23Q",
        "",
        f"Quantiles: {QUANTILES}",
        f"Primary rows: {len(reg)}",
        f"Zero-SE rows retained: {(reg['bootstrap_se'] == 0).sum()}",
        f"Zero-sparsity rows retained: {(reg['local_sparsity'] == 0).sum()}",
        (
            "Zero-theory-proxy rows retained: "
            f"{(reg['theory_se_proxy'] == 0).sum()}"
        ),
        "",
        "Primary sparsity delta:",
        f"  {PRIMARY_SPARSITY_DELTA}",
        "",
        "Delta sensitivity grid:",
        f"  {SPARSITY_DELTAS}",
        "",
        "Theory predictor:",
        (
            "  theory_se_proxy = local_sparsity * "
            "sqrt(q * (1-q) / n_available)"
        ),
        "",
        "Source full-N SE file:",
        f"  {full_n_se_path}",
        "",
        "No bootstrap was rerun.",
        "",
        "99th-percentile sensitivity thresholds:",
        f"  bootstrap_se <= {se_cut}",
        f"  abs_full_quantile_value <= {mag_cut}",
        f"  local_sparsity <= {sparse_cut}",
        f"  theory_se_proxy <= {proxy_cut}",
        f"  sensitivity rows retained = {len(sens)}",
        "",
        (
            "OLS p-values are not the manuscript target because "
            "horizon/quantile cells are dependent."
        ),
    ]

    (
        output_dir / "14_unified_analysis_audit_ALL23Q_delta004.txt"
    ).write_text(
        "\n".join(audit),
        encoding="utf-8",
    )

    print("\nWrote:")
    for name in [
        "14_unified_explanatory_dataset_ALL23Q_delta004.csv",
        "14_unified_model_comparison_ALL23Q_delta004.csv",
        "14_unified_incremental_r2_ALL23Q_delta004.csv",
        "14_theory_proxy_calibration_ALL23Q_delta004.csv",
        "14_unified_regression_summaries_ALL23Q_delta004.txt",
        "14_MAIN_raw_sparsity_and_theory_proxy_ALL23Q_delta004_GROUPED.png",
        "14_observed_se_vs_theory_proxy_ALL23Q_delta004_GROUPED.png",
        "14_unified_analysis_audit_ALL23Q_delta004.txt",
    ]:
        print(" ", output_dir / name)


if __name__ == "__main__":
    main()
