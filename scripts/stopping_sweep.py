#!/usr/bin/env python3
"""Postprocess saved HTC IID 20,000-bootstrap checkpoint results.

This script ONLY reads existing replay_checkpoint_quantile_max.csv. It never
resamples trajectories or runs bootstrap checkpoints. Outputs preserve the
published stopping sweep CSV/figure filenames.
Run from the repository root.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Sequence
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_TAUS = [1000,1500,2000,2500,3000,4000,5000,6000,7500,10000,12500,15000,20000,30000]
Q = [0.01,0.025,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.975,0.99]
DEFAULT_STOPPING_SPECS = {
    "through_0.90": Q[:20],
    "through_0.95": Q[:21],
    "through_0.975": Q[:22],
    "through_0.99": Q[:23],
}


def parse_float_list(text: str) -> List[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one numeric value.")
    return vals

def stopping_path_for_spec(
    qmax_df: pd.DataFrame,
    stopping_quantiles: Sequence[float],
) -> pd.DataFrame:
    """
    Collapse q-specific horizon-max SEs into a checkpoint max over the selected
    stopping quantiles.
    """
    mask = np.zeros(len(qmax_df), dtype=bool)
    qarr = qmax_df["quantile"].to_numpy(float)
    for q in stopping_quantiles:
        mask |= np.isclose(qarr, q)

    sub = qmax_df.loc[mask].copy()

    rows = []
    group_cols = [
        "scenario", "scenario_short", "model", "replay", "n", "n_available"
    ]

    for keys, g in sub.groupby(group_cols, sort=False):
        vals = g["max_horizon_se"].to_numpy(float)
        if not np.isfinite(vals).any():
            continue
        idx = int(np.nanargmax(vals))
        lim = g.iloc[idx]

        rows.append({
            "scenario": keys[0],
            "scenario_short": keys[1],
            "model": keys[2],
            "replay": int(keys[3]),
            "n": int(keys[4]),
            "n_available": int(keys[5]),
            "stopping_max_se": float(lim["max_horizon_se"]),
            "limiting_quantile": float(lim["quantile"]),
            "limiting_horizon": int(lim["max_se_horizon"]),
        })

    return pd.DataFrame(rows)

def joint_sweep(
    qmax_df: pd.DataFrame,
    taus: Sequence[float],
    stopping_specs: Dict[str, Sequence[float]],
) -> pd.DataFrame:
    records = []

    for spec_name, qset in stopping_specs.items():
        print(f"Postprocessing stopping spec: {spec_name} -> {qset}")
        path = stopping_path_for_spec(qmax_df, qset)

        for keys, sub in path.groupby(
            ["scenario", "scenario_short", "model", "replay"], sort=False
        ):
            scenario, scenario_short, model, replay = keys
            sub = sub.sort_values("n").reset_index(drop=True)
            N = int(sub["n_available"].iloc[0])

            for tau in taus:
                passing = sub[
                    np.isfinite(sub["stopping_max_se"])
                    & (sub["stopping_max_se"] <= float(tau))
                ]

                if not passing.empty:
                    row = passing.iloc[0]
                    criterion_met = True
                    stop_n = int(row["n"])
                else:
                    row = sub.iloc[-1]
                    criterion_met = False
                    stop_n = N

                records.append({
                    "scenario": scenario,
                    "scenario_short": scenario_short,
                    "model": model,
                    "replay": int(replay),
                    "stopping_spec": spec_name,
                    "max_required_quantile": max(qset),
                    "n_stopping_quantiles": len(qset),
                    "tau": float(tau),
                    "criterion_met": bool(criterion_met),
                    "stop_n": int(stop_n),
                    "n_available": int(N),
                    "fraction_saved": float(max(0, (N - stop_n) / N)),
                    "max_se_at_stop_or_full": float(row["stopping_max_se"]),
                    "limiting_quantile_at_stop_or_full": float(row["limiting_quantile"]),
                    "limiting_horizon_at_stop_or_full": int(row["limiting_horizon"]),
                })

    return pd.DataFrame.from_records(records)

def summarize_joint(stop_df: pd.DataFrame):
    joint = (
        stop_df.groupby(
            ["stopping_spec", "max_required_quantile", "tau"], as_index=False
        )
        .agg(
            n_replay_outcomes=("stop_n", "size"),
            criterion_met_rate=("criterion_met", "mean"),
            median_stop_n=("stop_n", "median"),
            q25_stop_n=("stop_n", lambda x: np.quantile(x, 0.25)),
            q75_stop_n=("stop_n", lambda x: np.quantile(x, 0.75)),
            median_fraction_saved=("fraction_saved", "median"),
            q25_fraction_saved=("fraction_saved", lambda x: np.quantile(x, 0.25)),
            q75_fraction_saved=("fraction_saved", lambda x: np.quantile(x, 0.75)),
            mean_fraction_saved=("fraction_saved", "mean"),
        )
        .sort_values(["max_required_quantile", "tau"])
    )

    by_scenario = (
        stop_df.groupby(
            ["scenario", "scenario_short", "stopping_spec",
             "max_required_quantile", "tau"],
            as_index=False
        )
        .agg(
            n_replay_outcomes=("stop_n", "size"),
            criterion_met_rate=("criterion_met", "mean"),
            median_stop_n=("stop_n", "median"),
            median_fraction_saved=("fraction_saved", "median"),
        )
    )

    by_model = (
        stop_df.groupby(
            ["scenario", "scenario_short", "model", "stopping_spec",
             "max_required_quantile", "tau"],
            as_index=False
        )
        .agg(
            n_replays=("replay", "nunique"),
            criterion_met_rate=("criterion_met", "mean"),
            median_stop_n=("stop_n", "median"),
            q25_stop_n=("stop_n", lambda x: np.quantile(x, 0.25)),
            q75_stop_n=("stop_n", lambda x: np.quantile(x, 0.75)),
            median_fraction_saved=("fraction_saved", "median"),
            n_available=("n_available", "first"),
        )
    )

    lim = (
        stop_df[stop_df["criterion_met"]]
        .groupby(
            ["stopping_spec", "max_required_quantile", "tau",
             "limiting_quantile_at_stop_or_full"],
            as_index=False,
        )
        .size()
        .rename(columns={
            "limiting_quantile_at_stop_or_full": "limiting_quantile",
            "size": "count",
        })
    )
    if not lim.empty:
        denom = lim.groupby(
            ["stopping_spec", "tau"]
        )["count"].transform("sum")
        lim["fraction"] = lim["count"] / denom

    return joint, by_scenario, by_model, lim

def incremental_tail_cost(stop_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare nested stopping specs replay-by-replay:
      through .95  - through .90
      through .975 - through .95
      through .99  - through .975
    """
    order = ["through_0.90", "through_0.95", "through_0.975", "through_0.99"]

    wide = (
        stop_df.pivot_table(
            index=["scenario", "scenario_short", "model", "replay", "tau"],
            columns="stopping_spec",
            values="stop_n",
            aggfunc="first",
        )
        .reset_index()
    )

    rows = []
    for prev, curr in zip(order[:-1], order[1:]):
        if prev not in wide.columns or curr not in wide.columns:
            continue
        tmp = wide[
            ["scenario", "scenario_short", "model", "replay", "tau", prev, curr]
        ].copy()
        tmp["from_spec"] = prev
        tmp["to_spec"] = curr
        tmp["incremental_n"] = tmp[curr] - tmp[prev]
        rows.append(tmp[
            ["scenario", "scenario_short", "model", "replay", "tau",
             "from_spec", "to_spec", "incremental_n"]
        ])

    detailed = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    if detailed.empty:
        return detailed, detailed

    summary = (
        detailed.groupby(["from_spec", "to_spec", "tau"], as_index=False)
        .agg(
            median_incremental_n=("incremental_n", "median"),
            q25_incremental_n=("incremental_n", lambda x: np.quantile(x, 0.25)),
            q75_incremental_n=("incremental_n", lambda x: np.quantile(x, 0.75)),
            mean_incremental_n=("incremental_n", "mean"),
            fraction_with_extra_cost=("incremental_n", lambda x: np.mean(np.asarray(x) > 0)),
        )
    )
    return detailed, summary

def heatmap_from_summary(
    summary: pd.DataFrame,
    value_col: str,
    title: str,
    cbar_label: str,
    path: Path,
    dpi: int,
    percent: bool = False,
):
    pivot = summary.pivot(
        index="max_required_quantile",
        columns="tau",
        values=value_col
    ).sort_index()

    data = pivot.to_numpy(float)
    if percent:
        data = data * 100.0

    fig, ax = plt.subplots(figsize=(11, 4.8))
    im = ax.imshow(data, aspect="auto", origin="lower")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{x:g}" for x in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{x:g}" for x in pivot.index])

    ax.set_xlabel(r"Raw bootstrap-SE threshold $\tau$ (hospitalizations)")
    ax.set_ylabel("Highest mandatory quantile")
    ax.set_title(title)

    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar_label)

    # Annotate cells.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                text = f"{v:.0f}%" if percent else f"{v:.2f}" if "fraction" in value_col else f"{v:.0f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def plot_incremental_tail_cost(summary: pd.DataFrame, path: Path, dpi: int):
    if summary.empty:
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    pairs = summary[["from_spec", "to_spec"]].drop_duplicates().itertuples(index=False)
    for pair in pairs:
        sub = summary[
            (summary["from_spec"] == pair.from_spec)
            & (summary["to_spec"] == pair.to_spec)
        ].sort_values("tau")

        label = f"{pair.from_spec.replace('through_', '')} → {pair.to_spec.replace('through_', '')}"
        ax.plot(
            sub["tau"],
            sub["median_incremental_n"],
            marker="o",
            label=label,
        )

    ax.set_xlabel(r"Raw bootstrap-SE threshold $\tau$ (hospitalizations)")
    ax.set_ylabel("Median additional trajectories required")
    ax.set_title("Incremental simulation cost of extending required tail coverage")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, title="Required tail extension")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("htc_iid_20k/analysis"))
    parser.add_argument("--taus", default=",".join(str(x) for x in DEFAULT_TAUS))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    output_dir = args.output_dir
    checkpoint_path = output_dir / "replay_checkpoint_quantile_max.csv"
    if not checkpoint_path.is_file():
        parser.error(f"Missing HTC checkpoint output: {checkpoint_path}")
    taus = sorted(set(parse_float_list(args.taus)))
    print(f"Reading saved HTC checkpoints: {checkpoint_path}")
    qmax_df = pd.read_csv(checkpoint_path)
    required = {"scenario","scenario_short","model","replay","n","n_available","quantile","max_horizon_se","max_se_horizon"}
    missing = required.difference(qmax_df.columns)
    if missing:
        parser.error(f"Checkpoint table missing columns: {sorted(missing)}")
    print("\nRunning cheap joint tau x stopping-grid sweep...")
    stop_df = joint_sweep(
        qmax_df=qmax_df,
        taus=taus,
        stopping_specs=DEFAULT_STOPPING_SPECS,
    )

    joint, by_scenario, by_model, lim = summarize_joint(stop_df)
    inc_detail, inc_summary = incremental_tail_cost(stop_df)

    stop_df.to_csv(
        output_dir / "replay_stop_results_joint_tau_quantiles.csv",
        index=False,
    )
    joint.to_csv(output_dir / "joint_sweep_summary.csv", index=False)
    by_scenario.to_csv(
        output_dir / "joint_sweep_summary_by_scenario.csv",
        index=False,
    )
    by_model.to_csv(
        output_dir / "model_scenario_joint_summary.csv",
        index=False,
    )
    lim.to_csv(
        output_dir / "limiting_quantile_frequency_joint.csv",
        index=False,
    )
    inc_detail.to_csv(
        output_dir / "incremental_tail_cost_replay_level.csv",
        index=False,
    )
    inc_summary.to_csv(
        output_dir / "incremental_tail_cost_summary.csv",
        index=False,
    )

    heatmap_from_summary(
        joint,
        "median_fraction_saved",
        "Prospective stopping: median fraction of trajectories saved",
        "Median fraction saved",
        output_dir / "FIG_joint_heatmap_median_fraction_saved.png",
        args.dpi,
        percent=False,
    )

    heatmap_from_summary(
        joint,
        "median_stop_n",
        "Prospective stopping: median trajectory requirement",
        "Median stopping n",
        output_dir / "FIG_joint_heatmap_median_stop_n.png",
        args.dpi,
        percent=False,
    )

    heatmap_from_summary(
        joint,
        "criterion_met_rate",
        "Prospective stopping feasibility",
        "Replay outcomes meeting criterion (%)",
        output_dir / "FIG_joint_heatmap_criterion_met_rate.png",
        args.dpi,
        percent=True,
    )

    plot_incremental_tail_cost(
        inc_summary,
        output_dir / "FIG_incremental_tail_cost_by_tau.png",
        args.dpi,
    )

    print("\nDone. Main outputs:")
    for name in [
        "replay_checkpoint_quantile_max.csv",
        "replay_stop_results_joint_tau_quantiles.csv",
        "joint_sweep_summary.csv",
        "joint_sweep_summary_by_scenario.csv",
        "model_scenario_joint_summary.csv",
        "limiting_quantile_frequency_joint.csv",
        "incremental_tail_cost_summary.csv",
        "FIG_joint_heatmap_median_fraction_saved.png",
        "FIG_joint_heatmap_median_stop_n.png",
        "FIG_joint_heatmap_criterion_met_rate.png",
        "FIG_incremental_tail_cost_by_tau.png",
    ]:
        print(" ", output_dir / name)

    print("\nRecompute the sweep with another tolerance grid using:")
    print(
        f"  python scripts/stopping_sweep.py --output-dir {output_dir} "
        '--taus "4000,5000,6000,7000,8000,9000,10000"'
    )


if __name__ == "__main__":
    main()
