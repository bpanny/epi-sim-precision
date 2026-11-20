#!/usr/bin/env python3
"""Prepare national incident-hospitalization trajectory matrices from Hub parquet.

This retains the exact filtering, pivoting, parquet-file sorting, concatenation,
and CSV column order of precompute_model_distances.py (January 2026), omitting
only pairwise distance computations unused by this manuscript.
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

SCENARIOS = ("A-2025-07-29", "B-2025-07-29", "C-2025-07-29")
EXPECTED_MODELS = {
    "ACCIDDA-FlepiMoP": 600, "CEPH-MetaFlu": 300,
    "CFA-Scenarios": 500, "MOBS_NEU-GLEAM_FLU": 600,
    "NIH-Flu_TS": 500, "OHT_JHU-gpop": 360,
    "PSI-M3": 600, "SigSci-SWIFT": 500,
    "UNCC-Hierbin": 300, "UT-ImmunoSEIRS": 300,
    "UVA-FluXSim": 300,
}



def load_scenario_slice(filepath, scenario_id):
    """Original loading/pivot semantics; return (43 x N, run IDs, horizons)."""
    try:
        try:
            meta_df = pd.read_parquet(filepath, columns=["scenario_id"])
            if scenario_id not in meta_df["scenario_id"].unique():
                return None, None, None
        except Exception:
            return None, None, None

        df = pd.read_parquet(filepath)
        subset = df[
            (df["scenario_id"] == scenario_id)
            & (df["location"] == "US")
            & (df["target"] == "inc hosp")
            & (df["output_type"] == "sample")
            & (df["age_group"] == "0-130")
        ].copy()
        if subset.empty:
            return None, None, None
        subset["run_id"] = (
            subset["run_grouping"].astype(str)
            + "_"
            + subset["stochastic_run"].astype(str)
        )
        matrix_df = subset.pivot(
            index="horizon", columns="run_id", values="value"
        ).sort_index()
        return matrix_df.values, matrix_df.columns.values, matrix_df.index.values
    except Exception as exc:
        print(f"  Error loading {filepath}: {exc}")
        return None, None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub-root", type=Path, default=Path("data/hub"),
                        help="Hub checkout directory containing model-output/")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    model_dir = args.hub_root / "model-output"
    files = sorted(glob.glob(os.path.join(str(model_dir), "**", "*.parquet"), recursive=True))
    if not files:
        parser.error(f"No parquet files found under {model_dir}. Download the pinned Hub snapshot first.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for scenario_id in SCENARIOS:
        print(f"\n=== Preparing {scenario_id} ===")
        matrices = []
        metadata = []
        reference_horizons = None
        for filepath in files:
            model_name = os.path.basename(os.path.dirname(filepath))
            if "Ensemble" in model_name:
                continue
            mat, runs, times = load_scenario_slice(filepath, scenario_id)
            if mat is None:
                continue
            if reference_horizons is None:
                reference_horizons = times
            if not np.array_equal(times, reference_horizons):
                print(f"  Warning: {model_name} has different time horizon. Skipping.")
                continue
            matrices.append(mat)
            metadata.extend({"model": model_name, "run_id": r, "scenario": scenario_id} for r in runs)
            print(f"  Loaded {mat.shape[1]} runs from {model_name}")

        if not matrices:
            raise RuntimeError(f"No eligible model trajectories for {scenario_id}")
        observed = pd.Series([r["model"] for r in metadata]).value_counts().to_dict()
        if observed != EXPECTED_MODELS:
            raise RuntimeError(
                f"Unexpected model counts for {scenario_id}: {observed}; "
                f"expected {EXPECTED_MODELS}. Check pinned Hub data."
            )
        if len(reference_horizons) != 43 or not np.array_equal(reference_horizons, np.arange(1, 44)):
            raise RuntimeError(f"Unexpected horizons for {scenario_id}: {reference_horizons}")
        matrix = np.hstack(matrices)
        meta = pd.DataFrame(metadata)
        meta["global_index"] = range(len(meta))
        prefix = f"Global_{scenario_id}"
        npy_path = args.output_dir / f"{prefix}_raw_matrix.npy"
        csv_path = args.output_dir / f"{prefix}_metadata.csv"
        np.save(npy_path, matrix)
        meta.to_csv(csv_path, index=False)
        print(f"  Saved: {matrix.shape}; models={meta['model'].nunique()}")

    print("\nDONE: all three scenarios prepared.")


if __name__ == "__main__":
    main()
