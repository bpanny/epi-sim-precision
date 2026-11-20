# Precision-Based Sample Size Determination for Epidemiological Simulations

This repository contains source code for Precision-Based Sample Size Determination for Epidemiological Simulations; running the analyses requires downloading raw data separately (details below).

## Analysis design

- Influenza Scenario Modeling Hub round `2025-07-29`; scenarios `A-2025-07-29`, `B-2025-07-29`, `C-2025-07-29`; 11 contributing models, 43 forecast horizons.
- Nonparametric bootstrapping: **20,000 trajectory bootstrap samples**, model-specific submitted trajectory count, a 103-level quantile grid (101 centiles including endpoints plus 0.025/0.975).
- Adaptive stopping on HTC cluster: **100 stopping experiments per model–scenario**, **20,000 trajectory bootstrap samples at each checkpoint**, monitored 23 quantiles and all 43 horizons. 
- Ensemble validation compares stopped prefix ensembles from full-submission ensembles.
- DKW CDF-error simulation.
- Quantile convergence analysis.

## Requirements

Python 3 with `numpy`, `pandas`, `matplotlib`, `statsmodels`, and `pyarrow`. R with `tidyverse`, `readr`, `dplyr`, `ggplot2`, `scales`, `reticulate`, `matrixStats`, `here`, `arrow`, `patchwork`, `hubUtils`, `hubEnsembles`, `hubVis`, `CombineDistributions`. Install the Hubverse packages according to Hubverse instructions. 

## 1. Download and prepare SMH trajectories

Download the model-output files from the [Flu Scenario Modeling Hub](https://github.com/midas-network/flu-scenario-modeling-hub) at commit `dc81b85485dd4e268a8be4ef7a87d78cee294a01` ([commit link](https://github.com/midas-network/flu-scenario-modeling-hub/tree/dc81b85485dd4e268a8be4ef7a87d78cee294a01)).

From the root of **this** `epi-sim-precision` repository, with Git installed:

```bash
git clone https://github.com/midas-network/flu-scenario-modeling-hub.git data/hub
git -C data/hub checkout --detach dc81b85485dd4e268a8be4ef7a87d78cee294a01
git -C data/hub rev-parse HEAD
```

The printed commit should be `dc81b85485dd4e268a8be4ef7a87d78cee294a01`. The upstream checkout must contain `data/hub/model-output/` with its submitted Parquet files. 

Data preprocessing is implemented in `scripts/prepare_trajectories.py`. It reads the upstream `model-output/**/*.parquet`, filters to `scenario_id` A/B/C of the `2025-07-29` round, US location, `inc hosp` target, `sample` output type and `0-130` age group, constructs trajectory IDs from `run_grouping` and `stochastic_run`, then pivots into 43-week trajectory matrices.

Run from this repository root:

```bash
python scripts/prepare_trajectories.py --hub-root data/hub --output-dir data/processed
```

It writes six inputs for subsequent analyses:

```text
data/processed/Global_A-2025-07-29_raw_matrix.npy
data/processed/Global_A-2025-07-29_metadata.csv
data/processed/Global_B-2025-07-29_raw_matrix.npy
data/processed/Global_B-2025-07-29_metadata.csv
data/processed/Global_C-2025-07-29_raw_matrix.npy
data/processed/Global_C-2025-07-29_metadata.csv
```

Continue with the quantile uncertainty quantification and quantile-theory analyses:

```bash
python scripts/bootstrap_model_quantiles.py --B 20000
python scripts/theory_sensitivity.py
python scripts/quantile_theory.py
```

Uncertainty quantification results appear in `results_raw_quantile_heterogeneity/full_n_model_bootstrap_unified103/`. 

## 2. Copy processed trajectories to HPC environment

Transfer processed trajectories to HTC environment:

```bash
scp data/processed/Global_*_raw_matrix.npy data/processed/Global_*_metadata.csv \
  username@host:/path/to/epi-sim-precision/data/processed/
```

Replace the SSH username, host, and destination path with those for your HPC environment.

In HPC environment, from the repository root:

```bash
mkdir -p logs
sbatch hpc/htc_iid_20k.sbatch
# after all array tasks finish:
python3 scripts/merge_stopping_results.py --chunks htc_iid_20k/chunks --output-dir htc_iid_20k/analysis
```

Submit from the repository root so `SLURM_SUBMIT_DIR` resolves correctly. The Slurm specification is specific to Pitt CRC (cluster, account, partition and QoS); adapt scheduler directives for another HPC environment. The script executes `hpc/htc_iid_stopping.py`, array `0-99`, 100 stopping experiments, 20,000 bootstrap samples per checkpoint. The merger requires all 100 result and 100 saved-index chunk files and checks experiment/index integrity.

## 3. Copy HPC results back to local environment

Create the target directory and retrieve the two mandatory merged files:

```bash
mkdir -p htc_iid_20k/analysis
scp username@host:/path/to/epi-sim-precision/htc_iid_20k/analysis/replay_checkpoint_quantile_max.csv htc_iid_20k/analysis/
scp username@host:/path/to/epi-sim-precision/htc_iid_20k/analysis/replay_trajectory_indices.csv.gz htc_iid_20k/analysis/
```

Then compute stopping summaries, figures and the stopped-ensemble validation/

```bash
python scripts/stopping_sweep.py
Rscript scripts/plot_adaptive_trajectory_savings.R
Rscript scripts/validate_stopped_ensembles.R
Rscript scripts/plot_stopped_vs_full_vs_official_ensembles.R
```

`stopping_sweep.py` postprocesses HTC checkpoint CSVs and writes joint tolerance/quantile sweep summaries to `htc_iid_20k/analysis/`. `validate_stopped_ensembles.R` writes the caches and summary CSVs to `results_raw_quantile_heterogeneity/prospective_stopped_ensemble_validation/`. The Figure 4 plotter additionally needs the official Hub ensemble data accessed by the Hubverse R packages.

## 4. Additional figure and supplement generation

```bash
python scripts/plot_model_quantile_uncertainty.py --scenario A
python scripts/plot_trajectory_sparsity_precision.py
python scripts/simulate_dkw_cdf_error.py
python scripts/analyze_empirical_quantile_convergence.py --scenario A B C --replays 1000
```

Figure 1 for scenarios B/C can be produced by passing `--scenario B` / `C`. The supplementary joint sweep comes from `scripts/stopping_sweep.py`; supplementary quantile-theory analyses come from `scripts/theory_sensitivity.py` and `scripts/quantile_theory.py`.

## Manuscript result map

| Publication output | Source |
|---|---|
| Main Figure 1 (model quantile SE) | `scripts/plot_model_quantile_uncertainty.py` |
| Main Figure 2 (trajectories, sparsity, quantile SE) | `scripts/plot_trajectory_sparsity_precision.py` |
| Main Figure 3 (trajectory savings) | `scripts/plot_adaptive_trajectory_savings.R` |
| Main Figure 4 (stopped, full, official Hub ensembles) | `scripts/plot_stopped_vs_full_vs_official_ensembles.R` (after validator) |
| Main Figure 5 (DKW simulation) | `scripts/simulate_dkw_cdf_error.py` |
| Supplement theory tables/figure | `scripts/theory_sensitivity.py`, `scripts/quantile_theory.py` |
| Supplement joint stopping sweep table/figure | `scripts/stopping_sweep.py` |
| Supplement empirical CDF gap convergence | `scripts/analyze_empirical_quantile_convergence.py` |

