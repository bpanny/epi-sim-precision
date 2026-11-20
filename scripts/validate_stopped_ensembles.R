#!/usr/bin/env Rscript
# =============================================================================
# 02_validate_prospective_stopped_vs_full_ensembles_AUTO.R
# =============================================================================
#
# PURPOSE
# -------
# 1) Automatically select three representative JOINT prospective stopping
#    operating points closest to target overall median trajectory savings
#    (default 25%, 50%, 75%).
#
# 2) Among configurations equally close to a target, prefer:
#      a) higher criterion-met rate
#      b) broader monitored quantile coverage
#
# 3) Make replay-level stopping plots:
#      x = fraction saved
#      row facet = model
#      column facet = scenario
#    One figure per selected operating point.
#
# 4) Reconstruct the EXACT replay-specific stopped prefixes for every model,
#    rebuild the downstream ensembles, and compare stopped vs full ensembles.
#
# NO operating-points template CSV is required.
#
# IMPORTANT
# ---------
# - Uses replay_stop_results_joint_tau_quantiles.csv from the completed
#   HTC IID 20K checkpoint analysis.
# - Reads the exact IID-with-replacement replay indices saved by the HTC worker.
# - Stopped prefix consists of the first n saved indices (duplicates retained).
# - Full reference uses every submitted trajectory.
# - No bootstrap occurs in the stopped-vs-full validation itself.
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(readr)
  library(reticulate)
  library(hubEnsembles)
  library(hubUtils)
  library(matrixStats)
  library(here)
})
if (!requireNamespace("CombineDistributions", quietly=TRUE)) {
  stop("Package CombineDistributions is required.")
}

Sys.setenv(
  OMP_NUM_THREADS="1",
  MKL_NUM_THREADS="1",
  OPENBLAS_NUM_THREADS="1"
)

# =============================================================================
# PATHS / SETTINGS FOR RSTUDIO
# =============================================================================
#
# This script uses the `here` package rather than getwd()/setwd().
#
# For the most reliable behavior, put this script somewhere inside the
# epi-sim-precision project and make sure the project root
# contains:
#
#   data/processed/
#   htc_iid_20k/analysis/
#
# If you use an .Rproj file at the project root, here() will normally resolve
# the root automatically.
#
# You can inspect it with:
#   here::here()
#
# =============================================================================

args <- commandArgs(trailingOnly=TRUE)

arg_value <- function(flag, default=NULL) {
  hit <- grep(paste0("^", flag, "="), args, value=TRUE)
  if (length(hit)) {
    return(sub(paste0("^", flag, "="), "", hit[[1]]))
  }

  i <- which(args == flag)
  if (length(i) && i[[1]] < length(args)) {
    return(args[[i[[1]] + 1]])
  }

  default
}

parse_num_list <- function(x) {
  as.numeric(strsplit(x, ",", fixed=TRUE)[[1]])
}

# Use the current publication repository as the project root,
# so explicitly descend into the flu Scenario Hub repository.
flu_repo <- normalizePath(getwd(), winslash="/", mustWork=TRUE)

input_dir <- file.path(
  flu_repo,
  arg_value("--input-dir", "data/processed")
)

stopping_dir <- file.path(
  flu_repo,
  arg_value("--stopping-dir", "htc_iid_20k/analysis")
)

output_dir <- file.path(
  flu_repo,
  arg_value(
    "--output-dir",
    "results_raw_quantile_heterogeneity/prospective_stopped_ensemble_validation"
  )
)

target_savings <- parse_num_list(
  arg_value("--target-savings", "0.25,0.50,0.75")
)

# Validation is run sequentially in RStudio for robustness.
# The expensive ensemble operators remain the official package implementations.
workers <- 1L

full_n_ensemble_dir <- file.path(
  flu_repo,
  "results_raw_quantile_heterogeneity",
  "full_n_model_ensemble_bootstrap"
)

message("flu repo:            ", flu_repo)
message("input_dir:           ", input_dir)
message("stopping_dir:        ", stopping_dir)
message("output_dir:          ", output_dir)
message("full-N ensemble SE:  ", full_n_ensemble_dir)

dir.create(
  output_dir,
  recursive=TRUE,
  showWarnings=FALSE
)

# =============================================================================
# SETTINGS
# =============================================================================

EVAL_QUANTILES <- c(
  .01, .025, .05, .10, .15, .20, .25, .30, .35, .40, .45, .50,
  .55, .60, .65, .70, .75, .80, .85, .90, .95, .975, .99
)

TASK_ID_COLS <- c(
  "scenario",
  "location",
  "horizon",
  "target"
)

location_setting <- "US"
target_setting <- "inc hosp"

lop_trim_weight_scheme <- "cdf_exterior"
lop_trim_n <- 2L

np <- reticulate::import(
  "numpy",
  convert=TRUE
)

# =============================================================================
# HELPERS
# =============================================================================

as_quantile_id <- function(q) {
  sapply(q, function(z) {
    out <- format(
      z,
      scientific=FALSE,
      trim=TRUE
    )
    out <- sub("0+$", "", out)
    sub("\\.$", "", out)
  })
}

scenario_short <- function(x) {
  sub("-.*$", "", as.character(x))
}

load_scenario <- function(scenario) {

  raw_path <- file.path(
    input_dir,
    paste0(
      "Global_",
      scenario,
      "_raw_matrix.npy"
    )
  )

  meta_path <- file.path(
    input_dir,
    paste0(
      "Global_",
      scenario,
      "_metadata.csv"
    )
  )

  X <- np$load(raw_path)
  X <- reticulate::py_to_r(X)
  X <- as.matrix(X)

  meta <- read_csv(
    meta_path,
    show_col_types=FALSE
  ) |>
    mutate(
      model=as.character(model)
    )

  if (
    ncol(X) != nrow(meta) &&
    nrow(X) == nrow(meta)
  ) {
    X <- t(X)
  }

  if (ncol(X) != nrow(meta)) {
    stop(
      "Matrix/metadata mismatch for ",
      scenario
    )
  }

  list(
    X=X,
    meta=meta
  )
}

compute_model_quantiles_fast <- function(
  X_h_by_n,
  probs=EVAL_QUANTILES
) {

  qm <- matrixStats::rowQuantiles(
    X_h_by_n,
    probs=probs,
    na.rm=TRUE,
    drop=FALSE
  )

  tibble(
    horizon=rep(
      seq_len(nrow(qm)),
      each=length(probs)
    ),
    quantile=rep(
      probs,
      times=nrow(qm)
    ),
    value=as.vector(t(qm))
  )
}

make_model_rows <- function(
  qdf,
  model_id,
  scenario
) {

  qdf |>
    transmute(
      model_id=as.character(model_id),
      scenario=as.character(scenario),
      location=location_setting,
      horizon=as.integer(horizon),
      target=target_setting,
      output_type="quantile",
      output_type_id=as_quantile_id(quantile),
      value=as.numeric(value)
    )
}

prep_model_out <- function(df) {

  df |>
    mutate(
      output_type_id=as.character(
        as.numeric(output_type_id)
      ),
      horizon=as.integer(horizon),
      location=as.character(location),
      scenario=as.character(scenario),
      target=as.character(target),
      value=as.numeric(value)
    ) |>
    select(
      model_id,
      scenario,
      location,
      horizon,
      target,
      output_type,
      output_type_id,
      value
    ) |>
    hubUtils::as_model_out_tbl()
}

make_median <- function(df) {

  prep_model_out(df) |>
    hubEnsembles::simple_ensemble(
      agg_fun=stats::median,
      model_id="prospective_validation_median",
      task_id_cols=TASK_ID_COLS
    ) |>
    as_tibble() |>
    transmute(
      scenario=as.character(scenario),
      horizon=as.integer(horizon),
      quantile=as.numeric(output_type_id),
      ensemble_method="median_equal_weight",
      value=as.numeric(value)
    )
}

make_lop <- function(
  df,
  trim=FALSE
) {

  method <- if (trim) {
    paste0(
      "lop_trimmed_",
      lop_trim_weight_scheme,
      "_n",
      lop_trim_n
    )
  } else {
    "lop_untrimmed_equal_weight"
  }

  inp <- df |>
    transmute(
      model_id=as.character(model_id),
      scenario=as.character(scenario),
      location=as.character(location),
      horizon=as.integer(horizon),
      target=as.character(target),
      quantile=as.numeric(output_type_id),
      value=as.numeric(value)
    )

  CombineDistributions::aggregate_cdfs(
    data=inp,
    id_var="model_id",
    group_by=c(
      "scenario",
      "location",
      "horizon",
      "target"
    ),
    method="LOP",
    ret_quantiles=EVAL_QUANTILES,
    weighting_scheme=if (trim) {
      lop_trim_weight_scheme
    } else {
      "equal"
    },
    n_trim=as.integer(lop_trim_n)
  ) |>
    as_tibble() |>
    transmute(
      scenario=as.character(scenario),
      horizon=as.integer(horizon),
      quantile=as.numeric(quantile),
      ensemble_method=method,
      value=as.numeric(value)
    )
}

make_ensembles <- function(model_rows) {

  bind_rows(
    make_median(model_rows),
    make_lop(
      model_rows,
      FALSE
    ),
    make_lop(
      model_rows,
      TRUE
    )
  )
}

# -----------------------------------------------------------------------------
# STRICT ENSEMBLE OUTPUT CHECK
# -----------------------------------------------------------------------------
# Official ensemble functions must yield exactly one value for each
# scenario x ensemble_method x horizon x quantile cell.
#
# If a package returns IDENTICAL duplicate rows, collapse them safely.
# If duplicate rows disagree in value, STOP immediately.
# This prevents the previous 11x row multiplication from propagating.
collapse_and_check_ensemble <- function(df, context) {

  key_cols <- c(
    "scenario",
    "ensemble_method",
    "horizon",
    "quantile"
  )

  bad <- df |>
    dplyr::group_by(
      scenario,
      ensemble_method,
      horizon,
      quantile
    ) |>
    dplyr::summarise(
      n_rows=dplyr::n(),
      n_values=dplyr::n_distinct(value),
      .groups="drop"
    ) |>
    dplyr::filter(n_values != 1L)

  if (nrow(bad) > 0L) {
    print(bad, n=Inf)
    stop(
      context,
      ": duplicate ensemble keys contain different values."
    )
  }

  out <- df |>
    dplyr::distinct(
      scenario,
      ensemble_method,
      horizon,
      quantile,
      .keep_all=TRUE
    )

  n_methods <- dplyr::n_distinct(out$ensemble_method)
  n_horizons <- dplyr::n_distinct(out$horizon)
  n_quantiles <- dplyr::n_distinct(out$quantile)

  expected_rows <- 3L * 43L * length(EVAL_QUANTILES)

  if (
    n_methods != 3L ||
    n_horizons != 43L ||
    n_quantiles != length(EVAL_QUANTILES) ||
    nrow(out) != expected_rows
  ) {
    stop(
      context,
      ": bad ensemble grid after duplicate collapse. ",
      "rows=", nrow(out),
      " methods=", n_methods,
      " horizons=", n_horizons,
      " quantiles=", n_quantiles,
      "; expected rows=", expected_rows,
      ", methods=3, horizons=43, quantiles=", length(EVAL_QUANTILES), "."
    )
  }

  out
}

# Read exact 1-based IID trajectory indices saved by the HTC workers.
# Never reconstruct them from replay seeds: duplicate indices are essential.
index_path <- file.path(stopping_dir, "replay_trajectory_indices.csv.gz")
if (!file.exists(index_path)) stop("Missing saved IID replay indices: ", index_path)
message("Reading exact IID trajectory indices: ", index_path)
index_tbl <- readr::read_csv(
  index_path,
  show_col_types=FALSE,
  col_types=readr::cols(
    scenario=readr::col_character(), model=readr::col_character(),
    replay=readr::col_integer(), replay_seed=readr::col_double(),
    n_available=readr::col_integer(), position=readr::col_integer(),
    local_index_1based=readr::col_integer(), sequence_sha256=readr::col_character()
  )
)
if (nrow(index_tbl) != 1458000L) {
  stop("Expected 1,458,000 recorded index rows, got ", nrow(index_tbl))
}
index_keys <- paste(index_tbl$scenario, index_tbl$model, index_tbl$replay, sep="|")
index_lookup <- split(seq_len(nrow(index_tbl)), index_keys)
if (length(index_lookup) != 3300L) stop("Expected 3,300 distinct replay sequences")
rm(index_keys)

prefix_local_indices <- function(N, replay_seed, n_keep, scenario, model, replay) {
  key <- paste(scenario, model, replay, sep="|")
  rows <- index_lookup[[key]]
  if (is.null(rows)) stop("Missing recorded IID sequence: ", key)
  seq_data <- index_tbl[rows, , drop=FALSE]
  seq_data <- seq_data[order(seq_data$position), , drop=FALSE]
  if (nrow(seq_data) != N || !identical(as.integer(seq_data$position), seq_len(N))) {
    stop("Recorded IID positions invalid for ", key)
  }
  if (any(seq_data$n_available != N) ||
      any(seq_data$replay_seed != as.numeric(replay_seed))) {
    stop("Recorded sequence N or seed mismatch for ", key)
  }
  idx <- as.integer(seq_data$local_index_1based)
  if (anyNA(idx) || any(idx < 1L | idx > N)) {
    stop("Recorded local indices out of bounds for ", key)
  }
  if (length(unique(seq_data$sequence_sha256)) != 1L) {
    stop("Recorded SHA256 inconsistent for ", key)
  }
  idx[seq_len(n_keep)]
}

# =============================================================================
# INPUT FILES
# =============================================================================

stop_path <- file.path(
  stopping_dir,
  "replay_stop_results_joint_tau_quantiles.csv"
)

joint_summary_path <- file.path(
  stopping_dir,
  "joint_sweep_summary.csv"
)

checkpoint_path <- file.path(
  stopping_dir,
  "replay_checkpoint_quantile_max.csv"
)

for (p in c(
  stop_path,
  joint_summary_path,
  checkpoint_path
)) {
  if (!file.exists(p)) {
    stop("Missing required file: ", p)
  }
}

# =============================================================================
# AUTO-SELECT OPERATING POINTS
# =============================================================================

joint <- read_csv(
  joint_summary_path,
  show_col_types=FALSE
) |>
  mutate(
    stopping_spec=as.character(stopping_spec),
    tau=as.numeric(tau),
    median_fraction_saved=as.numeric(
      median_fraction_saved
    ),
    criterion_met_rate=as.numeric(
      criterion_met_rate
    ),
    max_required_quantile=as.numeric(
      max_required_quantile
    )
  )

choose_one <- function(target) {

  joint |>
    filter(
      is.finite(median_fraction_saved),
      is.finite(criterion_met_rate)
    ) |>
    mutate(
      target_fraction_saved=target,
      savings_distance=abs(
        median_fraction_saved-target
      )
    ) |>
    arrange(
      savings_distance,
      desc(criterion_met_rate),
      desc(max_required_quantile),
      tau
    ) |>
    slice(1) |>
    mutate(
      operating_point=paste0(
        "target_",
        round(target*100),
        "pct"
      )
    )
}

selected_ops <- bind_rows(
  lapply(
    target_savings,
    choose_one
  )
) |>
  select(
    operating_point,
    target_fraction_saved,
    stopping_spec,
    max_required_quantile,
    tau,
    median_fraction_saved,
    criterion_met_rate,
    median_stop_n,
    everything()
  )

write_csv(
  selected_ops,
  file.path(
    output_dir,
    "selected_operating_points.csv"
  )
)

message(
  "\nAutomatically selected operating points:"
)

print(
  selected_ops |>
    select(
      operating_point,
      target_fraction_saved,
      stopping_spec,
      tau,
      median_fraction_saved,
      criterion_met_rate
    )
)

# =============================================================================
# SELECT REPLAY-LEVEL STOPPING RESULTS
# =============================================================================

stops <- read_csv(
  stop_path,
  show_col_types=FALSE
) |>
  mutate(
    stopping_spec=as.character(stopping_spec),
    tau=as.numeric(tau),
    model=as.character(model),
    scenario=as.character(scenario)
  )

selected_stops <- stops |>
  inner_join(
    selected_ops |>
      select(
        operating_point,
        stopping_spec,
        tau
      ),
    by=c(
      "stopping_spec",
      "tau"
    )
  )

# =============================================================================
# STOPPING DISTRIBUTION FIGURES
#
# user-requested:
# x-axis = fraction saved
# row facet = model
# column facet = scenario
# one figure per selected operating point
# =============================================================================

make_stopping_boxplot <- function(
  op_name
) {

  pdat <- selected_stops |>
    filter(
      operating_point == op_name
    ) |>
    mutate(
      scenario_label=paste0(
        "Scenario ",
        scenario_short(scenario)
      ),
      model=factor(
        model,
        levels=sort(unique(model))
      )
    )

  p <- ggplot(
    pdat,
    aes(
      x=fraction_saved,
      y=1
    )
  ) +
    geom_boxplot(
      aes(group=interaction(model,scenario_label)),
      width=.55,
      outlier.alpha=.25
    ) +
    facet_grid(
      model ~ scenario_label,
      scales="free_y",
      switch="y"
    ) +
    scale_x_continuous(
      labels=scales::percent_format(
        accuracy=1
      ),
      limits=c(0,1)
    ) +
    scale_y_continuous(
      breaks=NULL
    ) +
    labs(
      x="Fraction of trajectories saved",
      y=NULL,
      title=paste0(
        "Replay-level trajectory savings: ",
        op_name
      )
    ) +
    theme_bw(
      base_size=9
    ) +
    theme(
      panel.grid.minor=element_blank(),
      strip.text.y.left=element_text(
        angle=0
      ),
      axis.text.y=element_blank(),
      axis.ticks.y=element_blank()
    )

  ggsave(
    file.path(
      output_dir,
      paste0(
        "FIG5_model_by_scenario_fraction_saved_",
        op_name,
        ".png"
      )
    ),
    p,
    width=11,
    height=max(
      8,
      0.55*length(unique(pdat$model))
    ),
    dpi=300
  )
}

for (
  op_name in unique(
    selected_stops$operating_point
  )
) {
  make_stopping_boxplot(op_name)
}

# Also save compact summaries by model/scenario.
model_savings_summary <- selected_stops |>
  group_by(
    operating_point,
    stopping_spec,
    tau,
    scenario,
    scenario_short,
    model
  ) |>
  summarise(
    n_replays=n_distinct(replay),
    median_fraction_saved=median(
      fraction_saved,
      na.rm=TRUE
    ),
    q25_fraction_saved=quantile(
      fraction_saved,
      .25,
      na.rm=TRUE
    ),
    q75_fraction_saved=quantile(
      fraction_saved,
      .75,
      na.rm=TRUE
    ),
    criterion_met_rate=mean(
      criterion_met,
      na.rm=TRUE
    ),
    median_stop_n=median(
      stop_n,
      na.rm=TRUE
    ),
    n_available=first(
      n_available
    ),
    .groups="drop"
  )

write_csv(
  model_savings_summary,
  file.path(
    output_dir,
    "TABLE_selected_operating_points_by_model_scenario.csv"
  )
)

# =============================================================================
# REPLAY SEED MAP
# =============================================================================

seed_map <- read_csv(
  checkpoint_path,
  show_col_types=FALSE,
  col_select=c(
    scenario,
    model,
    replay,
    replay_seed
  )
) |>
  distinct(
    scenario,
    model,
    replay,
    replay_seed
  ) |>
  mutate(
    scenario=as.character(scenario),
    model=as.character(model)
  )

selected_stops <- selected_stops |>
  left_join(
    seed_map,
    by=c(
      "scenario",
      "model",
      "replay"
    )
  )

if (
  any(
    is.na(
      selected_stops$replay_seed
    )
  )
) {
  stop(
    "Missing replay_seed after joining stopping results to checkpoint file."
  )
}

# =============================================================================
# SCENARIO x OPERATING-POINT VALIDATION TASK
# =============================================================================

run_one_scenario_op <- function(task_row) {

  scenario <- task_row$scenario[[1]]
  op <- task_row$operating_point[[1]]
  spec <- task_row$stopping_spec[[1]]
  tau_value <- task_row$tau[[1]]

  message(
    "START ",
    scenario,
    " | ",
    op,
    " | ",
    spec,
    " | tau=",
    tau_value
  )

  loaded <- load_scenario(
    scenario
  )

  X <- loaded$X
  meta <- loaded$meta

  groups <- split(
    seq_len(nrow(meta)),
    meta$model
  )

  sub_stop <- selected_stops |>
    filter(
      scenario == !!scenario,
      operating_point == !!op,
      stopping_spec == !!spec,
      abs(tau - !!tau_value) < 1e-12
    )

  # ---------------------------------------------------------------------------
  # Full ensemble reference once per scenario
  # ---------------------------------------------------------------------------

  full_model_rows <- lapply(
    names(groups),
    function(m) {

      qdf <- compute_model_quantiles_fast(
        X[
          ,
          groups[[m]],
          drop=FALSE
        ]
      )

      make_model_rows(
        qdf,
        m,
        scenario
      )
    }
  ) |>
    bind_rows()

  full_ens <- make_ensembles(
    full_model_rows
  ) |>
    collapse_and_check_ensemble(
      paste0("FULL ensemble | ", scenario, " | ", op)
    ) |>
    dplyr::rename(
      value_full=value
    )

  replay_ids <- sort(
    unique(
      sub_stop$replay
    )
  )

  err_list <- vector(
    "list",
    length(replay_ids)
  )

  stopped_list <- vector(
    "list",
    length(replay_ids)
  )

  savings_list <- vector(
    "list",
    length(replay_ids)
  )

  for (
    rr in seq_along(
      replay_ids
    )
  ) {

    replay_id <- replay_ids[[rr]]

    rs <- sub_stop |>
      filter(
        replay == !!replay_id
      )

    if (
      nrow(rs) != length(groups)
    ) {
      stop(
        "Expected one stop row/model for ",
        scenario,
        " ",
        op,
        " replay ",
        replay_id,
        "; got ",
        nrow(rs),
        " rows for ",
        length(groups),
        " models."
      )
    }

    stopped_model_rows <- lapply(
      names(groups),
      function(m) {

        z <- rs |>
          filter(
            model == !!m
          )

        if (
          nrow(z) != 1
        ) {
          stop(
            "Missing or duplicate stop row for ",
            scenario,
            " | ",
            m,
            " | replay ",
            replay_id
          )
        }

        cols <- groups[[m]]
        N <- length(cols)

        n_keep <- min(
          as.integer(
            z$stop_n[[1]]
          ),
          N
        )

        local_idx <- prefix_local_indices(
          N=N,
          replay_seed=z$replay_seed[[1]],
          n_keep=n_keep,
          scenario=scenario, model=m, replay=replay_id
        )

        kept_cols <- cols[
          local_idx
        ]

        qdf <- compute_model_quantiles_fast(
          X[
            ,
            kept_cols,
            drop=FALSE
          ]
        )

        make_model_rows(
          qdf,
          m,
          scenario
        )
      }
    ) |>
      bind_rows()

    stop_ens <- make_ensembles(
      stopped_model_rows
    ) |>
      collapse_and_check_ensemble(
        paste0(
          "STOPPED ensemble | ",
          scenario,
          " | ",
          op,
          " | replay ",
          replay_id
        )
      ) |>
      dplyr::rename(
        value_stopped=value
      ) |>
      mutate(
        operating_point=op,
        stopping_spec=spec,
        tau=tau_value,
        replay=replay_id
      )

    stopped_list[[rr]] <- stop_ens

    one_error <- full_ens |>
      dplyr::inner_join(
        stop_ens,
        by=c(
          "scenario",
          "horizon",
          "quantile",
          "ensemble_method"
        )
      ) |>
      dplyr::mutate(
        operating_point=op,
        stopping_spec=spec,
        tau=tau_value,
        replay=replay_id,
        signed_error=value_stopped-value_full,
        abs_error=abs(
          signed_error
        )
      ) |>
      dplyr::select(
        scenario,
        operating_point,
        stopping_spec,
        tau,
        replay,
        ensemble_method,
        horizon,
        quantile,
        value_full,
        value_stopped,
        signed_error,
        abs_error
      )

    expected_replay_rows <- 3L * 43L * length(EVAL_QUANTILES)

    dup_error <- one_error |>
      dplyr::count(
        scenario,
        operating_point,
        replay,
        ensemble_method,
        horizon,
        quantile,
        name="n"
      ) |>
      dplyr::filter(n != 1L)

    if (
      nrow(one_error) != expected_replay_rows ||
      nrow(dup_error) > 0L
    ) {
      if (nrow(dup_error) > 0L) {
        print(dup_error, n=Inf)
      }
      stop(
        "Bad validation grid for ",
        scenario,
        " | ",
        op,
        " | replay ",
        replay_id,
        ": rows=", nrow(one_error),
        "; expected=", expected_replay_rows,
        "."
      )
    }

    err_list[[rr]] <- one_error

    savings_list[[rr]] <- rs |>
      summarise(
        scenario=scenario,
        operating_point=op,
        stopping_spec=spec,
        tau=tau_value,
        replay=replay_id,
        total_full=sum(
          n_available
        ),
        total_retained=sum(
          stop_n
        ),
        fraction_saved=
          1-total_retained/total_full,
        criterion_met_fraction=mean(
          criterion_met
        )
      )

    if (
      rr %% 20L == 0L ||
      rr == length(replay_ids)
    ) {
      message(
        "  ",
        scenario,
        " | ",
        op,
        " replay ",
        rr,
        "/",
        length(replay_ids)
      )
    }
  }

  message(
    "DONE ",
    scenario,
    " | ",
    op
  )

  task_errors <- dplyr::bind_rows(err_list)
  task_stopped <- dplyr::bind_rows(stopped_list)
  task_savings <- dplyr::bind_rows(savings_list)

  expected_task_rows <- length(replay_ids) * 3L * 43L * length(EVAL_QUANTILES)

  if (length(replay_ids) != 100L) {
    stop(
      "Expected 100 replays for ",
      scenario,
      " | ",
      op,
      "; found ",
      length(replay_ids),
      "."
    )
  }

  if (nrow(task_errors) != expected_task_rows) {
    stop(
      "Task row-count failure for ",
      scenario,
      " | ",
      op,
      ": rows=", nrow(task_errors),
      "; expected=", expected_task_rows,
      "."
    )
  }

  list(
    errors=task_errors,
    stopped=task_stopped,
    savings=task_savings,
    full=full_ens |>
      dplyr::mutate(
        operating_point=op
      )
  )
}

# =============================================================================
# TASK LIST + HARD COVERAGE CHECK
# =============================================================================

task_table <- selected_stops |>
  dplyr::distinct(
    scenario,
    operating_point,
    stopping_spec,
    tau
  ) |>
  dplyr::arrange(
    operating_point,
    scenario
  )

expected_scenarios <- c(
  "A-2025-07-29",
  "B-2025-07-29",
  "C-2025-07-29"
)

expected_ops <- c(
  "target_25pct",
  "target_50pct",
  "target_75pct"
)

expected_tasks <- tidyr::crossing(
  scenario=expected_scenarios,
  operating_point=expected_ops
)

missing_tasks <- expected_tasks |>
  dplyr::anti_join(
    task_table |>
      dplyr::select(
        scenario,
        operating_point
      ),
    by=c(
      "scenario",
      "operating_point"
    )
  )

extra_tasks <- task_table |>
  dplyr::select(
    scenario,
    operating_point
  ) |>
  dplyr::anti_join(
    expected_tasks,
    by=c(
      "scenario",
      "operating_point"
    )
  )

if (
  nrow(task_table) != 9L ||
  nrow(missing_tasks) > 0L ||
  nrow(extra_tasks) > 0L
) {
  message("TASK TABLE:")
  print(task_table, n=Inf)

  if (nrow(missing_tasks) > 0L) {
    message("MISSING TASKS:")
    print(missing_tasks, n=Inf)
  }

  if (nrow(extra_tasks) > 0L) {
    message("EXTRA TASKS:")
    print(extra_tasks, n=Inf)
  }

  stop(
    "Validation task grid is not exactly 3 scenarios x 3 operating points."
  )
}

tasks <- split(
  task_table,
  seq_len(nrow(task_table))
)

# =============================================================================
# SAFE RESUMABLE TASK CACHE
# =============================================================================
#
# Each scenario x operating-point task is written to disk immediately.
# If R crashes later, rerun this script: completed valid task files are reused.
# =============================================================================

cache_dir <- file.path(
  output_dir,
  "_validation_task_cache_IID20K_SAVED_INDICES_V1"
)

dir.create(
  cache_dir,
  recursive=TRUE,
  showWarnings=FALSE
)

safe_name <- function(x) {
  gsub("[^A-Za-z0-9._-]+", "_", as.character(x))
}

task_cache_path <- function(tr) {
  file.path(
    cache_dir,
    paste0(
      safe_name(tr$scenario[[1]]),
      "__",
      safe_name(tr$operating_point[[1]]),
      ".rds"
    )
  )
}

validate_cached_task <- function(obj, tr) {

  expected_rows <- 100L * 3L * 43L * length(EVAL_QUANTILES)

  if (
    !is.list(obj) ||
    !all(c("errors", "stopped", "savings", "full") %in% names(obj))
  ) {
    return(FALSE)
  }

  e <- obj$errors

  required <- c(
    "scenario",
    "operating_point",
    "replay",
    "ensemble_method",
    "horizon",
    "quantile"
  )

  if (!all(required %in% names(e))) {
    return(FALSE)
  }

  if (nrow(e) != expected_rows) {
    return(FALSE)
  }

  if (dplyr::n_distinct(e$replay) != 100L) {
    return(FALSE)
  }

  if (dplyr::n_distinct(e$ensemble_method) != 3L) {
    return(FALSE)
  }

  if (dplyr::n_distinct(e$horizon) != 43L) {
    return(FALSE)
  }

  if (dplyr::n_distinct(e$quantile) != length(EVAL_QUANTILES)) {
    return(FALSE)
  }

  dup <- e |>
    dplyr::count(
      scenario,
      operating_point,
      replay,
      ensemble_method,
      horizon,
      quantile,
      name="n"
    ) |>
    dplyr::filter(n != 1L)

  nrow(dup) == 0L
}

atomic_save_rds <- function(obj, path) {

  tmp <- paste0(
    path,
    ".tmp_",
    Sys.getpid()
  )

  saveRDS(
    obj,
    tmp,
    compress=FALSE
  )

  if (file.exists(path)) {
    unlink(path)
  }

  ok <- file.rename(
    tmp,
    path
  )

  if (!ok) {
    stop("Could not atomically move task cache into place: ", path)
  }
}

# =============================================================================
# RUN OR RESUME — 9 TASKS
# =============================================================================
#
# IMPORTANT DESIGN:
#   * One scenario x operating-point task is computed at a time.
#   * The task is validated and atomically saved to RDS immediately.
#   * We DO NOT keep all nine large task objects in memory.
#   * We DO NOT write a multi-gigabyte master CSV.
#   * The nine validated RDS files are the canonical cell-level validation data.
# =============================================================================

for (task_i in seq_along(tasks)) {

  tr <- tasks[[task_i]]
  cache_path <- task_cache_path(tr)

  message("")
  message(
    "VALIDATION TASK ",
    task_i,
    "/",
    length(tasks),
    " | ",
    tr$scenario[[1]],
    " | ",
    tr$operating_point[[1]]
  )

  use_cache <- FALSE

  if (file.exists(cache_path)) {

    cached <- tryCatch(
      readRDS(cache_path),
      error=function(e) NULL
    )

    if (
      !is.null(cached) &&
      validate_cached_task(cached, tr)
    ) {
      message("  USING VALIDATED CACHE: ", cache_path)
      use_cache <- TRUE
    } else {
      message("  INVALID CACHE; deleting and recomputing: ", cache_path)
      rm(cached)
      gc(verbose=FALSE)
      unlink(cache_path)
    }
  }

  if (!use_cache) {

    ans <- run_one_scenario_op(tr)

    if (!validate_cached_task(ans, tr)) {
      stop(
        "Freshly computed task failed validation: ",
        tr$scenario[[1]],
        " | ",
        tr$operating_point[[1]]
      )
    }

    atomic_save_rds(
      ans,
      cache_path
    )

    message("  SAVED VALIDATED CACHE: ", cache_path)

    rm(ans)
    gc(verbose=FALSE)
  } else {
    rm(cached)
    gc(verbose=FALSE)
  }
}

# =============================================================================
# FINAL AUDIT — READ ONE CACHE AT A TIME
# =============================================================================

message("")
message("=== FINAL CACHE AUDIT ===")

manifest_rows <- vector("list", length(tasks))
summary_q_rows <- vector("list", length(tasks))
summary_overall_rows <- vector("list", length(tasks))
summary_replay_rows <- vector("list", length(tasks))
savings_rows <- vector("list", length(tasks))

for (task_i in seq_along(tasks)) {

  tr <- tasks[[task_i]]
  cache_path <- task_cache_path(tr)

  obj <- readRDS(cache_path)

  if (!validate_cached_task(obj, tr)) {
    stop(
      "Previously saved cache failed final audit: ",
      cache_path
    )
  }

  e <- obj$errors

  scenario_i <- tr$scenario[[1]]
  op_i <- tr$operating_point[[1]]

  # One row in the manifest per validated task.
  manifest_rows[[task_i]] <- tibble::tibble(
    scenario=scenario_i,
    operating_point=op_i,
    stopping_spec=tr$stopping_spec[[1]],
    tau=as.numeric(tr$tau[[1]]),
    cache_file=basename(cache_path),
    cache_bytes=as.numeric(file.info(cache_path)$size),
    n_rows=nrow(e),
    n_replays=dplyr::n_distinct(e$replay),
    n_ensemble_methods=dplyr::n_distinct(e$ensemble_method),
    n_horizons=dplyr::n_distinct(e$horizon),
    n_quantiles=dplyr::n_distinct(e$quantile)
  )

  # Compact paper-useful summaries. These are tiny compared with the raw cells.
  summary_q_rows[[task_i]] <- e |>
    dplyr::group_by(
      scenario,
      operating_point,
      stopping_spec,
      tau,
      ensemble_method,
      quantile
    ) |>
    dplyr::summarise(
      n_cells=dplyr::n(),
      n_replays=dplyr::n_distinct(replay),
      median_signed_error=stats::median(signed_error, na.rm=TRUE),
      q25_signed_error=stats::quantile(signed_error, .25, na.rm=TRUE),
      q75_signed_error=stats::quantile(signed_error, .75, na.rm=TRUE),
      median_abs_error=stats::median(abs_error, na.rm=TRUE),
      p90_abs_error=stats::quantile(abs_error, .90, na.rm=TRUE),
      p95_abs_error=stats::quantile(abs_error, .95, na.rm=TRUE),
      max_abs_error=max(abs_error, na.rm=TRUE),
      .groups="drop"
    )

  summary_overall_rows[[task_i]] <- e |>
    dplyr::group_by(
      scenario,
      operating_point,
      stopping_spec,
      tau,
      ensemble_method
    ) |>
    dplyr::summarise(
      n_cells=dplyr::n(),
      n_replays=dplyr::n_distinct(replay),
      median_signed_error=stats::median(signed_error, na.rm=TRUE),
      median_abs_error=stats::median(abs_error, na.rm=TRUE),
      p90_abs_error=stats::quantile(abs_error, .90, na.rm=TRUE),
      p95_abs_error=stats::quantile(abs_error, .95, na.rm=TRUE),
      max_abs_error=max(abs_error, na.rm=TRUE),
      .groups="drop"
    )

  summary_replay_rows[[task_i]] <- e |>
    dplyr::group_by(
      scenario,
      operating_point,
      stopping_spec,
      tau,
      replay,
      ensemble_method
    ) |>
    dplyr::summarise(
      median_signed_error=stats::median(signed_error, na.rm=TRUE),
      median_abs_error=stats::median(abs_error, na.rm=TRUE),
      p95_abs_error=stats::quantile(abs_error, .95, na.rm=TRUE),
      max_abs_error=max(abs_error, na.rm=TRUE),
      .groups="drop"
    )

  savings_rows[[task_i]] <- obj$savings

  message(
    "  PASS ",
    scenario_i,
    " | ",
    op_i,
    " | ",
    format(nrow(e), big.mark=","),
    " unique validation cells"
  )

  rm(obj, e)
  gc(verbose=FALSE)
}

manifest <- dplyr::bind_rows(manifest_rows) |>
  dplyr::arrange(operating_point, scenario)

summary_by_q <- dplyr::bind_rows(summary_q_rows)
summary_overall <- dplyr::bind_rows(summary_overall_rows)
summary_replay <- dplyr::bind_rows(summary_replay_rows)
replay_savings <- dplyr::bind_rows(savings_rows)

# =============================================================================
# FINAL GLOBAL ASSERTIONS
# =============================================================================

if (nrow(manifest) != 9L) {
  stop("Final manifest does not contain exactly 9 scenario x operating-point tasks.")
}

if (any(manifest$n_rows != 100L * 3L * 43L * length(EVAL_QUANTILES))) {
  print(manifest, n=Inf)
  stop("At least one validated task has the wrong row count.")
}

if (
  any(manifest$n_replays != 100L) ||
  any(manifest$n_ensemble_methods != 3L) ||
  any(manifest$n_horizons != 43L) ||
  any(manifest$n_quantiles != length(EVAL_QUANTILES))
) {
  print(manifest, n=Inf)
  stop("At least one validated task has the wrong dimensions.")
}

total_validated_cells <- sum(manifest$n_rows)
expected_total_cells <- 3L * 3L * 100L * 3L * 43L * length(EVAL_QUANTILES)

if (total_validated_cells != expected_total_cells) {
  stop(
    "Final total validation-cell count wrong: got ",
    total_validated_cells,
    "; expected ",
    expected_total_cells,
    "."
  )
}

# =============================================================================
# WRITE ONLY SMALL / MODERATE OUTPUTS
# =============================================================================
#
# NO giant MASTER CSV is written.
# NO duplicate cell-level CSV is written.
# NO figures are generated here.
#
# Canonical cell-level data:
#   _validation_task_cache_SAFE_V2/*.rds
#
# Plotting:
#   02_PLOT_FROM_TASK_CACHES_NO_MASTER.R
# =============================================================================

readr::write_csv(
  manifest,
  file.path(
    output_dir,
    "VALIDATION_TASK_MANIFEST.csv"
  )
)

readr::write_csv(
  summary_by_q,
  file.path(
    output_dir,
    "TABLE_ensemble_validation_by_quantile_ALL_23Q.csv"
  )
)

readr::write_csv(
  summary_overall,
  file.path(
    output_dir,
    "TABLE_ensemble_validation_overall.csv"
  )
)

readr::write_csv(
  summary_replay,
  file.path(
    output_dir,
    "TABLE_ensemble_validation_replay_level.csv"
  )
)

readr::write_csv(
  replay_savings,
  file.path(
    output_dir,
    "replay_level_trajectory_savings.csv"
  )
)

readme_lines <- c(
  "02 prospective stopped-vs-full ensemble validation",
  "",
  "Canonical cell-level outputs:",
  paste0("  ", cache_dir),
  "",
  "There are exactly 9 RDS task files:",
  "  3 scenarios x 3 selected savings targets.",
  "",
  "Each validated task contains:",
  "  100 replays x 3 ensemble methods x 43 horizons x 23 quantiles",
  paste0("  = ", format(100L * 3L * 43L * length(EVAL_QUANTILES), big.mark=","), " unique cell-level error rows."),
  "",
  paste0("Across all 9 tasks: ", format(expected_total_cells, big.mark=","), " validated cells."),
  "",
  "The validator intentionally DOES NOT write a giant combined master CSV.",
  "The validator intentionally DOES NOT make figures.",
  "",
  "Use 02_PLOT_FROM_TASK_CACHES_NO_MASTER.R for the paper plots.",
  "",
  "Cell-level signed error is:",
  "  stopped ensemble quantile - full ensemble quantile",
  "",
  "The existing prospective stopping sweep is NOT recomputed by this validator."
)

writeLines(
  readme_lines,
  con=file.path(
    output_dir,
    "README_VALIDATION_OUTPUTS.txt"
  )
)

# Warn about obsolete giant files if they still exist, but NEVER delete data
# automatically from a scientific workflow.
obsolete_big_files <- c(
  file.path(
    output_dir,
    "MASTER_stopped_vs_full_ensemble_validation_ALL_23Q.csv"
  ),
  file.path(
    output_dir,
    "stopped_vs_full_ensemble_cell_errors.csv"
  )
)

obsolete_existing <- obsolete_big_files[file.exists(obsolete_big_files)]

if (length(obsolete_existing) > 0L) {
  message("")
  message("NOTE: obsolete giant CSV(s) still exist and are NOT needed:")
  for (p in obsolete_existing) {
    message(
      "  ",
      p,
      "  [",
      format(round(file.info(p)$size / 1024^3, 2), nsmall=2),
      " GiB]"
    )
  }
  message("They are not read or written by this validator.")
}

message("")
message("============================================================")
message("VALIDATION COMPLETE")
message("============================================================")
message("Canonical raw data: ", cache_dir)
message("Validated tasks:     9 / 9")
message(
  "Validated cells:     ",
  format(total_validated_cells, big.mark=",")
)
message("Manifest:            ", file.path(output_dir, "VALIDATION_TASK_MANIFEST.csv"))
message("No giant master CSV was created.")
message("No plotting was attempted.")
message("============================================================")
