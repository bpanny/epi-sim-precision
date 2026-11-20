#!/usr/bin/env Rscript
# =============================================================================
# 02_PLOT_75PCT_5ROW_WITH_OFFICIAL_HUB.R
# =============================================================================
# PURPOSE
#   Make one figure per scenario for the ~75% savings operating point using one
#   reproducibly selected replay.
#
#   IMPORTANT:
#   - The TOP THREE ROWS are plotted with hubVis itself.
#   - The BOTTOM TWO ROWS are custom ggplot panels, because hubVis does not have
#     a native geometry for "upper endpoint deviation" and "lower endpoint
#     deviation" as separate rows.
#
#   Layout per scenario:
#     Row 1: Stopped ensemble predictive distributions   [hubVis]
#     Row 2: Full ensemble predictive distributions      [hubVis]
#     Row 3: Official Hub ensemble predictive distributions [hubVis]
#     Row 4: Upper PI endpoint deviations (Stopped-Full) [ggplot]
#     Row 5: Lower PI endpoint deviations (Stopped-Full) [ggplot]
#
#   Columns:
#     Quantile median | LOP untrimmed | LOP trimmed
#
#   Intervals:
#     50%, 80%, 90%, 95%
#
#   x-axis:
#     Forecast horizon (week), 1-43
#
#   Right-hand secondary y-axis:
#     Rates per 100,000 using U.S. population 341,784,857
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(here)
  library(readr)
  library(arrow)
  library(hubUtils)
  library(hubVis)
})

if (!requireNamespace("patchwork", quietly = TRUE)) {
  stop(
    "Package 'patchwork' is required.\n",
    "Install once with: install.packages('patchwork')"
  )
}

flu_repo <- normalizePath(getwd(), winslash="/", mustWork=TRUE)

validation_dir <- file.path(
  flu_repo,
  "results_raw_quantile_heterogeneity",
  "prospective_stopped_ensemble_validation"
)

cache_dir <- file.path(
  validation_dir,
  "_validation_task_cache_IID20K_SAVED_INDICES_V1"
)

out_dir <- file.path(
  validation_dir,
  "paper_75pct_5row_with_official_hub"
)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

if (!dir.exists(cache_dir)) {
  stop("Missing validation task-cache directory: ", cache_dir)
}

scenario_map <- c(
  "A-2025-07-29" = "Scenario A",
  "B-2025-07-29" = "Scenario B",
  "C-2025-07-29" = "Scenario C"
)

scenario_short_map <- c(
  "A-2025-07-29" = "A",
  "B-2025-07-29" = "B",
  "C-2025-07-29" = "C"
)

method_map <- c(
  "median_equal_weight" = "Quantile median",
  "lop_untrimmed_equal_weight" = "LOP untrimmed",
  "lop_trimmed_cdf_exterior_n2" = "LOP trimmed"
)

OPERATING_POINT <- "target_75pct"
RANDOM_SEED <- 20260908L
US_POP_2025 <- 341784857

# Exact interval colors from the user's prior "better colors" script.
INTERVAL_COLORS <- c(
  "50%" = "#0B3C5D",
  "80%" = "#1D6FA5",
  "90%" = "#4FA3C8",
  "95%" = "#8CC7D9"
)

# hubVis maps fill to model_id in these one-model top-row calls, so force the
# model fill to the same blue family rather than accepting hubVis's default green.
HUBVIS_MODEL_BLUE <- "#1D6FA5"

interval_defs <- tibble::tribble(
  ~interval_label, ~lower_q, ~upper_q, ~interval_order,
  "50%",             0.25,     0.75,      1L,
  "80%",             0.10,     0.90,      2L,
  "90%",             0.05,     0.95,      3L,
  "95%",             0.025,    0.975,     4L
)

INTERVAL_LINE_COLORS <- INTERVAL_COLORS

expected_task_rows <- 100L * 3L * 43L * 23L

safe_name <- function(x) {
  gsub("[^A-Za-z0-9._-]+", "_", as.character(x))
}

cache_path <- function(scenario) {
  file.path(
    cache_dir,
    paste0(
      safe_name(scenario),
      "__",
      OPERATING_POINT,
      ".rds"
    )
  )
}

# -----------------------------------------------------------------------------
# Choose one replay reproducibly from the first scenario cache.
# -----------------------------------------------------------------------------
seed_file <- cache_path(names(scenario_map)[1])

if (!file.exists(seed_file)) {
  stop("Missing seed cache: ", seed_file)
}

seed_obj <- readRDS(seed_file)

if (!is.list(seed_obj) || !"errors" %in% names(seed_obj)) {
  stop("Invalid seed cache: ", seed_file)
}

available_replays <- sort(unique(seed_obj$errors$replay))

if (length(available_replays) != 100L) {
  stop("Expected 100 replays; found ", length(available_replays))
}

set.seed(RANDOM_SEED)
SELECTED_REPLAY <- sample(available_replays, size = 1)

message("Selected replay: ", SELECTED_REPLAY)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
# Official round-6 Hub submissions: one column per ensemble operator.
# Keep these paths rooted exactly as the existing script does.
official_files <- c(
  "median_equal_weight" = file.path(
    flu_repo, "data", "hub", "model-output", "Ensemble", "2025-08-10-Ensemble.gz.parquet"
  ),
  "lop_untrimmed_equal_weight" = file.path(
    flu_repo, "data", "hub", "model-output", "Ensemble_LOP_untrimmed",
    "2025-08-10-Ensemble_LOP_untrimmed.gz.parquet"
  ),
  "lop_trimmed_cdf_exterior_n2" = file.path(
    flu_repo, "data", "hub", "model-output", "Ensemble_LOP", "2025-08-10-Ensemble_LOP.gz.parquet"
  )
)

missing_official <- official_files[!file.exists(official_files)]
if (length(missing_official)) {
  stop("Missing official Hub ensemble parquet file(s):\n",
       paste(missing_official, collapse = "\n"))
}

# Only quantiles used for the four central predictive intervals (plus median)
# need to be present in the official submissions.
plot_quantiles <- sort(unique(c(0.5, interval_defs$lower_q, interval_defs$upper_q)))

official_all <- dplyr::bind_rows(lapply(names(official_files), function(method) {
  raw <- arrow::read_parquet(official_files[[method]])
  required <- c("scenario_id", "location", "target", "age_group",
                "output_type", "output_type_id", "horizon", "value")
  absent <- setdiff(required, names(raw))
  if (length(absent)) {
    stop("Official file missing columns for ", method, ": ",
         paste(absent, collapse = ", "))
  }
  raw |>
    dplyr::filter(
      .data$scenario_id %in% names(scenario_map),
      .data$location == "US",
      .data$target == "inc hosp",
      .data$age_group == "0-130",
      .data$output_type == "quantile"
    ) |>
    dplyr::transmute(
      scenario = as.character(.data$scenario_id),
      ensemble_method = method,
      horizon = as.numeric(.data$horizon),
      quantile = as.numeric(.data$output_type_id),
      value_official = as.numeric(.data$value)
    ) |>
    dplyr::filter(!is.na(.data$quantile),
                  round(.data$quantile, 6) %in% round(plot_quantiles, 6)) |>
    dplyr::mutate(quantile = round(.data$quantile, 6))
}))

# Fail rather than silently plotting missing quantiles, duplicate submissions,
# or different horizons. Expected: 3 scenarios x 3 methods x 43 weeks x 9 q.
duplicates <- official_all |>
  dplyr::count(.data$scenario, .data$ensemble_method,
               .data$horizon, .data$quantile) |>
  dplyr::filter(.data$n != 1L)
if (nrow(duplicates)) stop("Duplicate official ensemble quantile entries.")
expected_official_rows <- length(scenario_map) * length(method_map) * 43L *
  length(plot_quantiles)
if (nrow(official_all) != expected_official_rows ||
    anyNA(official_all$value_official)) {
  stop("Official ensemble values are missing, incomplete, or non-numeric: found ",
       nrow(official_all), " rows; expected ", expected_official_rows,
       ". Check original parquet quantile levels and filtering criteria.")
}

build_hubvis_tbl <- function(d, which_value = c("stopped", "full", "official")) {
  which_value <- match.arg(which_value)

  value_col <- switch(which_value, stopped="value_stopped", full="value_full", official="value_official")
  model_lab <- switch(which_value, stopped="Stopped", full="Full", official="Official Hub")

  d |>
    transmute(
      model_id = model_lab,
      horizon = as.numeric(horizon),
      method_label = factor(
        method_label,
        levels = unname(method_map)
      ),
      output_type = "quantile",
      output_type_id = as.numeric(quantile),
      value = as.numeric(.data[[value_col]])
    ) |>
    hubUtils::as_model_out_tbl()
}

add_common_hubvis_axis <- function(p, show_x_text = FALSE) {
  p +
    ggplot2::scale_y_continuous(
      labels = scales::label_comma(accuracy = 1),
      sec.axis = ggplot2::sec_axis(
        ~ . * 100000 / US_POP_2025,
        name = "Incident hospitalizations per 100,000"
      )
    ) +
    ggplot2::scale_x_continuous(
      breaks = c(1, 5, 10, 15, 20, 25, 30, 35, 40, 43),
      limits = c(1, 43),
      minor_breaks = NULL
    ) +
    ggplot2::labs(
      x = if (show_x_text) "Forecast horizon (week)" else NULL,
      y = "Incident hospitalizations"
    ) +
    ggplot2::theme_bw(base_size = 11) +
    ggplot2::theme(
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = 14),
      panel.grid.minor = element_blank(),
      legend.position = "bottom",
      panel.spacing.x = grid::unit(0.65, "lines"),
      axis.text.x = if (show_x_text) element_text(size = 12.5) else element_blank(),
      axis.ticks.x = if (show_x_text) element_line() else element_blank(),
      axis.text.y = element_text(size = 12.5, margin = margin(r = 0)),
      axis.ticks.y = element_line(),
      axis.title.y = element_text(margin = margin(r = 4)),
      axis.ticks.length = grid::unit(0.08, "cm"),
      plot.margin = margin(1, 4, 1, 1)
    )
}

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
for (sc in names(scenario_map)) {

  message("")
  message("Building figure for ", scenario_map[[sc]])

  path <- cache_path(sc)
  if (!file.exists(path)) {
    stop("Missing validated cache: ", path)
  }

  obj <- readRDS(path)

  if (!is.list(obj) || !"errors" %in% names(obj)) {
    stop("Invalid cache object: ", path)
  }

  d <- obj$errors

  needed <- c(
    "scenario",
    "operating_point",
    "replay",
    "ensemble_method",
    "horizon",
    "quantile",
    "value_full",
    "value_stopped"
  )

  missing_cols <- setdiff(needed, names(d))
  if (length(missing_cols) > 0L) {
    stop("Missing columns: ", paste(missing_cols, collapse = ", "))
  }

  if (nrow(d) != expected_task_rows) {
    stop(
      "Bad cache row count for ", sc,
      ": got ", nrow(d),
      "; expected ", expected_task_rows
    )
  }

  ds <- d |>
    filter(replay == SELECTED_REPLAY) |>
    mutate(
      horizon = as.numeric(horizon),
      quantile = as.numeric(quantile),
      method_label = factor(
        unname(method_map[ensemble_method]),
        levels = unname(method_map)
      )
    )

  # Pair this replay's stopped reconstruction with the published Hub ensembles.
  # The original full-reconstruction values stay in ds for auditability only.
  official_sc <- official_all |>
    dplyr::filter(.data$scenario == sc)
  official_ds <- ds |>
    dplyr::inner_join(
      official_sc,
      by = c("scenario", "ensemble_method", "horizon", "quantile")
    )
  expected_paired <- length(method_map) * 43L * length(plot_quantiles)
  if (nrow(official_ds) != expected_paired) {
    stop("Stopped/official join incomplete for ", sc,
         ": got ", nrow(official_ds), "; expected ", expected_paired)
  }

  # ---------------------------------------------------------------------------
  # TOP ROW 1: stopped ensemble via hubVis
  # ---------------------------------------------------------------------------
  stopped_tbl <- build_hubvis_tbl(ds, "stopped")

  p_stopped <- hubVis::plot_step_ahead_model_output(
    model_out_tbl = stopped_tbl,
    target_data = data.frame(),
    use_median_as_point = TRUE,
    intervals = c(0.50, 0.80, 0.90, 0.95),
    show_plot = FALSE,
    plot_target = FALSE,
    x_col_name = "horizon",
    show_legend = FALSE,
    facet = "method_label",
    facet_scales = "fixed",
    facet_nrow = 1,
    facet_ncol = 3,
    interactive = FALSE,
    fill_by = "model_id"
  )

  p_stopped <- p_stopped +
    ggplot2::scale_fill_manual(values = c("Stopped" = HUBVIS_MODEL_BLUE)) +
    ggplot2::scale_color_manual(values = c("Stopped" = HUBVIS_MODEL_BLUE))

  p_stopped <- add_common_hubvis_axis(p_stopped, show_x_text = FALSE) +
    ggplot2::geom_label(
      data = data.frame(
        method_label = factor(unname(method_map), levels = unname(method_map)),
        panel_letter = c("A", "B", "C")
      ),
      mapping = ggplot2::aes(x = -Inf, y = Inf, label = panel_letter),
      inherit.aes = FALSE, hjust = -0.3, vjust = 1.3,
      fontface = "bold", size = 4.2,
      fill = "white", label.size = 0, label.padding = grid::unit(0.12, "lines")
    ) +
    ggplot2::theme(
      legend.position = "none"
    )

  # ---------------------------------------------------------------------------
  # TOP ROW 2: full ensemble via hubVis
  # ---------------------------------------------------------------------------
  full_tbl <- build_hubvis_tbl(ds, "full")

  p_full <- hubVis::plot_step_ahead_model_output(
    model_out_tbl = full_tbl,
    target_data = data.frame(),
    use_median_as_point = TRUE,
    intervals = c(0.50, 0.80, 0.90, 0.95),
    show_plot = FALSE,
    plot_target = FALSE,
    x_col_name = "horizon",
    show_legend = FALSE,
    facet = "method_label",
    facet_scales = "fixed",
    facet_nrow = 1,
    facet_ncol = 3,
    interactive = FALSE,
    fill_by = "model_id"
  )

  p_full <- p_full +
    ggplot2::scale_fill_manual(values = c("Full" = HUBVIS_MODEL_BLUE)) +
    ggplot2::scale_color_manual(values = c("Full" = HUBVIS_MODEL_BLUE))

  p_full <- add_common_hubvis_axis(p_full, show_x_text = FALSE) +
    ggplot2::geom_label(
      data = data.frame(
        method_label = factor(unname(method_map), levels = unname(method_map)),
        panel_letter = c("D", "E", "F")
      ),
      mapping = ggplot2::aes(x = -Inf, y = Inf, label = panel_letter),
      inherit.aes = FALSE, hjust = -0.3, vjust = 1.3,
      fontface = "bold", size = 4.2,
      fill = "white", label.size = 0, label.padding = grid::unit(0.12, "lines")
    ) +
    ggplot2::theme(
      strip.text.x = element_blank(),
      legend.position = "none"
    )

  # ---------------------------------------------------------------------------
  # TOP ROW 3: official Hub ensemble via hubVis
  # ---------------------------------------------------------------------------
  official_tbl <- build_hubvis_tbl(official_ds, "official")

  p_official <- hubVis::plot_step_ahead_model_output(
    model_out_tbl = official_tbl,
    target_data = data.frame(),
    use_median_as_point = TRUE,
    intervals = c(0.50, 0.80, 0.90, 0.95),
    show_plot = FALSE,
    plot_target = FALSE,
    x_col_name = "horizon",
    show_legend = FALSE,
    facet = "method_label",
    facet_scales = "fixed",
    facet_nrow = 1,
    facet_ncol = 3,
    interactive = FALSE,
    fill_by = "model_id"
  )

  p_official <- p_official +
    ggplot2::scale_fill_manual(values = c("Official Hub" = HUBVIS_MODEL_BLUE)) +
    ggplot2::scale_color_manual(values = c("Official Hub" = HUBVIS_MODEL_BLUE))

  p_official <- add_common_hubvis_axis(p_official, show_x_text = FALSE) +
    ggplot2::geom_label(
      data = data.frame(
        method_label = factor(unname(method_map), levels = unname(method_map)),
        panel_letter = c("G", "H", "I")
      ),
      mapping = ggplot2::aes(x = -Inf, y = Inf, label = panel_letter),
      inherit.aes = FALSE, hjust = -0.3, vjust = 1.3,
      fontface = "bold", size = 4.2,
      fill = "white", label.size = 0, label.padding = grid::unit(0.12, "lines")
    ) +
    ggplot2::theme(
      strip.text.x = element_blank(),
      legend.position = "none"
    )

  # ---------------------------------------------------------------------------
  # BOTTOM ROWS: separate upper and lower endpoint deviations
  # ---------------------------------------------------------------------------
  endpoint_map <- bind_rows(
    interval_defs |>
      transmute(
        quantile = upper_q,
        interval = interval_label,
        endpoint = "Upper PI endpoint deviation\n(Stopped - Full)"
      ),
    interval_defs |>
      transmute(
        quantile = lower_q,
        interval = interval_label,
        endpoint = "Lower PI endpoint deviation\n(Stopped - Full)"
      )
  )

  diff_rows <- ds |>
    inner_join(endpoint_map, by = "quantile") |>
    transmute(
      method_label,
      horizon,
      endpoint = factor(
        endpoint,
        levels = c(
          "Upper PI endpoint deviation\n(Stopped - Full)",
          "Lower PI endpoint deviation\n(Stopped - Full)"
        )
      ),
      interval = factor(interval, levels = interval_defs$interval_label),
      signed_diff = as.numeric(value_stopped - value_full)
    )

  p_diff <- ggplot(diff_rows) +
    geom_hline(
      yintercept = 0,
      linetype = "dashed",
      linewidth = 0.55,
      colour = "black"
    ) +
    geom_line(
      aes(
        x = horizon,
        y = signed_diff,
        colour = interval,
        group = interval
      ),
      linewidth = 0.82
    ) +
    geom_point(
      aes(
        x = horizon,
        y = signed_diff,
        colour = interval,
        group = interval
      ),
      size = 0.9,
      stroke = 0
    ) +
    facet_grid(
      rows = vars(endpoint),
      cols = vars(method_label),
      scales = "fixed",
      switch = "y"
    ) +
    geom_label(
      data = data.frame(
        endpoint = factor(
          rep(c("Upper PI endpoint deviation\n(Stopped - Full)",
                "Lower PI endpoint deviation\n(Stopped - Full)"), each = 3),
          levels = c("Upper PI endpoint deviation\n(Stopped - Full)",
                     "Lower PI endpoint deviation\n(Stopped - Full)")
        ),
        method_label = factor(rep(unname(method_map), times = 2),
                              levels = unname(method_map)),
        panel_letter = LETTERS[10:15]
      ),
      aes(x = -Inf, y = Inf, label = panel_letter),
      inherit.aes = FALSE, hjust = -0.3, vjust = 1.3,
      fontface = "bold", size = 4.2,
      fill = "white", label.size = 0, label.padding = grid::unit(0.12, "lines")
    ) +
    scale_colour_manual(
      values = INTERVAL_LINE_COLORS,
      breaks = interval_defs$interval_label,
      name = "Prediction interval"
    ) +
    scale_y_continuous(
      sec.axis = sec_axis(
        ~ . * 100000 / US_POP_2025,
        name = "Stopped - Full per 100,000"
      )
    ) +
    scale_x_continuous(
      breaks = c(1, 5, 10, 15, 20, 25, 30, 35, 40, 43),
      limits = c(1, 43),
      minor_breaks = NULL
    ) +
    labs(
      x = "Forecast horizon (week)",
      y = NULL
    ) +
    theme_bw(base_size = 11) +
    theme(
      strip.background = element_blank(),
      strip.text.x = element_blank(),
      strip.text.y.left = element_text(face = "plain", angle = 90, size = 11),
      strip.text.y.right = element_blank(),
      strip.placement = "outside",
      panel.grid.minor = element_blank(),
      legend.position = "bottom",
      panel.spacing.x = grid::unit(0.65, "lines"),
      panel.spacing.y = grid::unit(0.55, "lines"),
      axis.text.y = element_text(size = 12.5, margin = margin(r = 0)),
      axis.text.x = element_text(size = 12.5),
      axis.title.y = element_text(margin = margin(r = 4)),
      axis.ticks.length = grid::unit(0.08, "cm"),
      plot.margin = margin(1, 4, 1, 1)
    )

  # ---------------------------------------------------------------------------
  # Save data used
  # ---------------------------------------------------------------------------
  readr::write_csv(
    ds,
    file.path(out_dir, paste0("DATA_SELECTED_REPLAY_", scenario_short_map[[sc]], ".csv"))
  )

  readr::write_csv(
    diff_rows |>
      mutate(
        method_label = as.character(method_label),
        endpoint = as.character(endpoint),
        interval = as.character(interval)
      ),
    file.path(out_dir, paste0("DATA_ENDPOINT_DIFFS_", scenario_short_map[[sc]], ".csv"))
  )

  readr::write_csv(
    official_ds,
    file.path(out_dir, paste0("DATA_SELECTED_REPLAY_WITH_OFFICIAL_", scenario_short_map[[sc]], ".csv"))
  )

  # ---------------------------------------------------------------------------
  # Combine: 3 hubVis rows + 2 custom rows
  # ---------------------------------------------------------------------------
  # Bottom facet_grid has outer strip/axis structure that makes patchwork
  # reserve extra horizontal space for ALL rows. Free the top rows' label
  # alignment on BOTH left and right so their y-axis labels sit against the
  # panel borders like the bottom rows, while preserving panel alignment.
  p_stopped_free <- p_stopped |>
    patchwork::free(side = "l", type = "label") |>
    patchwork::free(side = "r", type = "label")

  p_full_free <- p_full |>
    patchwork::free(side = "l", type = "label") |>
    patchwork::free(side = "r", type = "label")

  p_official_free <- p_official |>
    patchwork::free(side = "l", type = "label") |>
    patchwork::free(side = "r", type = "label")

  p <- p_stopped_free /
    p_full_free /
    p_official_free /
    p_diff +
    patchwork::plot_layout(
      heights = c(1.05, 1.05, 1.05, 2.10),
      guides = "collect"
    ) &
    theme(legend.position = "bottom")

  out_file <- file.path(
    out_dir,
    paste0("FIG_75PCT_5ROW_HUBVISTOP_OFFICIAL_", scenario_short_map[[sc]], ".png")
  )

  ggsave(
    filename = out_file,
    plot = p,
    width = 16,
    height = 17,
    dpi = 300,
    limitsize = FALSE,
    bg = "white"
  )

  message("Wrote: ", out_file)

  rm(obj, d, ds, stopped_tbl, full_tbl, official_tbl, official_ds, official_sc, diff_rows, p_stopped, p_full, p_official, p_stopped_free, p_full_free, p_official_free, p_diff, p)
  gc(verbose = FALSE)
}

writeLines(
  c(
    paste0("random_seed=", RANDOM_SEED),
    paste0("selected_replay=", SELECTED_REPLAY),
    paste0("operating_point=", OPERATING_POINT),
    paste0("population_for_rates=", US_POP_2025)
  ),
  file.path(out_dir, "selected_replay.txt")
)

message("")
message("DONE.")
message("Selected replay: ", SELECTED_REPLAY)
message("Output directory: ", out_dir)
