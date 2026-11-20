suppressPackageStartupMessages({
  library(here)
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(scales)
})

# =============================================================================
# Replay-level trajectory savings by model and scenario
# Uses EXISTING stopping results only. No bootstrap/stopping is rerun.
#
# Frozen operating point shown here:
#   monitored quantiles through q = 0.95
#   tau = 7500 hospitalizations
#
# Outputs:
#   FIG_replay_fraction_saved_by_model_boxplot.png
#   FIG_replay_fraction_saved_by_model_violin.png
# =============================================================================

# -----------------------------------------------------------------------------
# Locate existing stopping-results file
# -----------------------------------------------------------------------------

stop_file <- file.path(getwd(), "htc_iid_20k", "analysis", "replay_stop_results_joint_tau_quantiles.csv")
if (!file.exists(stop_file)) stop("Missing stopping results: ", stop_file)

stop_dir <- dirname(stop_file)
out_dir <- file.path(stop_dir, "paper_replay_fraction_saved_by_model")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

message("Reading: ", stop_file)

# -----------------------------------------------------------------------------
# Read and select the frozen high-savings operating point
# -----------------------------------------------------------------------------

dat <- read_csv(stop_file, show_col_types = FALSE) %>%
  filter(
    stopping_spec == "through_0.95",
    tau == 7500
  ) %>%
  mutate(
    scenario_label = case_when(
      scenario %in% c("A", "A-2025-07-29") ~ "Scenario A",
      scenario %in% c("B", "B-2025-07-29") ~ "Scenario B",
      scenario %in% c("C", "C-2025-07-29") ~ "Scenario C",
      TRUE ~ as.character(scenario)
    )
  )

if (nrow(dat) == 0) {
  stop("No rows found for stopping_spec = through_0.95 and tau = 7500.")
}

# Order models by overall median savings across all scenarios/replays.
model_levels <- dat %>%
  group_by(model) %>%
  summarise(
    median_fraction_saved = median(fraction_saved, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(median_fraction_saved) %>%
  pull(model)

dat <- dat %>%
  mutate(
    model = factor(model, levels = model_levels),
    scenario_label = factor(
      scenario_label,
      levels = c("Scenario A", "Scenario B", "Scenario C")
    )
  )

# Save the exact data used by the figures.
write_csv(
  dat,
  file.path(out_dir, "DATA_replay_fraction_saved_by_model.csv")
)

summary_tab <- dat %>%
  group_by(scenario_label, model) %>%
  summarise(
    n_replays = n(),
    criterion_met_rate = mean(criterion_met, na.rm = TRUE),
    median_fraction_saved = median(fraction_saved, na.rm = TRUE),
    q25_fraction_saved = quantile(fraction_saved, 0.25, na.rm = TRUE),
    q75_fraction_saved = quantile(fraction_saved, 0.75, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(
  summary_tab,
  file.path(out_dir, "TABLE_replay_fraction_saved_by_model.csv")
)

# -----------------------------------------------------------------------------
# Shared publication theme
# -----------------------------------------------------------------------------

base_plot <- ggplot(dat, aes(x = fraction_saved, y = model)) +
  facet_grid(. ~ scenario_label) +
  scale_x_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.25),
    labels = percent_format(accuracy = 1),
    expand = expansion(mult = c(0.01, 0.02))
  ) +
  labs(
    x = "Fraction of trajectories saved",
    y = NULL
  ) +
  theme_bw(base_size = 12) +
  theme(
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = 11),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text.y = element_text(size = 8.5),
    axis.title.x = element_text(size = 11),
    panel.spacing.x = grid::unit(0.8, "lines"),
    plot.margin = margin(5.5, 5.5, 5.5, 5.5)
  )

# -----------------------------------------------------------------------------
# Variant 1: boxplot + replay points
# -----------------------------------------------------------------------------

p_box <- base_plot +
  geom_boxplot(
    width = 0.58,
    outlier.shape = NA,
    linewidth = 0.45
  ) +
  geom_jitter(
    height = 0.12,
    width = 0,
    alpha = 0.30,
    size = 0.9
  )

ggsave(
  filename = file.path(
    out_dir,
    "FIG_replay_fraction_saved_by_model_boxplot.png"
  ),
  plot = p_box,
  width = 12.0,
  height = 7.8,
  dpi = 300,
  bg = "white"
)

# -----------------------------------------------------------------------------
# Variant 2: violin + compact boxplot + replay points
# -----------------------------------------------------------------------------

p_violin <- base_plot +
  geom_violin(
    scale = "width",
    trim = TRUE,
    linewidth = 0.4
  ) +
  geom_boxplot(
    width = 0.16,
    outlier.shape = NA,
    linewidth = 0.4
  ) +
  geom_jitter(
    height = 0.08,
    width = 0,
    alpha = 0.20,
    size = 0.7
  )

ggsave(
  filename = file.path(
    out_dir,
    "FIG_replay_fraction_saved_by_model_violin.png"
  ),
  plot = p_violin,
  width = 12.0,
  height = 7.8,
  dpi = 300,
  bg = "white"
)

message("\nSaved:")
message(file.path(out_dir, "FIG_replay_fraction_saved_by_model_boxplot.png"))
message(file.path(out_dir, "FIG_replay_fraction_saved_by_model_violin.png"))
message(file.path(out_dir, "DATA_replay_fraction_saved_by_model.csv"))
message(file.path(out_dir, "TABLE_replay_fraction_saved_by_model.csv"))
