#!/usr/bin/env python3
"""Model-only dense quantile-SE fan figure from unified 20,000-bootstrap results.

Layout:
- Rows 1-3: 11 individual models.
  * First 10 non-UT models share a common y-scale.
  * UT-ImmunoSEIRS has its own y-scale and spans columns 3-4 of row 3.
  * Model panels show a dense quantile fan colored by bootstrap SE, with raw
    spaghetti trajectories overlaid for visibility.
- Row 4: 3 ensemble panels.
  * This script renders only the 11 model panels; ensemble panels are excluded.

The left y-axis tick labels show both counts and rate equivalents per 100,000,
with the rate placed in parentheses on the line below each count tick.

Outputs PNG only, with titled and no-title versions.

Fan uses q=0,0.01,...,1.00 from the unified 103-level bootstrap CSV;
q=0.025 and q=0.975 remain in the data but are not plotted as centile bands.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import FuncNorm
from matplotlib.ticker import FuncFormatter, MaxNLocator

PROJECT_ROOT = Path.cwd()
RAW_DIR = PROJECT_ROOT / 'data/processed'
SE_DENSE_DIR = PROJECT_ROOT / 'results_raw_quantile_heterogeneity' / 'full_n_model_bootstrap_unified103'
OUTPUT_DIR = PROJECT_ROOT / 'results_figures_quantile_heterogeneity' / 'full_n_quantile_se_heatmaps'
SCENARIO_SUFFIX = '-2025-07-29'
MODEL_SE_FILE = 'full_n_model_quantile_bootstrap_se_FULL_GRID_long_dense01.csv'
ENSEMBLE_SE_FILE = 'full_n_ensemble_quantile_bootstrap_se_FULL_GRID_long_dense01.csv'
ENSEMBLE_POSITIONS_FILE = 'full_n_ensemble_quantile_fan_positions_FULL_SAMPLE_99Q.csv'
Q_GRID = np.round(np.linspace(0.0, 1.0, 101), 2)  # full centile fan, including sample min and max
POPULATION_US_2025 = 341_784_857
RATE_FACTOR_PER_100K = 100_000.0 / POPULATION_US_2025
SPECIAL_MODEL_OWN_YSCALE = 'UT-ImmunoSEIRS'
ENSEMBLE_ORDER = ['Quantile median', 'LOP untrimmed', 'LOP trimmed']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scenario', default='A', help='Scenario letter (A, B, C) or full scenario ID')
    p.add_argument('--raw-dir', type=Path, default=RAW_DIR,
                   help='Folder containing Global_<scenario>_raw_matrix.npy and metadata.csv')
    p.add_argument('--se-dir', type=Path, default=SE_DENSE_DIR,
                   help='Folder containing the dense 0.01-grid bootstrap SE CSVs')
    p.add_argument('--output-dir', type=Path, default=OUTPUT_DIR,
                   help='Folder for output PNGs')
    p.add_argument('--dpi', type=int, default=400)
    p.add_argument('--max-trajectories', type=int, default=None,
                   help='Optional cap on the number of raw spaghetti trajectories drawn per model panel')
    p.add_argument('--subsample-seed', type=int, default=20260917)
    p.add_argument('--line-alpha', type=float, default=0.18,
                   help='Alpha for raw spaghetti trajectories')
    p.add_argument('--line-width', type=float, default=0.60,
                   help='Line width for raw spaghetti trajectories')
    p.add_argument('--fan-alpha', type=float, default=0.52,
                   help='Alpha for the quantile fan polygons')
    p.add_argument('--asinh-knee-counts', type=float, default=10.0,
                   help='Asinh knee for the bootstrap-SE color scale (count scale)')
    return p.parse_args()


def counts_to_rate_per100k(y):
    return np.asarray(y) * RATE_FACTOR_PER_100K


def resolve_scenario(requested: str, raw_dir: Path) -> str:
    requested = str(requested)
    candidates = [requested] if '-' in requested else [requested + SCENARIO_SUFFIX, requested.upper() + SCENARIO_SUFFIX]
    for s in candidates:
        matrix_path = raw_dir / f'Global_{s}_raw_matrix.npy'
        metadata_path = raw_dir / f'Global_{s}_metadata.csv'
        if matrix_path.exists() and metadata_path.exists():
            return s
    raise FileNotFoundError(
        f'Could not find raw matrix + metadata for scenario {requested!r} in {raw_dir}.\n'
        f'Tried: {candidates}'
    )


def load_raw_scenario(raw_dir: Path, scenario: str) -> tuple[np.ndarray, pd.DataFrame]:
    matrix_path = raw_dir / f'Global_{scenario}_raw_matrix.npy'
    metadata_path = raw_dir / f'Global_{scenario}_metadata.csv'
    if not matrix_path.exists():
        raise FileNotFoundError(f'Missing raw matrix: {matrix_path}')
    if not metadata_path.exists():
        raise FileNotFoundError(f'Missing metadata CSV: {metadata_path}')

    X = np.load(matrix_path)
    meta = pd.read_csv(metadata_path)
    if X.ndim != 2:
        raise ValueError(f'Expected a 2D raw matrix, got shape {X.shape}')
    if len(meta) != X.shape[1] and len(meta) == X.shape[0]:
        X = X.T
    if len(meta) != X.shape[1]:
        raise ValueError('Metadata rows do not match raw matrix columns.')

    meta = meta.copy()
    meta['model'] = meta['model'].astype(str)
    return X, meta


def get_model_trajectories(raw_dir: Path, scenario: str) -> dict[str, np.ndarray]:
    X_h_by_t, meta = load_raw_scenario(raw_dir, scenario)
    models = sorted(meta['model'].dropna().astype(str).unique())
    out = {}
    for model in models:
        idx = np.where(meta['model'].to_numpy().astype(str) == model)[0]
        if idx.size:
            out[model] = X_h_by_t[:, idx].T
    if len(out) != 11:
        raise ValueError(f'Expected 11 individual models, found {len(out)}: {sorted(out)}')
    return out


def load_dense_model_se(se_dir: Path, scenario: str) -> pd.DataFrame:
    path = se_dir / MODEL_SE_FILE
    if not path.exists():
        raise FileNotFoundError(f'Missing dense model SE CSV: {path}')
    df = pd.read_csv(path)
    found = sorted(df['scenario'].astype(str).unique())
    candidates = [s for s in found if s == scenario or s.split('-')[0].upper() == scenario.upper()]
    if len(candidates) != 1:
        raise ValueError(f'Scenario {scenario!r} matches {candidates}; available: {found}')
    scenario_id = candidates[0]
    out = df.loc[df['scenario'].astype(str).eq(scenario_id), ['model', 'horizon', 'quantile', 'bootstrap_se']].copy()
    out['model'] = out['model'].astype(str)
    return out


def load_dense_ensemble_se(se_dir: Path, scenario: str) -> pd.DataFrame:
    path = se_dir / ENSEMBLE_SE_FILE
    if not path.exists():
        raise FileNotFoundError(f'Missing dense ensemble SE CSV: {path}')
    df = pd.read_csv(path)
    found = sorted(df['scenario'].astype(str).unique())
    candidates = [s for s in found if s == scenario or s.split('-')[0].upper() == scenario.upper()]
    if len(candidates) != 1:
        raise ValueError(f'Scenario {scenario!r} matches {candidates}; available: {found}')
    scenario_id = candidates[0]
    out = df.loc[df['scenario'].astype(str).eq(scenario_id), ['entity', 'horizon', 'quantile', 'bootstrap_se']].copy()
    out['entity'] = out['entity'].astype(str)
    return out


def load_ensemble_fan_positions(se_dir: Path, scenario: str) -> dict[str, np.ndarray]:
    path = se_dir / ENSEMBLE_POSITIONS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f'Missing full-sample Hub ensemble quantile positions: {path}\n'
            'Run save_full_sample_ensemble_fan_positions_99Q.R first.'
        )
    df = pd.read_csv(path)
    scenarios = sorted(df['scenario'].astype(str).unique())
    matches = [s for s in scenarios if s == scenario or s.split('-')[0].upper() == scenario.upper()]
    if len(matches) != 1:
        raise ValueError(f'Scenario {scenario!r} not uniquely present in ensemble positions: {scenarios}')
    df = df.loc[df['scenario'].eq(matches[0])].copy()
    # R/CSV floating-point values such as 0.29 and 0.29000000000000004
    # are equivalent quantile levels. Snap to the intended 0.01 grid before
    # pivot/reindex; never interpolate missing quantiles or fill missing values.
    df['quantile'] = pd.to_numeric(df['quantile'], errors='raise')
    df['horizon'] = pd.to_numeric(df['horizon'], errors='raise')
    q_rounded = df['quantile'].round(2)
    if not np.allclose(df['quantile'], q_rounded, rtol=0, atol=1e-8):
        raise ValueError('Some quantile values are not on the expected 0.01 grid')
    df['quantile'] = q_rounded
    df['horizon'] = df['horizon'].astype(int)

    out = {}
    for name in ENSEMBLE_ORDER:
        sub = df.loc[df['entity'].eq(name)]
        if sub.duplicated(['quantile', 'horizon']).any():
            raise ValueError(f'Duplicate quantile/horizon cells in {name}')
        matrix = (sub.pivot(index='quantile', columns='horizon', values='value')
                  .reindex(index=Q_GRID, columns=np.arange(1, 44)))
        if matrix.shape != (99, 43) or matrix.isna().any().any():
            missing_cells = int(matrix.isna().sum().sum())
            missing_qs = matrix.index[matrix.isna().any(axis=1)].tolist()[:12]
            missing_hs = matrix.columns[matrix.isna().any(axis=0)].tolist()[:12]
            raise ValueError(
                f'Incomplete full-sample fan positions for {name}: '
                f'{missing_cells} missing/NA cells, quantiles={missing_qs}, horizons={missing_hs}. '
                f'Source rows={len(sub)}, NA values={int(sub["value"].isna().sum())}'
            )
        out[name] = matrix.to_numpy()
    return out


def maybe_subsample_trajectories(X: np.ndarray, max_trajectories: int | None, seed: int) -> np.ndarray:
    if max_trajectories is None or X.shape[0] <= max_trajectories:
        return X
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(X.shape[0], size=max_trajectories, replace=False))
    return X[idx]


def compute_quantile_curves(X: np.ndarray, probs: np.ndarray = Q_GRID) -> np.ndarray:
    return np.quantile(X, probs, axis=0)


def compute_quantile_boundaries(qcurves: np.ndarray) -> np.ndarray:
    q, h = qcurves.shape
    bounds = np.empty((q + 1, h), dtype=float)
    bounds[1:q, :] = 0.5 * (qcurves[:-1, :] + qcurves[1:, :])
    bounds[0, :] = qcurves[0, :]  # sample minimum; do not extrapolate below observed support
    bounds[q, :] = qcurves[q - 1, :]  # sample maximum; do not extrapolate above observed support
    bounds[0, :] = np.maximum(bounds[0, :], 0.0)
    return bounds


def build_polycollection(qbounds: np.ndarray, se_grid: np.ndarray, norm, cmap='viridis', alpha: float = 0.52):
    """Connect quantile boundaries at the actual observed horizon coordinates.

    Unlike averaging curves at half-week edges, this guarantees that the
    q=0 and q=1 fan envelopes enclose every piecewise-linear trajectory.
    """
    n_cells_q, n_h = se_grid.shape
    if qbounds.shape != (n_cells_q + 1, n_h):
        raise ValueError(f'Boundary shape {qbounds.shape} disagrees with SE shape {se_grid.shape}')
    polygons, colors = [], []
    cmap_obj = plt.get_cmap(cmap)
    for h in range(n_h - 1):
        x0, x1 = float(h + 1), float(h + 2)
        for qi in range(n_cells_q):
            polygons.append([(x0, qbounds[qi, h]),
                             (x1, qbounds[qi, h + 1]),
                             (x1, qbounds[qi + 1, h + 1]),
                             (x0, qbounds[qi + 1, h])])
            colors.append(cmap_obj(norm(0.5 * (se_grid[qi, h] + se_grid[qi, h + 1]))))
    # Extend the first and last horizon's colors to the plot edges.
    for h, x0, x1 in [(0, 0.5, 1.0), (n_h - 1, float(n_h), float(n_h) + 0.5)]:
        for qi in range(n_cells_q):
            polygons.append([(x0, qbounds[qi, h]), (x1, qbounds[qi, h]),
                             (x1, qbounds[qi + 1, h]), (x0, qbounds[qi + 1, h])])
            colors.append(cmap_obj(norm(se_grid[qi, h])))
    return PolyCollection(polygons, facecolors=colors, edgecolors='none', alpha=alpha, closed=True)


def format_count_rate_tick(y, pos):
    rate = counts_to_rate_per100k(y)
    return f'{y:,.0f}\n({rate:.2f})'


def format_count_rate_yaxis(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(format_count_rate_tick))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def get_ylim_from_arrays(arrays: list[np.ndarray]) -> tuple[float, float]:
    ymin = min(float(np.min(a)) for a in arrays)
    ymax = max(float(np.max(a)) for a in arrays)
    lower = min(0.0, ymin)
    upper = ymax + 0.03 * (ymax - lower if ymax > lower else 1.0)
    return lower, upper


def make_common_ticks(lower: float, upper: float, nbins: int = 5) -> np.ndarray:
    locator = MaxNLocator(nbins=nbins)
    ticks = locator.tick_values(lower, upper)
    ticks = ticks[(ticks >= lower - 1e-9) & (ticks <= upper + 1e-9)]
    return ticks


def draw_model_panel(ax, X: np.ndarray, se_model_df: pd.DataFrame, model_name: str, args, norm,
                     show_ylabel: bool = False, own_scale: bool = False, y_ticks=None):
    horizons = np.arange(1, X.shape[1] + 1)
    se_pivot = (se_model_df.pivot(index='quantile', columns='horizon', values='bootstrap_se')
                .reindex(index=Q_GRID, columns=np.arange(1, X.shape[1] + 1)))
    if se_pivot.isna().any().any():
        raise ValueError(f'Incomplete dense SE grid for model {model_name}')
    se_grid = se_pivot.to_numpy()
    qcurves = compute_quantile_curves(X, probs=Q_GRID)
    qbounds = compute_quantile_boundaries(qcurves)
    # The exterior is exactly the observed sample min/max at every week.
    assert np.allclose(qbounds[0], X.min(axis=0))
    assert np.allclose(qbounds[-1], X.max(axis=0))
    ax.add_collection(build_polycollection(qbounds=qbounds, se_grid=se_grid, norm=norm, alpha=args.fan_alpha))

    X_draw = maybe_subsample_trajectories(X, max_trajectories=args.max_trajectories, seed=args.subsample_seed)
    for tr in X_draw:
        ax.plot(horizons, tr, linewidth=args.line_width, alpha=args.line_alpha, color='0.18', zorder=2)

    title = model_name
    ax.set_title(title, pad=5.5)
    ax.grid(alpha=0.16, linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    xtick_vals = [1, 10, 20, 30, 43]
    ax.set_xticks([h for h in xtick_vals if h <= X.shape[1]])
    ax.tick_params(length=2.2, width=0.7, labelbottom=True)
    if show_ylabel:
        ax.set_ylabel('Incident hospitalizations\n(per 100,000)', fontsize=12.0)
        ax.tick_params(axis='y', labelleft=True, left=True, labelsize=7.8)
        format_count_rate_yaxis(ax)
    else:
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelleft=True, left=True, labelsize=7.8)
        format_count_rate_yaxis(ax)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)


def draw_ensemble_panel(ax, qcurves: np.ndarray, se_ens_df: pd.DataFrame, title: str, args, norm,
                        show_ylabel: bool = False, y_ticks=None):
    horizons = np.arange(1, qcurves.shape[1] + 1)
    se_pivot = (se_ens_df.pivot(index='quantile', columns='horizon', values='bootstrap_se')
                .reindex(index=Q_GRID, columns=np.arange(1, qcurves.shape[1] + 1)))
    if se_pivot.isna().any().any():
        raise ValueError(f'Incomplete dense SE grid for ensemble {title}')
    se_grid = se_pivot.to_numpy()
    qbounds = compute_quantile_boundaries(qcurves)
    ax.add_collection(build_polycollection(qbounds=qbounds, se_grid=se_grid, norm=norm, alpha=args.fan_alpha))

    ax.set_title(title, pad=5.5, fontweight='bold')
    ax.grid(alpha=0.16, linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    xtick_vals = [1, 10, 20, 30, 43]
    ax.set_xticks([h for h in xtick_vals if h <= qcurves.shape[1]])
    ax.tick_params(length=2.2, width=0.7, labelbottom=True)
    if show_ylabel:
        ax.set_ylabel('Incident hospitalizations\n(per 100,000)', fontsize=12.0)
        ax.tick_params(axis='y', labelleft=True, left=True, labelsize=7.8)
        format_count_rate_yaxis(ax)
    else:
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelleft=True, left=True, labelsize=7.8)
        format_count_rate_yaxis(ax)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)


def make_figure(model_to_X: dict[str, np.ndarray], dense_model_se: pd.DataFrame,
                scenario: str, args, include_suptitle: bool, output_path: Path):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9.5,
        'axes.titlesize': 10.5,
        'axes.labelsize': 10.0,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.0,
    })

    all_se_max = float(dense_model_se.loc[dense_model_se['quantile'].isin(Q_GRID), 'bootstrap_se'].max())
    norm = FuncNorm(
        (lambda x: np.arcsinh(np.asarray(x) / args.asinh_knee_counts),
         lambda y: args.asinh_knee_counts * np.sinh(np.asarray(y))),
        vmin=0.0,
        vmax=all_se_max,
    )

    # Layout order.
    sorted_models = sorted(model_to_X)
    row1_models = sorted_models[:4]
    row2_models = sorted_models[4:8]
    # Explicit third row order with UVA before UT, and UT widened.
    row3_left = ['UNCC-Hierbin', 'UVA-FluXSim']
    if any(m not in model_to_X for m in row3_left + [SPECIAL_MODEL_OWN_YSCALE]):
        raise ValueError('Expected UNCC-Hierbin, UVA-FluXSim, and UT-ImmunoSEIRS in model set.')

    # Model-only: no Hubverse ensemble files required.
    shared_models = [m for m in model_to_X if m != SPECIAL_MODEL_OWN_YSCALE]
    shared_arrays = [model_to_X[m] for m in shared_models]
    shared_lower, shared_upper = get_ylim_from_arrays(shared_arrays)
    ut_lower, ut_upper = get_ylim_from_arrays([model_to_X[SPECIAL_MODEL_OWN_YSCALE]])
    shared_ticks = make_common_ticks(shared_lower, shared_upper, nbins=5)
    ut_ticks = make_common_ticks(ut_lower, ut_upper, nbins=5)
    print(f"Common non-UT y-range: {shared_lower:,.0f} to {shared_upper:,.0f} hospitalizations")
    print(f"Common shared y-ticks: {shared_ticks.tolist()}")
    print(f"UT-ImmunoSEIRS y-range: {ut_lower:,.0f} to {ut_upper:,.0f} hospitalizations")
    print(f"UT-ImmunoSEIRS y-ticks: {ut_ticks.tolist()}")

    fig = plt.figure(figsize=(16.6, 10.4))
    gs = fig.add_gridspec(3, 4, wspace=0.25, hspace=0.28)

    axes = {}
    # Row 1.
    for j, model in enumerate(row1_models):
        axes[model] = fig.add_subplot(gs[0, j])
    # Row 2.
    for j, model in enumerate(row2_models):
        axes[model] = fig.add_subplot(gs[1, j])
    # Row 3.
    axes['UNCC-Hierbin'] = fig.add_subplot(gs[2, 0])
    axes['UVA-FluXSim'] = fig.add_subplot(gs[2, 1])
    axes[SPECIAL_MODEL_OWN_YSCALE] = fig.add_subplot(gs[2, 2:4])
    # Draw model panels.
    for model in row1_models + row2_models + row3_left + [SPECIAL_MODEL_OWN_YSCALE]:
        ax = axes[model]
        se_df = dense_model_se.loc[dense_model_se['model'].eq(model)].copy()
        show_ylabel = model in {row1_models[0], row2_models[0], 'UNCC-Hierbin'}
        draw_model_panel(ax, model_to_X[model], se_df, model, args=args, norm=norm,
                         show_ylabel=show_ylabel, own_scale=(model == SPECIAL_MODEL_OWN_YSCALE),
                         y_ticks=(ut_ticks if model == SPECIAL_MODEL_OWN_YSCALE else shared_ticks))
        if model == SPECIAL_MODEL_OWN_YSCALE:
            ax.set_ylabel('')
            ax.set_ylim(ut_lower, ut_upper)
        else:
            ax.set_ylim(shared_lower, shared_upper)
        ax.set_xlabel('Forecast horizon (weeks)', fontsize=10.2)

    # Explicitly reapply the common count/rate tick labels to every non-UT facet.
    # Keep UT's independent scale and tick locations.  The position of UT is
    # controlled solely by GridSpec (columns 3-4, no rightward shift).
    for name, ax in axes.items():
        if name == SPECIAL_MODEL_OWN_YSCALE:
            ax.set_ylim(ut_lower, ut_upper)
            ax.set_yticks(ut_ticks)
        else:
            ax.set_ylim(shared_lower, shared_upper)
            ax.set_yticks(shared_ticks)
        ax.yaxis.set_major_formatter(FuncFormatter(format_count_rate_tick))
        ax.tick_params(axis='y', labelleft=True, left=True, labelsize=7.8)

    # Shared colorbar on far right.
    import matplotlib.cm as cm
    mappable = cm.ScalarMappable(norm=norm, cmap='viridis')
    mappable.set_array([])
    cax = fig.add_axes([0.928, 0.20, 0.013, 0.60])
    cb = fig.colorbar(mappable, cax=cax, orientation='vertical')
    cb.set_label('Bootstrap SE of estimated quantiles (incident hospitalizations) · asinh scale', fontsize=10.8, labelpad=8)
    ticks = [v for v in [0, 1, 10, 100, 1000, 10000, 30000] if v <= all_se_max]
    if len(ticks) >= 2:
        cb.set_ticks(ticks)
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:,.0f}' if abs(x) >= 1 else f'{x:,.3g}'))
    cb.ax.tick_params(labelsize=8.8)

    scenario_letter = scenario.split('-')[0]
    if include_suptitle:
        fig.suptitle(
            f'Scenario {scenario_letter} · Submitted trajectories with dense quantile-SE fans (20,000 bootstraps)',
            fontsize=14.5, y=0.985,
        )
        fig.subplots_adjust(left=0.075, right=0.91, bottom=0.06, top=0.95)
    else:
        fig.subplots_adjust(left=0.075, right=0.91, bottom=0.06, top=0.97)

    # UT panel uses its original GridSpec columns 3-4 placement, aligned above.

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(output_path)


def main():
    args = parse_args()
    scenario = resolve_scenario(args.scenario, args.raw_dir)
    model_to_X = get_model_trajectories(args.raw_dir, scenario)
    dense_model_se = load_dense_model_se(args.se_dir, scenario)
    scenario_letter = scenario.split('-')[0]
    base = f'FIG_spaghetti_quantilefan_models_only_20k_Scenario_{scenario_letter}'
    make_figure(model_to_X, dense_model_se, scenario=scenario, args=args,
                include_suptitle=True, output_path=args.output_dir / f'{base}.png')
    make_figure(model_to_X, dense_model_se, scenario=scenario, args=args,
                include_suptitle=False, output_path=args.output_dir / f'{base}_notitle.png')


if __name__ == '__main__':
    main()
