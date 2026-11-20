#!/usr/bin/env python3
"""IID-with-replacement quantile SE convergence for three difficult models and one automatically selected regular comparator.

Samples EXACTLY from each fixed empirical trajectory distribution. We use
multinomial trajectory multiplicities, which have the same distribution as
n IID trajectory-index draws, without materializing 10,000,000 x 43 arrays.
One sampled count vector is shared across all 43 horizons, preserving the
within-trajectory cross-horizon sampling dependence.

The estimand is Monte Carlo SD of sample quantiles over independent IID draws
from the empirical distribution, NOT bootstrap Monte Carlo uncertainty of SE.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Q = np.array([.01, .025] + [round(float(x), 3) for x in np.arange(.05, .951, .05)] + [.975, .99])
PROBLEM_MODELS = ('ACCIDDA-FlepiMoP', 'CEPH-MetaFlu', 'UT-ImmunoSEIRS')
MODELS = PROBLEM_MODELS
DEFAULT_N = (50, 100, 200, 400, 600, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 250000, 500000, 1000000, 2000000, 5000000, 10000000)


def locate_data(root: Path, scenario: str):
    options = [root / 'data/processed', root / 'precomputed_global_distances_full']
    for d in options:
        a = d / f'Global_{scenario}-2025-07-29_raw_matrix.npy'
        b = d / f'Global_{scenario}-2025-07-29_metadata.csv'
        if a.exists() and b.exists():
            return a, b
    raise FileNotFoundError(f'Cannot find Global_{scenario}-2025-07-29_raw_matrix.npy and metadata.csv in data/processed or precomputed_global_distances_full')


def load_model(mat: Path, metadata: Path, model: str):
    a = np.load(mat, mmap_mode='r')
    m = pd.read_csv(metadata)
    if 'model' not in m.columns:
        raise ValueError(f'Missing model column in {metadata}')
    if a.shape[1] == len(m):
        x = a
    elif a.shape[0] == len(m):
        x = a.T
    else:
        raise ValueError(f'No matrix axis matches metadata: {a.shape} vs {len(m)} rows')
    inds = np.flatnonzero(m['model'].astype(str).to_numpy() == model)
    if len(inds) == 0:
        raise ValueError(f'No {model} in {metadata}; examples: {m.model.unique().tolist()}')
    return np.ascontiguousarray(x[:, inds].T, dtype=float)


def empirical_gap_floor(values: np.ndarray, qs: np.ndarray):
    """Theoretical limiting quantile SD: gap/2 only at exact CDF plateau level.

    For independent draws from a finite-support distribution, if F(a)=q and
    F(x)=q for a < x < b, limiting sample quantile is a/b with probability
    1/2 each. Away from such plateau levels, limiting SD is zero.
    """
    vals, counts = np.unique(values, return_counts=True)
    p = np.cumsum(counts) / len(values)
    floor = np.zeros(len(qs))
    gap = np.zeros(len(qs))
    is_boundary = np.zeros(len(qs), dtype=bool)
    for j, q in enumerate(qs):
        candidates = np.where(np.isclose(p[:-1], q, atol=1e-12, rtol=0))[0]
        if len(candidates):
            k = int(candidates[0])
            gap[j] = float(vals[k+1] - vals[k])
            floor[j] = gap[j] / 2
            is_boundary[j] = gap[j] > 0
    return floor, gap, is_boundary


def quantile_replicates_from_counts(x: np.ndarray, counts: np.ndarray, qs: np.ndarray):
    """Match np.quantile(..., method='linear') exactly for expanded IID draws."""
    B, N = counts.shape
    H = x.shape[1]
    n = int(counts[0].sum())
    if not np.all(counts.sum(axis=1) == n):
        raise ValueError('Counts do not all sum to n')
    positions = (n - 1) * qs
    lo = np.floor(positions).astype(np.int64)
    hi = np.ceil(positions).astype(np.int64)
    weight = positions - lo
    out = np.empty((B, H, len(qs)), dtype=float)
    for h in range(H):
        order = np.argsort(x[:, h], kind='stable')
        sorted_values = x[order, h]
        cum = np.cumsum(counts[:, order], axis=1, dtype=np.int64)
        # For 0-based integer rank k, index into sorted support is the number
        # of entries with cumulative multiplicity <= k.
        for b in range(B):
            il = np.searchsorted(cum[b], lo, side='right')
            ih = np.searchsorted(cum[b], hi, side='right')
            out[b, h] = sorted_values[il] * (1.0 - weight) + sorted_values[ih] * weight
    return out


def plot_curves(d: pd.DataFrame, outdir: Path, scenario: str, models):
    fig, axes = plt.subplots(1, len(models), figsize=(5.4 * len(models), 5.8), constrained_layout=True)
    q_colors = plt.get_cmap('viridis')(np.linspace(.04, .96, len(Q)))
    for ax, model in zip(axes, models):
        z = d[(d.model == model) & (d.scenario == scenario)]
        if z.empty:
            ax.set_visible(False)
            continue
        # Select horizon with largest theoretical boundary floor; if none,
        # choose horizon with highest observed large-n quantile SE.
        last_n = z.n.max()
        f = z[z.n == last_n]
        if model not in PROBLEM_MODELS:
            # Show a contrasting horizon with appreciable finite-n uncertainty
            # but the smallest limiting gap (prefer genuinely regular behavior).
            candidate = f.groupby('horizon').agg(max_floor=('theoretical_se_floor','max'),max_se=('se','max')).reset_index()
            hbest = int(candidate.sort_values(['max_floor','max_se'],ascending=[True,False]).iloc[0].horizon)
            target = f[f.horizon == hbest].iloc[0]
        else:
            target = f.sort_values(['theoretical_se_floor', 'se'], ascending=False).iloc[0]
        h = int(target.horizon)
        z = z[z.horizon == h]
        for idx, q in enumerate(Q):
            zz = z[np.isclose(z.q, q)].sort_values('n')
            boundary = zz.is_exact_gap_boundary.iloc[0]
            ax.plot(zz.n, zz.se, color=q_colors[idx], lw=1.7 if boundary else .85,
                    alpha=1 if boundary else .38, label=f'q={q:g}' if boundary else None)
            floor = float(zz.theoretical_se_floor.iloc[0])
            if floor > 0:
                ax.axhline(floor, color=q_colors[idx], ls='--', lw=.8, alpha=.8)
        ax.set_xscale('log')
        ax.set_title(f'{model}\nScenario {scenario}; horizon {h}')
        ax.set_xlabel('IID trajectories drawn from empirical distribution, n')
        ax.set_ylabel('SD of empirical quantile (hospitalizations)')
        ax.grid(alpha=.2)
        if any(z.is_exact_gap_boundary):
            ax.legend(title='Exact plateau q; dashed = (b−a)/2', fontsize=7.5, frameon=False)
        else:
            ax.axhline(0, color='0.3', ls='--', lw=1, label='Predicted limit = 0')
            ax.legend(fontsize=8, frameon=False)

    path = outdir / f'FIG_IID_quantile_SE_convergence_scenario_{scenario}.png'
    fig.savefig(path, dpi=240)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)
    print('Saved', path)


def plot_heatmap(d: pd.DataFrame, outdir: Path, scenario: str, models):
    last_n = int(d.n.max())
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 6), constrained_layout=True)
    from matplotlib.colors import AsinhNorm
    vmin = 0
    vmax = float(d[d.n == last_n].se.max())
    for ax, model in zip(axes, models):
        z = d[(d.model == model) & (d.scenario == scenario) & (d.n == last_n)]
        if z.empty:
            ax.set_visible(False)
            continue
        grid = z.pivot(index='q', columns='horizon', values='se').reindex(index=Q)
        im = ax.imshow(grid.to_numpy(), origin='lower', aspect='auto',
                       extent=[.5, grid.shape[1] + .5, 0, len(Q)],
                       cmap='magma', norm=AsinhNorm(linear_width=max(vmax / 100, .01), vmin=vmin, vmax=vmax))
        ax.set_title(model)
        ax.set_xlabel('Forecast horizon')
        ax.set_yticks(np.arange(len(Q)) + .5)
        ax.set_yticklabels([f'{q:g}' for q in Q], fontsize=7)
        ax.set_ylabel('Monitored quantile q')
        for _, row in z[z.is_exact_gap_boundary].iterrows():
            qi = np.where(np.isclose(Q, row.q))[0][0]
            ax.plot(row.horizon, qi+.5, 'c.', ms=3)
    fig.colorbar(im, ax=list(axes), shrink=.72, label=f'SE at n={last_n:,} (hospitalizations); cyan = exact gap boundary')
    path = outdir / f'FIG_IID_quantile_SE_at_{last_n}_scenario_{scenario}.png'
    fig.savefig(path, dpi=240)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)
    print('Saved', path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--scenario', nargs='+', choices=['A', 'B', 'C'], default=['A'])
    p.add_argument('--replays', type=int, default=400, help='Independent IID samples per n (default: 400)')
    p.add_argument('--seed', type=int, default=20261102)
    p.add_argument('--sizes', default=','.join(map(str, DEFAULT_N)))
    p.add_argument('--output-dir', type=Path, default=Path('iid_empirical_asymptotics_10m'))
    p.add_argument('--comparison-model', default='auto', help='Fourth model, or auto: select smallest worst-case theoretical gap floor across scenarios')
    args = p.parse_args()
    if args.replays < 2:
        p.error('--replays must be at least 2')
    ns = sorted(set(int(s) for s in args.sizes.split(',')))
    if any(n < 2 for n in ns):
        p.error('Sample sizes must be >=2')
    root = args.root.resolve()
    outdir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    if args.comparison_model == 'auto':
        # Minimize worst-case (b-a)/2 over all monitored q, horizons and selected scenarios.
        # Selection uses only the fixed empirical distribution, not experiment outcomes.
        scores = {}
        for scenario in args.scenario:
            mp, md = locate_data(root, scenario)
            meta = pd.read_csv(md)
            for candidate in sorted(set(meta['model'].astype(str)) - set(PROBLEM_MODELS)):
                x_candidate = load_model(mp, md, candidate)
                worst = max(float(np.max(empirical_gap_floor(x_candidate[:, h], Q)[0])) for h in range(x_candidate.shape[1]))
                scores[candidate] = max(scores.get(candidate, 0.), worst)
        if not scores:
            raise RuntimeError('No available comparison model')
        comparison = min(scores, key=lambda m: (scores[m], m))
        print(f'Automatically selected comparator: {comparison}, worst exact-boundary floor={scores[comparison]:.5g}', flush=True)
    else:
        comparison = args.comparison_model
    models = PROBLEM_MODELS + (comparison,)
    records = []
    boundary_records = []
    for scenario in args.scenario:
        mpath, mdpath = locate_data(root, scenario)
        for model in models:
            x = load_model(mpath, mdpath, model)
            N,H=x.shape
            if not np.isfinite(x).all():
                raise ValueError(f'Nonfinite values in {scenario} {model}')
            floors = np.empty((H,len(Q)))
            gaps = np.empty_like(floors)
            boundary = np.zeros((H,len(Q)), dtype=bool)
            for h in range(H):
                floors[h], gaps[h], boundary[h] = empirical_gap_floor(x[:, h], Q)
                for j,q in enumerate(Q):
                    if boundary[h,j]:
                        boundary_records.append(dict(scenario=scenario,model=model,horizon=h+1,q=float(q),
                                                     n_empirical=N,gap_width=gaps[h,j],theoretical_se_floor=floors[h,j]))
            print(f'RUN {scenario} {model}: empirical N={N}, horizons={H}, exact monitored q/h gap-boundaries={boundary.sum()}', flush=True)
            probs = np.full(N, 1.0 / N)
            for n in ns:
                counts = rng.multinomial(n, probs, size=args.replays)
                estimates = quantile_replicates_from_counts(x,counts,Q)
                se = estimates.std(axis=0, ddof=1)
                for h in range(H):
                    for j,q in enumerate(Q):
                        records.append((scenario,model,N,n,args.replays,h+1,float(q),
                                        float(se[h,j]),float(floors[h,j]),float(gaps[h,j]),bool(boundary[h,j])))
                print(f'  n={n:,}, median SE={np.median(se):.3f}, max SE={np.max(se):.3f}',flush=True)
        d = pd.DataFrame.from_records(records,columns=['scenario','model','n_empirical','n','replays','horizon','q',
                                                        'se','theoretical_se_floor','gap_width','is_exact_gap_boundary'])
        plot_curves(d,outdir,scenario,models)
        plot_heatmap(d,outdir,scenario,models)
    d = pd.DataFrame.from_records(records,columns=['scenario','model','n_empirical','n','replays','horizon','q',
                                                   'se','theoretical_se_floor','gap_width','is_exact_gap_boundary'])
    d.to_csv(outdir/'TABLE_IID_quantile_SE_all_sizes_horizons.csv',index=False)
    pd.DataFrame(boundary_records,columns=['scenario','model','horizon','q','n_empirical','gap_width','theoretical_se_floor']).to_csv(
        outdir/'TABLE_exact_empirical_CDF_gap_boundaries.csv',index=False)
    final = d[d.n == max(ns)].copy()
    final['absolute_error_from_limit']=(final.se-final.theoretical_se_floor).abs()
    final.to_csv(outdir/'TABLE_IID_quantile_SE_at_max_n.csv',index=False)
    summary = d[d.n == max(ns)].groupby(['scenario','model'],as_index=False).agg(
        max_se=('se','max'), median_se=('se','median'), max_theoretical_floor=('theoretical_se_floor','max'),
        exact_plateau_cells=('is_exact_gap_boundary','sum'))
    summary.to_csv(outdir/'TABLE_model_convergence_summary.csv', index=False)
    (outdir/'COMPARATOR_SELECTION.txt').write_text(
        f'Comparator: {comparison}\nSelection: minimum worst-case exact-plateau SE floor across scenarios/quantiles/horizons.\n'
        f'Replays: {args.replays} IID draws per n. These are independent Monte Carlo replications, NOT inner bootstrap replications.\n')
    print('DONE. Files in:',outdir)
    print('Note: a finite empirical distribution has a nonzero quantile-SD limit ONLY when q equals the cumulative mass at a positive gap.')

if __name__ == '__main__':
    main()
