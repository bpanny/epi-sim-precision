#!/usr/bin/env python3
"""Full-N, model-only 20,000-replicate bootstrap of 103 quantiles x 43 weeks.

Run from epi-sim-precision repository root. No R/Hubverse/ensemble dependency.
Each model-scenario output is atomic and resumable. Quantiles match NumPy
method='linear' / R's default type-7 sample quantile.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

PAPER_Q = np.array([.01, .025, .05, .10, .15, .20, .25, .30, .35,
                    .40, .45, .50, .55, .60, .65, .70, .75, .80,
                    .85, .90, .95, .975, .99])
Q = np.unique(np.round(np.r_[np.linspace(0, 1, 101), PAPER_Q], 6))
SCENARIOS = ['A-2025-07-29', 'B-2025-07-29', 'C-2025-07-29']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, default=Path('data/processed'))
    p.add_argument('--output-dir', type=Path, default=Path('results_raw_quantile_heterogeneity/full_n_model_bootstrap_unified103'))
    p.add_argument('--B', type=int, default=20000)
    p.add_argument('--seed', type=int, default=20261028)
    p.add_argument('--scenarios', nargs='+', default=SCENARIOS)
    p.add_argument('--models', nargs='+', help='Optional specific model name(s)')
    p.add_argument('--progress-every', type=int, default=1000)
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    if a.B < 2:
        p.error('--B must be at least 2')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    by_cell = a.output_dir / 'by_model_scenario'
    by_cell.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    print(f'B={a.B}, quantiles={len(Q)}, horizons=43, no ensembles', flush=True)
    print('Quantile grid:', Q.tolist(), flush=True)
    for si, scenario in enumerate(a.scenarios):
        raw = a.input_dir / f'Global_{scenario}_raw_matrix.npy'
        meta_path = a.input_dir / f'Global_{scenario}_metadata.csv'
        if not raw.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f'Missing {raw} or {meta_path}')
        X = np.load(raw, mmap_mode='r')
        meta = pd.read_csv(meta_path)
        if 'model' not in meta.columns:
            raise ValueError(f'Missing model column in {meta_path}')
        if X.shape[0] != 43 or X.shape[1] != len(meta):
            raise ValueError(f'Bad shape {scenario}: matrix {X.shape}, metadata {len(meta)}')
        labels = meta['model'].astype(str).to_numpy()
        models = sorted(set(labels))
        if a.models:
            missing = set(a.models) - set(models)
            if missing:
                raise ValueError(f'Missing model(s) {missing} in {scenario}')
            models = [m for m in models if m in a.models]
        for mi, model in enumerate(models):
            safe_model = ''.join(ch if ch.isalnum() or ch in '_-' else '_' for ch in model)
            target = by_cell / f'{scenario}__{safe_model}.csv'
            metadata_file = target.with_suffix('.json')
            if target.exists() and not a.overwrite:
                if not metadata_file.exists():
                    raise RuntimeError(f'Incomplete output, missing {metadata_file}; use --overwrite')
                details = json.loads(metadata_file.read_text())
                if details.get('B') != a.B or details.get('seed') != a.seed or details.get('quantiles') != Q.tolist():
                    raise RuntimeError(f'Existing output has incompatible configuration: {target}; use --overwrite')
                print(f'SKIP {scenario} {model}: existing validated configuration', flush=True)
                continue
            cols = np.flatnonzero(labels == model)
            M = np.asarray(X[:, cols], dtype=np.float64)
            if not np.isfinite(M).all():
                raise ValueError(f'Nonfinite observations in {scenario} {model}')
            n = M.shape[1]
            # Stable independent streams per scenario / model: unaffected by skip/resume.
            # sorted order used by model index for reproducible selection.
            rng = np.random.default_rng(np.random.SeedSequence([a.seed, SCENARIOS.index(scenario), sorted(set(labels)).index(model)]))
            mean = np.zeros((43, len(Q)), dtype=np.float64)
            M2 = np.zeros_like(mean)
            t0 = time.perf_counter()
            print(f'START {scenario} {model}, n={n}, B={a.B}', flush=True)
            for b in range(1, a.B + 1):
                draw = rng.integers(n, size=n)
                quant = np.quantile(M[:, draw], Q, axis=1, method='linear').T
                delta = quant - mean
                mean += delta / b
                M2 += delta * (quant - mean)
                if a.progress_every and (b % a.progress_every == 0 or b == a.B):
                    elapsed = time.perf_counter() - t0
                    print(f'  {b}/{a.B} ({b/a.B:.0%}) | elapsed {elapsed/60:.1f} min | ETA {elapsed/b*(a.B-b)/60:.1f} min', flush=True)
            se = np.sqrt(M2 / (a.B - 1))
            full_quantiles = np.quantile(M, Q, axis=1, method='linear').T
            out = pd.DataFrame({
                'scenario': scenario, 'model': model, 'n_available': n,
                'horizon': np.repeat(np.arange(1, 44), len(Q)),
                'quantile': np.tile(Q, 43),
                'bootstrap_se': se.ravel(), 'bootstrap_reps': a.B,
                'bootstrap_mean_quantile': mean.ravel(),
                'full_sample_quantile': full_quantiles.ravel(),
            })
            tmp = target.with_suffix('.tmp')
            out.to_csv(tmp, index=False)
            os.replace(tmp, target)
            details = {'scenario': scenario, 'model': model, 'B': a.B, 'seed': a.seed,
                       'n_available': n, 'quantiles': Q.tolist(),
                       'elapsed_seconds': time.perf_counter()-t0,
                       'numpy': np.__version__, 'quantile_method': 'linear'}
            metadata_file.write_text(json.dumps(details, indent=2))
            print(f'DONE {target} ({details["elapsed_seconds"]:.1f}s)', flush=True)
    files = sorted(by_cell.glob('*.csv'))
    all_outputs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # Fail on mixed B/seed outputs, rather than silently combining obsolete runs.
    meta_files = [f.with_suffix('.json') for f in files]
    info = [json.loads(f.read_text()) for f in meta_files]
    if any(item['B'] != a.B or item['seed'] != a.seed or item['quantiles'] != Q.tolist() for item in info):
        raise ValueError('Existing by-cell outputs have mixed bootstrap settings: remove old outputs or use --overwrite')
    combined = a.output_dir / 'full_n_model_quantile_bootstrap_se_UNIFIED_103Q.csv'
    all_outputs.to_csv(combined, index=False)
    # Compatibility with existing Figure 1 loader, which looks for this exact name.
    compat = a.output_dir / 'full_n_model_quantile_bootstrap_se_FULL_GRID_long_dense01.csv'
    all_outputs.to_csv(compat, index=False)
    print(f'Wrote {len(all_outputs):,} rows from {len(files)} model-scenario cells, elapsed {(time.perf_counter()-start)/60:.1f} min')
    print(f'Output: {combined}\nFigure 1 compatibility: {compat}')

if __name__ == '__main__':
    main()
