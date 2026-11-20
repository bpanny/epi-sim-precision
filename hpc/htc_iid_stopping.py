#!/usr/bin/env python3
"""HTC IID-with-replacement replay chunks; compatible with scripts/stopping_sweep.py postprocessing."""
import argparse
import glob
from typing import List, Sequence
import csv
import gzip
import hashlib
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd

FULL_QUANTILES = [
    0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
    0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
    0.975, 0.99,
]

def discover_scenarios(input_dir: Path) -> List[str]:
    files = sorted(glob.glob(str(input_dir / "Global_*_raw_matrix.npy")))
    return [
        Path(f).name.replace("Global_", "").replace("_raw_matrix.npy", "")
        for f in files
    ]

def ordered_scenarios(found: Sequence[str]) -> List[str]:
    preferred = ["A-2025-07-29", "B-2025-07-29", "C-2025-07-29"]
    found = list(map(str, found))
    return [s for s in preferred if s in found] + sorted(
        [s for s in found if s not in preferred]
    )

def scenario_short_name(x: str) -> str:
    return str(x).split("-")[0]

def load_scenario(input_dir: Path, scenario: str):
    raw_path = input_dir / f"Global_{scenario}_raw_matrix.npy"
    meta_path = input_dir / f"Global_{scenario}_metadata.csv"

    X = np.load(raw_path)
    meta = pd.read_csv(meta_path)

    if X.ndim != 2:
        raise ValueError(f"{scenario}: expected 2D matrix, got {X.shape}")
    if "model" not in meta.columns:
        raise ValueError(f"{meta_path} missing model column")

    if X.shape[1] != len(meta) and X.shape[0] == len(meta):
        X = X.T

    if X.shape[1] != len(meta):
        raise ValueError(
            f"{scenario}: matrix shape {X.shape} incompatible with {len(meta)} metadata rows"
        )

    meta = meta.copy()
    meta["model"] = meta["model"].astype(str)
    return X.astype(float, copy=False), meta

def checkpoint_grid(N: int, min_n: int, step: int) -> List[int]:
    if N <= min_n:
        return [N]
    vals = list(range(min_n, N + 1, step))
    if vals[-1] != N:
        vals.append(N)
    return sorted(set(vals))

def quantile_matrix(X: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    # H x Q
    return np.quantile(X, quantiles, axis=1).T

def bootstrap_se_current_prefix(
    X_prefix: np.ndarray,
    quantiles: Sequence[float],
    B: int,
    rng: np.random.Generator,
) -> np.ndarray:
    H, n = X_prefix.shape
    Q = len(quantiles)
    draws = np.empty((B, H, Q), dtype=np.float64)

    for b in range(B):
        idx = rng.integers(0, n, size=n)
        draws[b] = quantile_matrix(X_prefix[:, idx], quantiles)

    return draws.std(axis=0, ddof=1)

def _stable_task_seed(base_seed: int, scenario_index: int, model_index: int) -> int:
    """
    Deterministic task seed independent of process scheduling.

    SeedSequence is used rather than Python's hash(), whose result can vary
    across interpreter sessions.
    """
    ss = np.random.SeedSequence([int(base_seed), int(scenario_index), int(model_index)])
    return int(ss.generate_state(1, dtype=np.uint32)[0])

def task_catalog(inp):
    tasks = []
    for si, scenario in enumerate(ordered_scenarios(discover_scenarios(inp))):
        _, meta = load_scenario(inp, scenario)
        for mi, model in enumerate(sorted(meta['model'].astype(str).unique())):
            tasks.append((scenario, model, _stable_task_seed(20260831, si, mi)))
    if len(tasks) != 33:
        raise RuntimeError(f'Expected 33 model-scenario tasks, found {len(tasks)}')
    return tasks

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir', default='data/processed')
    p.add_argument('--output-dir', default='htc_iid_20k/chunks')
    p.add_argument('--task-id', type=int, required=True)
    p.add_argument('--task-count', type=int, default=100)
    p.add_argument('--replays', type=int, default=100)
    p.add_argument('--bootstrap-reps', type=int, default=20000)
    p.add_argument('--min-n', type=int, default=50)
    p.add_argument('--step', type=int, default=50)
    p.add_argument('--force', action='store_true')
    args=p.parse_args()
    if not 0 <= args.task_id < args.task_count: p.error('invalid task-id')
    if args.bootstrap_reps < 2: p.error('bootstrap-reps must be >= 2')
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    dest=out/f'chunk_{args.task_id:03d}.csv'
    index_dest=out/f'chunk_{args.task_id:03d}_indices.csv.gz'
    if dest.exists() and index_dest.exists() and not args.force:
        print('Already completed:',dest,'and',index_dest,flush=True); return
    if (dest.exists() or index_dest.exists()) and not args.force:
        raise RuntimeError(f'Partial prior output: {dest} / {index_dest}; inspect and use --force to replace')
    tasks=task_catalog(Path(args.input_dir))
    total=len(tasks)*args.replays
    # Map task ID to replay IDs; exactly disjoint partitions independent of task-count.
    assignments=[]
    for ordinal in range(args.task_id,total,args.task_count):
        model_task_index,replay0=divmod(ordinal,args.replays)
        assignments.append((model_task_index,replay0))
    if not assignments: raise RuntimeError('No assignments')
    cache={}
    records=[]
    index_records=[]
    begin=time.monotonic()
    for ai,(mti,replay0) in enumerate(assignments,1):
        scenario,model,task_seed=tasks[mti]
        if scenario not in cache:
            cache.clear() # conserve memory
            cache[scenario]=load_scenario(Path(args.input_dir),scenario)
        X,meta=cache[scenario]
        indices=np.flatnonzero(meta['model'].to_numpy()==model)
        Xm=X[:,indices]; N=len(indices)
        # Match the old model-task per-replay seed stream, but replace the outer permutation
        # with independent uniform empirical draws.
        task_rng=np.random.default_rng(task_seed)
        replay_seeds=task_rng.integers(0,np.iinfo(np.int32).max,size=args.replays)
        replay_seed=int(replay_seeds[replay0])
        rng=np.random.default_rng(replay_seed)
        order=rng.integers(0,N,size=N)
        # Persist the actual sequence used for all prefixes; 1-based local
        # trajectory indices, including duplicates, suitable for R indexing.
        # SHA256 pins the exact sequence for later integrity checks.
        order_1based=order.astype(np.int64)+1
        digest=hashlib.sha256(order_1based.astype('<i8').tobytes()).hexdigest()
        for position, local_idx in enumerate(order_1based,1):
            index_records.append(dict(scenario=scenario,model=model,replay=replay0+1,
                replay_seed=replay_seed,n_available=N,position=position,
                local_index_1based=int(local_idx),sequence_sha256=digest))
        for n in checkpoint_grid(N,args.min_n,args.step):
            prefix=Xm[:,order[:n]]
            boot_seed=int(rng.integers(0,np.iinfo(np.int32).max))
            se=bootstrap_se_current_prefix(prefix,FULL_QUANTILES,args.bootstrap_reps,np.random.default_rng(boot_seed))
            for qi,q in enumerate(FULL_QUANTILES):
                vals=se[:,qi]
                if not np.isfinite(vals).any():
                    raise RuntimeError(f'Nonfinite SE: {scenario} {model} replay={replay0+1} n={n} q={q}')
                h=int(np.nanargmax(vals))
                records.append(dict(scenario=scenario,scenario_short=scenario_short_name(scenario),
                    model=model,replay=replay0+1,replay_seed=replay_seed,n=n,n_available=N,
                    quantile=float(q),max_horizon_se=float(vals[h]),max_se_horizon=h+1,
                    bootstrap_reps=args.bootstrap_reps))
        print(f'[{ai}/{len(assignments)}] {scenario} {model} replay={replay0+1} N={N} '
              f'elapsed={(time.monotonic()-begin)/60:.1f} min',flush=True)
    df=pd.DataFrame(records).sort_values(['scenario','model','replay','n','quantile'])
    # Write index file first; never mark a results chunk complete without it.
    index_tmp=Path(str(index_dest)+'.tmp')
    idx_df=pd.DataFrame(index_records)
    with gzip.open(index_tmp,'wt',newline='',compresslevel=6) as f:
        idx_df.to_csv(f,index=False)
    os.replace(index_tmp,index_dest)
    tmp=dest.with_suffix('.tmp'); df.to_csv(tmp,index=False); os.replace(tmp,dest)
    print(f'DONE {dest} rows={len(df)}; {index_dest} index_rows={len(idx_df)} '
          f'elapsed={(time.monotonic()-begin)/60:.1f} min',flush=True)
if __name__=='__main__': main()
