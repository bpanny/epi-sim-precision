#!/usr/bin/env python3
"""Memory-safe merge of completed HTC IID stopping chunks; stdlib only.

Checks source record coverage, bootstrap count, replay seeds, positions,
trajectory bounds and SHA256. Never reconstructs indices or recomputes SEs.
"""
import argparse
import csv
import gzip
import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path

RESULT_KEYS = ('scenario', 'model', 'replay')
INDEX_FIELDS = ('scenario', 'model', 'replay', 'replay_seed', 'n_available',
                'position', 'local_index_1based', 'sequence_sha256')

def open_csv(path):
    return gzip.open(path, 'rt', newline='') if str(path).endswith('.gz') else open(path, 'r', newline='')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunks', default='htc_iid_20k/chunks')
    ap.add_argument('--output-dir', default='htc_iid_20k/analysis')
    ap.add_argument('--task-count', type=int, default=100)
    ap.add_argument('--replays', type=int, default=100)
    ap.add_argument('--bootstrap-reps', type=int, default=20000)
    args = ap.parse_args()
    chunk = Path(args.chunks)
    result_files = [chunk / f'chunk_{i:03d}.csv' for i in range(args.task_count)]
    index_files = [chunk / f'chunk_{i:03d}_indices.csv.gz' for i in range(args.task_count)]
    missing = [str(f) for f in result_files + index_files if not f.is_file()]
    if missing:
        raise SystemExit(f'MISSING {len(missing)} files, first: {missing[:8]}')

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    out_results = output / 'replay_checkpoint_quantile_max.csv'
    out_indices = output / 'replay_trajectory_indices.csv.gz'
    tmp_results = Path(str(out_results) + '.tmp')
    tmp_indices = Path(str(out_indices) + '.tmp')

    # Small per-replay bookkeeping; no full DataFrame materialization.
    info = {}
    result_counts = defaultdict(lambda: defaultdict(int))
    q_sets = defaultdict(set)
    result_total = 0
    index_total = 0

    try:
        with open(tmp_results, 'w', newline='') as fout:
            writer = None
            expected_header = None
            for i, file in enumerate(result_files):
                with open(file, 'r', newline='') as fin:
                    reader = csv.DictReader(fin)
                    if reader.fieldnames is None:
                        raise ValueError(f'Empty result CSV: {file}')
                    if expected_header is None:
                        expected_header = reader.fieldnames
                        required = set(RESULT_KEYS) | {'n','n_available','replay_seed','quantile','bootstrap_reps'}
                        assert required <= set(expected_header), (file, required - set(expected_header))
                        writer = csv.DictWriter(fout, fieldnames=expected_header)
                        writer.writeheader()
                    elif reader.fieldnames != expected_header:
                        raise ValueError(f'Inconsistent header in {file}')
                    for row in reader:
                        key = tuple(row[k] for k in RESULT_KEYS)
                        n = int(row['n'])
                        N = int(row['n_available'])
                        seed = int(row['replay_seed'])
                        if int(row['bootstrap_reps']) != args.bootstrap_reps:
                            raise ValueError(f'Wrong B: {file}, {key}')
                        if not 1 <= n <= N:
                            raise ValueError(f'Invalid n={n} N={N}: {file}, {key}')
                        if key not in info:
                            info[key] = (N, seed)
                        elif info[key] != (N, seed):
                            raise ValueError(f'Inconsistent replay metadata: {key}')
                        kn = (key, n)
                        q = row['quantile']
                        if q in q_sets[kn]:
                            raise ValueError(f'Duplicate quantile {q}: {key}, n={n}')
                        q_sets[kn].add(q)
                        result_counts[key][n] += 1
                        writer.writerow(row)
                        result_total += 1
                if (i+1) % 10 == 0:
                    print(f'Checked results {i+1}/{args.task_count}, rows={result_total:,}', flush=True)

        if len(info) != 33 * args.replays:
            raise ValueError(f'Expected {33*args.replays} replays; got {len(info)}')
        for key, (N, seed) in info.items():
            checkpoints = list(range(50, N+1, 50))
            if not checkpoints or checkpoints[-1] != N:
                checkpoints.append(N)
            got = result_counts[key]
            if set(got) != set(checkpoints) or any(count != 23 for count in got.values()):
                raise ValueError(f'Invalid checkpoint/quantile coverage: {key}, {dict(got)}')
        del q_sets, result_counts

        # Index files are written replay-by-replay by the worker. Validate each
        # complete sequence, hash all 1-based indices as little-endian int64.
        seen_keys = set()
        cur_key = None
        cur_N = 0
        cur_seed = 0
        cur_hash = None
        cur_sha = None
        cur_pos = 0
        index_header = None

        def finalize():
            if cur_key is None:
                return
            if cur_pos != cur_N:
                raise ValueError(f'Index sequence incomplete for {cur_key}: {cur_pos}/{cur_N}')
            if cur_hash.hexdigest() != cur_sha:
                raise ValueError(f'SHA256 mismatch for {cur_key}')
            seen_keys.add(cur_key)

        with gzip.open(tmp_indices, 'wt', newline='', compresslevel=6) as fout:
            writer = None
            for i, file in enumerate(index_files):
                with gzip.open(file, 'rt', newline='') as fin:
                    reader = csv.DictReader(fin)
                    if reader.fieldnames is None:
                        raise ValueError(f'Empty index CSV: {file}')
                    if index_header is None:
                        index_header = reader.fieldnames
                        assert set(INDEX_FIELDS) <= set(index_header)
                        writer = csv.DictWriter(fout, fieldnames=index_header)
                        writer.writeheader()
                    elif reader.fieldnames != index_header:
                        raise ValueError(f'Inconsistent index header: {file}')
                    for row in reader:
                        key = tuple(row[k] for k in RESULT_KEYS)
                        if key != cur_key:
                            finalize()
                            if key in seen_keys:
                                raise ValueError(f'Non-contiguous or duplicate replay index sequence {key}')
                            if key not in info:
                                raise ValueError(f'Index has no matching result: {key}')
                            cur_key = key
                            cur_N, cur_seed = info[key]
                            cur_hash = hashlib.sha256()
                            cur_sha = row['sequence_sha256']
                            cur_pos = 0
                        if int(row['replay_seed']) != cur_seed or int(row['n_available']) != cur_N:
                            raise ValueError(f'Index metadata mismatch: {key}')
                        if row['sequence_sha256'] != cur_sha:
                            raise ValueError(f'Index SHA field inconsistent: {key}')
                        cur_pos += 1
                        if int(row['position']) != cur_pos:
                            raise ValueError(f'Position discontinuity: {key} position={row["position"]} expected={cur_pos}')
                        ix = int(row['local_index_1based'])
                        if not 1 <= ix <= cur_N:
                            raise ValueError(f'Index out of range: {key}, {ix}')
                        cur_hash.update(ix.to_bytes(8, 'little', signed=True))
                        writer.writerow(row)
                        index_total += 1
                if (i+1) % 10 == 0:
                    print(f'Checked indices {i+1}/{args.task_count}, rows={index_total:,}', flush=True)
            finalize()

        if seen_keys != set(info):
            raise ValueError(f'Missing indices for {len(set(info)-seen_keys)} replays')
        os.replace(tmp_results, out_results)
        os.replace(tmp_indices, out_indices)
        print(f'PASS: {len(info)} replays, {result_total:,} quantile/checkpoint rows -> {out_results}')
        print(f'PASS: {index_total:,} verified trajectory index rows -> {out_indices}')
        print('NEXT: python3 scripts/stopping_sweep.py --output-dir', output)
    except BaseException:
        for f in (tmp_results, tmp_indices):
            try: f.unlink()
            except FileNotFoundError: pass
        raise

if __name__ == '__main__':
    main()
