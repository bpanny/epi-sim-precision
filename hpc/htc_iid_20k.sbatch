#!/bin/bash
#SBATCH --job-name=epi-iid-20k
#SBATCH --clusters=htc
#SBATCH --partition=htc
#SBATCH --account=arahimian
#SBATCH --qos=htc-htc-n
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --array=0-99%64
#SBATCH --output=logs/iid_%A_%a.out
#SBATCH --error=logs/iid_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python3 -u hpc/htc_iid_stopping.py --task-id "$SLURM_ARRAY_TASK_ID" --task-count 100 --replays 100 --bootstrap-reps 20000
