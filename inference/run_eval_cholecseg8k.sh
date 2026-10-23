#!/bin/bash
#SBATCH --job-name=atlas_cholecseg8k
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/homes/eprato01/logs/atlas_inference/atlas_cholecseg8k_%j.out
#SBATCH --error=/homes/eprato01/logs/atlas_inference/atlas_cholecseg8k_%j.err
#SBATCH --account=tesi_eprato01
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=284215@studenti.unimore.it

# Usage:  sbatch inference/run_eval_cholecseg8k.sh      (from /homes/eprato01/ATLAS-model)
# Quantitative evaluation of ATLAS on CholecSeg8k, logged to its own W&B project.

set -euo pipefail

# The pip torch wheels ship their own CUDA runtime, so no cuda module is needed.
source /homes/eprato01/ATLAS-model/.venv/bin/activate

# ============================================================
# Configuration
# ============================================================
REPO_DIR=/homes/eprato01/ATLAS-model
DATA_ROOT=/work/tesi_eprato01/datasets/cholecseg8k

MODEL_VARIANT=vits                                   # vits | vitb | vitl (must match CKPT)
CKPT=/work/tesi_eprato01/checkpoints/ATLAS/vit-s/atlas_vits.ckpt
CONFIG=${REPO_DIR}/configs/ATLAS_DINOv3_${MODEL_VARIANT}.yaml

CLIP_LENGTH=1    # 1 = per-frame (as the repo's validation); 5 = temporal query propagation (as training)
BATCH_SIZE=16
NUM_WORKERS=6

OUT_DIR=/work/tesi_eprato01/ATLAS_inference/cholecseg8k/eval_${MODEL_VARIANT}_clip${CLIP_LENGTH}

# ---- Weights & Biases (login is read from ~/.netrc) ----
WANDB_PROJECT=ATLAS-CholecSeg8k
WANDB_RUN_NAME=cholecseg8k_${MODEL_VARIANT}_clip${CLIP_LENGTH}
mkdir -p "${OUT_DIR}"
export WANDB_DIR=${OUT_DIR}

# ============================================================
# Run
# ============================================================
cd "${REPO_DIR}"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) - GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Checkpoint: ${CKPT}"

python inference/eval_cholecseg8k.py \
    --data_root "${DATA_ROOT}" \
    --out_dir "${OUT_DIR}" \
    --ckpt "${CKPT}" \
    --config "${CONFIG}" \
    --clip_length ${CLIP_LENGTH} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_name ${WANDB_RUN_NAME}

echo "Results in ${OUT_DIR}"
