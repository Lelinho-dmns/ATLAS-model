#!/bin/bash
#SBATCH --job-name=atlas_inference
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/homes/eprato01/logs/atlas_inference/atlas_inference_%j.out
#SBATCH --error=/homes/eprato01/logs/atlas_inference/atlas_inference_%j.err
#SBATCH --account=tesi_eprato01
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=284215@studenti.unimore.it

# Usage:  sbatch inference/run_inference.sh            (from /homes/eprato01/ATLAS-model)
# Video, checkpoint and options can be changed in the configuration below.

set -euo pipefail

# The pip torch wheels ship their own CUDA runtime, so no cuda module is needed.
# The ffmpeg module is optional: the venv's imageio-ffmpeg binary is used as a fallback.
module load ffmpeg/8.1 2>/dev/null || echo "ffmpeg module not available, using imageio-ffmpeg"
source /homes/eprato01/ATLAS-model/.venv/bin/activate

# ============================================================
# Configuration
# ============================================================
REPO_DIR=/homes/eprato01/ATLAS-model
VIDEO=${REPO_DIR}/materials/videos/aries_crop_2025-08-25_073541_VID001.mp4

MODEL_VARIANT=vits                                   # vits | vitb | vitl (must match CKPT)
CKPT=/work/tesi_eprato01/checkpoints/ATLAS/vit-s/atlas_vits.ckpt
CONFIG=${REPO_DIR}/configs/ATLAS_DINOv3_${MODEL_VARIANT}.yaml

VIDEO_NAME=$(basename "${VIDEO%.*}")
OUTPUT_ROOT=/work/tesi_eprato01/ATLAS_inference/${VIDEO_NAME}
FRAMES_DIR=${OUTPUT_ROOT}/frames
PRED_DIR=${OUTPUT_ROOT}/predictions_${MODEL_VARIANT}

# ---- Preprocessing ----
FPS=0            # 0 = native frame rate (30 fps -> 5608 frames); e.g. 5 for a faster run
CROP=none        # none | auto (auto removes the black border around the endoscope view)

# ---- Inference ----
CLIP_LENGTH=1    # 1 = per-frame (as the repo's validation); 5 = temporal query propagation (as training)
BATCH_SIZE=16    # frames per forward pass, only used when CLIP_LENGTH=1
NUM_WORKERS=6

# ---- Report / Weights & Biases (login is read from ~/.netrc) ----
NUM_REP_FRAMES=12          # representative frames with the organs coloured
WANDB_PROJECT=ATLAS-inference
WANDB_RUN_NAME=${VIDEO_NAME}_${MODEL_VARIANT}_clip${CLIP_LENGTH}
export WANDB_DIR=${PRED_DIR}

# ============================================================
# Run
# ============================================================
cd "${REPO_DIR}"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) - GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Video: ${VIDEO}"
echo "Checkpoint: ${CKPT}"

if [ -f "${FRAMES_DIR}/meta.json" ]; then
    echo "Frames already extracted in ${FRAMES_DIR}, skipping preprocessing (delete the folder to redo it)."
else
    echo "=========== Preprocessing ==========="
    python inference/preprocess_video.py \
        --video "${VIDEO}" \
        --out_dir "${FRAMES_DIR}" \
        --fps ${FPS} \
        --crop ${CROP} \
        --overwrite
fi

echo "=========== Inference ==========="
python inference/infer_video.py \
    --frames_dir "${FRAMES_DIR}" \
    --out_dir "${PRED_DIR}" \
    --ckpt "${CKPT}" \
    --config "${CONFIG}" \
    --clip_length ${CLIP_LENGTH} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS}

echo "=========== Report + W&B ==========="
python inference/report.py \
    --frames_dir "${FRAMES_DIR}" \
    --pred_dir "${PRED_DIR}" \
    --num_frames ${NUM_REP_FRAMES} \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_name ${WANDB_RUN_NAME}

echo "Results in ${PRED_DIR} (representative frames in ${PRED_DIR}/representative_frames)"
