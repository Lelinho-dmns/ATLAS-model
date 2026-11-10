#!/bin/bash
# Submit the video inference and the CholecSeg8k evaluation for each backbone.
# Usage (from anywhere):  bash inference/submit_all_backbones.sh            -> vits vitb vitl
#                         bash inference/submit_all_backbones.sh vitl       -> only vitl
set -euo pipefail
cd /homes/eprato01/ATLAS-model
MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(vits vitb vitl)
for m in "${MODELS[@]}"; do
    sbatch --export=ALL,MODEL_VARIANT=${m} --job-name=atlas_eval_${m}  inference/run_eval_cholecseg8k.sh
    sbatch --export=ALL,MODEL_VARIANT=${m} --job-name=atlas_video_${m} inference/run_inference.sh
done
