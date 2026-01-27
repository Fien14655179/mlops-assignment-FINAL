#!/bin/bash
#SBATCH --job-name=vllm_dec_batch
#SBATCH --partition=gpu_course
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/vllm_batch_%j.log
#SBATCH --error=logs/vllm_batch_%j.err
#SBATCH --nodelist=gcn9

set -euo pipefail

SPLIT=${1:?Usage: sbatch vllm_batch_decoder.sbatch <train|val|test>}
PORT=8000
MODE=no_think
CONCURRENCY=8

SCRIPTS="/gpfs/scratch1/shared/scur2292/MLOps_2026_upstream/vllm/src/python_scripts"
CONTAINER="/projects/2/managed_datasets/containers/vllm/cuda-13.0-vllm.sif"

IN="data/processed/text/v1/decoder_input_${SPLIT}.jsonl"
OUT="data/processed/text/v1/decoder_output_${SPLIT}.jsonl"

echo "[INFO] NODE=$(hostname)"
echo "[INFO] SPLIT=$SPLIT"
echo "[INFO] IN=$IN"
echo "[INFO] OUT=$OUT"
echo "[INFO] PORT=$PORT"
echo "[INFO] MODE=$MODE"
echo "[INFO] CONCURRENCY=$CONCURRENCY"

apptainer exec --nv -B "$PWD" "$CONTAINER" \
  python3 "$SCRIPTS/batch_inference_decoder.py" \
    --input "$IN" \
    --output "$OUT" \
    --port "$PORT" \
    --mode "$MODE" \
    --concurrency "$CONCURRENCY"
