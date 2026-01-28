#!/bin/bash
#SBATCH --job-name=vllm_dec_allin1
#SBATCH --partition=gpu_course
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:30:00
#SBATCH --output=logs/vllm_dec_allin1_%j.log
#SBATCH --error=logs/vllm_dec_allin1_%j.err

set -euo pipefail

# --- Parameters per split ---
SPLIT=${1:?Usage: sbatch slurm_jobs/vllm_decode_split_all_in_one.sbatch <train|val|test>}
export PORT=$(shuf -i 10000-60000 -n 1)  # random port om conflicts te voorkomen
MODE=no_think
CONCURRENCY=8

SCRIPTS="/gpfs/scratch1/shared/scur2292/MLOps_2026_upstream/vllm/src/python_scripts"
CONTAINER="/projects/2/managed_datasets/containers/vllm/cuda-13.0-vllm.sif"
MODEL="Qwen/Qwen3-4B-AWQ"

IN="data/processed/text/v1/decoder_input_${SPLIT}.jsonl"
OUT="data/processed/text/v1/decoder_output_${SPLIT}.jsonl"

echo "[INFO] NODE=$(hostname)"
echo "[INFO] SPLIT=$SPLIT"
echo "[INFO] IN=$IN"
echo "[INFO] OUT=$OUT"
echo "[INFO] PORT=$PORT"

# --- 1) Start vLLM server in background (same node!) ---
apptainer exec --nv -B "$PWD" "$CONTAINER" \
  vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --max-num-seqs 8 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.90 \
  > "logs/vllm_server_inline_${SPLIT}_$SLURM_JOB_ID.log" 2>&1 &

VLLM_PID=$!
echo "[INFO] vLLM server PID=$VLLM_PID"

# --- 2) Wait until server is ready ---
echo "[INFO] Waiting for vLLM to become ready..."
for i in {1..60}; do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo "[INFO] vLLM is ready!"
    break
  fi
  sleep 2
done

# Hard fail if still not ready
curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null || (echo "[ERROR] vLLM not reachable on localhost:${PORT}" && kill $VLLM_PID && exit 1)

# --- 3) Run the batch decoder (connects to localhost) ---
apptainer exec --nv -B "$PWD" "$CONTAINER" \
  python3 "$SCRIPTS/batch_inference_decoder.py" \
    --input "$IN" \
    --output "$OUT" \
    --port "$PORT" \
    --mode "$MODE" \
    --concurrency "$CONCURRENCY"

# --- 4) Stop server ---
echo "[INFO] Stopping vLLM server..."
kill $VLLM_PID || true
sleep 2
echo "[INFO] Done."

#port!!