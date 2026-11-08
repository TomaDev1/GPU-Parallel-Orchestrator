#!/usr/bin/env bash
# Start one vLLM OpenAI-compatible server per GPU, on consecutive ports.
#
#   MODEL=<hf-repo-of-a-4bit-qwen-27b> ./scripts/launch_vllm.sh
#
# Settings (environment variables):
#   MODEL           required. Hugging Face repo or local path (AWQ/GPTQ 4-bit).
#   NUM_GPUS        default: all GPUs nvidia-smi can see
#   BASE_PORT       default 8000 (GPU i listens on BASE_PORT + i)
#   MAX_MODEL_LEN   default 16384 tokens of context per request
#   GPU_MEM         default 0.92 (fraction of VRAM vLLM may use)
#   KV_DTYPE        default auto; set fp8 to roughly double KV cache room
#   SERVED_NAME     default "qwen" (the model name clients send)
#   EXTRA_ARGS      anything else to pass to `vllm serve`
#
# Logs go to logs/vllm-gpuN.log. Stop everything with: ./scripts/launch_vllm.sh stop
set -euo pipefail

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

if [[ "${1:-}" == "stop" ]]; then
  for f in "$LOG_DIR"/vllm-gpu*.pid; do
    [[ -e "$f" ]] || continue
    kill "$(cat "$f")" 2>/dev/null && echo "stopped $(basename "$f" .pid)"
    rm -f "$f"
  done
  exit 0
fi

: "${MODEL:?Set MODEL to a 4-bit Qwen 27B repo or path}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
BASE_PORT="${BASE_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEM="${GPU_MEM:-0.92}"
KV_DTYPE="${KV_DTYPE:-auto}"
SERVED_NAME="${SERVED_NAME:-qwen}"

# Download once up front, so four servers don't race to fetch the same files.
if [[ ! -d "$MODEL" ]]; then
  echo "downloading $MODEL (skipped if already cached)..."
  python3 -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])" "$MODEL"
fi

for ((i = 0; i < NUM_GPUS; i++)); do
  port=$((BASE_PORT + i))
  echo "starting GPU $i on port $port"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$i nohup vllm serve "$MODEL" \
    --port "$port" \
    --served-model-name "$SERVED_NAME" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM" \
    --kv-cache-dtype "$KV_DTYPE" \
    --enable-prefix-caching \
    ${EXTRA_ARGS:-} \
    > "$LOG_DIR/vllm-gpu$i.log" 2>&1 &
  echo $! > "$LOG_DIR/vllm-gpu$i.pid"
done

echo "waiting for servers to load the model..."
for ((i = 0; i < NUM_GPUS; i++)); do
  port=$((BASE_PORT + i))
  until curl -sf "http://localhost:$port/v1/models" > /dev/null; do
    if ! kill -0 "$(cat "$LOG_DIR/vllm-gpu$i.pid")" 2>/dev/null; then
      echo "GPU $i server exited; last log lines:" >&2
      tail -n 30 "$LOG_DIR/vllm-gpu$i.log" >&2
      exit 1
    fi
    sleep 5
  done
  echo "GPU $i ready on port $port"
done
echo "all $NUM_GPUS servers ready"
