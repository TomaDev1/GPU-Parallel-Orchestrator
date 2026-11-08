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
# Vision-language builds (e.g. Qwen3.8 with a vision encoder) reserve memory
# for image/video inputs. For text-only agent work on 24GB cards, turn that off:
#   EXTRA_ARGS='--limit-mm-per-prompt {"image":0,"video":0}'
# Only 4-bit builds made for vLLM (AWQ / GPTQ / W4A16 / compressed-tensors)
# fit one copy per 24GB card. GGUF repos don't load in vLLM.
#
# Logs go to logs/vllm-gpuN.log. Stop everything with: ./scripts/launch_vllm.sh stop
set -euo pipefail

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

if [[ "${1:-}" == "stop" ]]; then
  # Each server runs in its own process group (see setsid below), so killing
  # the group also takes down vLLM's EngineCore child, which otherwise keeps
  # holding the GPU after the API server exits.
  pids=()
  for f in "$LOG_DIR"/vllm-gpu*.pid; do
    [[ -e "$f" ]] || continue
    pid="$(cat "$f")"
    if kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null; then
      echo "stopping $(basename "$f" .pid)"
      pids+=("$pid")
    fi
    rm -f "$f"
  done
  for _ in $(seq 1 20); do
    alive=0
    for pid in "${pids[@]}"; do
      pgrep -g "$pid" > /dev/null 2>&1 && alive=1
    done
    [[ $alive -eq 0 ]] && break
    sleep 1
  done
  for pid in "${pids[@]}"; do
    kill -KILL -- "-$pid" 2>/dev/null || true
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

# Refuse to start on GPUs that something else is already using (for example a
# cloud template's own vLLM server): vLLM would fail later with a less obvious
# "Free memory on device ... is less than desired" error.
busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
  | awk -F', *' -v n="$NUM_GPUS" '$1 < n && $2 > 2000 {print "  GPU " $1 ": " $2 " MiB in use"}')
if [[ -n "$busy" && -z "${FORCE:-}" ]]; then
  echo "Some GPUs are already in use:" >&2
  echo "$busy" >&2
  echo "Stop whatever holds them first (check 'nvidia-smi', 'supervisorctl status'," >&2
  echo "or run '$0 stop'), or set FORCE=1 to start anyway." >&2
  exit 1
fi

# Download once up front, so four servers don't race to fetch the same files.
if [[ ! -d "$MODEL" ]]; then
  echo "downloading $MODEL (skipped if already cached)..."
  python3 -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])" "$MODEL"
fi

for ((i = 0; i < NUM_GPUS; i++)); do
  port=$((BASE_PORT + i))
  echo "starting GPU $i on port $port"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$i setsid nohup vllm serve "$MODEL" \
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
      echo "GPU $i server exited. Likely cause (error lines from $LOG_DIR/vllm-gpu$i.log):" >&2
      grep -nE "Error|error:|out of memory|OutOfMemory|not supported|No available memory|exceeds" \
        "$LOG_DIR/vllm-gpu$i.log" | grep -v "Failed core proc" | tail -n 15 >&2 || true
      echo "--- last log lines ---" >&2
      tail -n 15 "$LOG_DIR/vllm-gpu$i.log" >&2
      echo "stopping the other servers..." >&2
      LOG_DIR="$LOG_DIR" "$0" stop >&2
      exit 1
    fi
    sleep 5
  done
  echo "GPU $i ready on port $port"
done
echo "all $NUM_GPUS servers ready"
