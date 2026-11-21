#!/usr/bin/env bash
# Launch/manage persistent Inkling-2-Mini serving on dual RTX 3090 (SM86, TP=2 NVLink).
#
# Why `vllm-inkling2-sm86:latest`?
#   The stock `vllm/vllm-openai:inkling-2-mini` image compiles `_C_stable_libtorch.abi3.so`
#   and `_moe_C_stable_libtorch.abi3.so` for SM100 only (Blackwell), causing
#   `cudaErrorNoKernelImageForDevice` on SM86 for ops like `moe_align_block_size`.
#   Overlaing the matching multi-arch `vllm` nightly wheel (`cp38-abi3`, `torch==2.14.0`,
#   containing `sm_75/80/86/89/90/100/120` cubins + PR #69028) and running
#   `vllm_sm86/apply_sm86_patch.py --installed` gives native SM86 serving + DSpark
#   7-token speculative decoding with zero C++ rebuild.
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-vllm-inkling2-mini}"
IMAGE="${IMAGE:-vllm-inkling2-sm86:latest}"
MODEL_DIR="${MODEL_DIR:-/data/models/thinkingmachines/Inkling-2-Mini}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/inkling2-runtime}"
HOST_PORT="${HOST_PORT:-8001}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
ENABLE_DSPARK="${ENABLE_DSPARK:-1}"

cmd="${1:-start}"

case "$cmd" in
  start)
    mkdir -p "$CACHE_DIR"/{home,triton,torch_ext,vllm}
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

    extra_args=()
    if [[ "$ENABLE_DSPARK" == "1" ]]; then
      extra_args+=(
        --speculative-config
        '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
      )
    fi

    docker run -d \
      --name "$CONTAINER_NAME" \
      --restart unless-stopped \
      --gpus all \
      --ipc=host \
      -p "${HOST_PORT}:8000" \
      -v "${MODEL_DIR}:/model:ro" \
      -v "${CACHE_DIR}:/cache" \
      -e HOME=/cache/home \
      -e TRITON_CACHE_DIR=/cache/triton \
      -e TORCH_EXTENSIONS_DIR=/cache/torch_ext \
      -e VLLM_CACHE_ROOT=/cache/vllm \
      -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
      "$IMAGE" \
      /model \
      --served-model-name thinkingmachines/Inkling-2-Mini \
      --host 0.0.0.0 \
      --port 8000 \
      --tensor-parallel-size 2 \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --compilation-config '{"max_cudagraph_capture_size": 64}' \
      --reasoning-parser inkling \
      --enable-auto-tool-choice \
      --tool-call-parser inkling \
      "${extra_args[@]}"
    ;;
  stop)
    docker rm -f "$CONTAINER_NAME"
    ;;
  logs)
    docker logs -f --tail 100 "$CONTAINER_NAME"
    ;;
  status)
    docker ps --filter "name=^/${CONTAINER_NAME}$"
    curl -sf "http://127.0.0.1:${HOST_PORT}/v1/models" && echo ""
    ;;
  *)
    echo "Usage: $0 {start|stop|logs|status}" >&2
    exit 1
    ;;
esac
