#!/usr/bin/env bash
# 1-command remote workflow for a Dual RTX 3090 (SM86) host.
# Set `INKLING_REMOTE_HOST=user@your-gpu-host` before invoking.
#
# Usage from local workstation (`~/inkling2-sm86-kernels`):
#   INKLING_REMOTE_HOST=ubuntu@gpu-box ./scripts/remote.sh probe
#   ./scripts/remote.sh setup       # Sync repo + create remote uv venv with CUDA PyTorch & Triton
#   ./scripts/remote.sh lab0        # Run Lab 0: GDDR6X bandwidth, Tensor Core TFLOPS & P2P latency
#   ./scripts/remote.sh lab1        # Run Lab 1: Fused Triton vs SM86 CUDA C++ (`cp.async` + shfl)
#   ./scripts/remote.sh lab2        # Run Lab 2: SM86 Sheared-Bias Paged FlashAttention + Split-KV
#   ./scripts/remote.sh lab3        # Run Lab 3: PTX `prmt.b32` NVFP4 & MXFP8 Tensor Core GEMM
#   ./scripts/remote.sh test        # Run full pytest suite on the 3090s
#   ./scripts/remote.sh vllm-probe  # Pull `vllm/vllm-openai:inkling-2-mini`, apply SM86 patch & smoke-test!

set -euo pipefail

REMOTE_HOST="${INKLING_REMOTE_HOST:?Set INKLING_REMOTE_HOST=user@host before running remote.sh}"
REMOTE_DIR="${INKLING_REMOTE_DIR:-inkling2-sm86-kernels}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Default to a pre-installed CUDA 13 + nvcc + PyTorch + Triton + Humming container on the GPU host.
# Override with INKLING_RUNTIME=uv to run in a host uv venv instead.
RUNTIME="${INKLING_RUNTIME:-docker}"
DOCKER_IMAGE="${INKLING_DOCKER_IMAGE:-vllm/vllm-openai:v0.27.1}"

sync_repo() {
  echo ">>> Syncing ${LOCAL_DIR} -> ${REMOTE_HOST}:~/${REMOTE_DIR}"
  rsync -az --delete \
    --exclude '.venv' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '.pytest_cache' \
    --exclude '.cache' \
    --exclude 'artifacts' \
    "${LOCAL_DIR}/" "${REMOTE_HOST}:${REMOTE_DIR}/"
  if [ -d "${LOCAL_DIR}/.cache" ]; then
    rsync -az "${LOCAL_DIR}/.cache/" "${REMOTE_HOST}:${REMOTE_DIR}/.cache/"
  fi
}

run_remote() {
  local inner_cmd="$1"
  if [ "${RUNTIME}" = "docker" ]; then
    ssh "${REMOTE_HOST}" "
      set -euo pipefail
      REMOTE_ABS=\"\$(cd ${REMOTE_DIR} && pwd)\"
      mkdir -p \"\$HOME/.cache/inkling2-runtime/triton\" \"\$HOME/.cache/inkling2-runtime/torch_ext\" \"\$HOME/.cache/inkling2-runtime/home\"
      docker run --rm --gpus all --ipc=host \
        -u \$(id -u):\$(id -g) \
        -e HOME=/remote_home \
        -e PYTHONPATH=/workspace/inkling2-sm86-kernels \
        -e TORCH_CUDA_ARCH_LIST=8.6 \
        -e TORCH_EXTENSIONS_DIR=/remote_torch_ext \
        -e TRITON_CACHE_DIR=/remote_triton \
        -v \"\${REMOTE_ABS}:/workspace/inkling2-sm86-kernels\" \
        -v \"\$HOME/.cache/inkling2-runtime/home:/remote_home\" \
        -v \"\$HOME/.cache/inkling2-runtime/torch_ext:/remote_torch_ext\" \
        -v \"\$HOME/.cache/inkling2-runtime/triton:/remote_triton\" \
        -w /workspace/inkling2-sm86-kernels \
        --entrypoint bash \
        ${DOCKER_IMAGE} -lc $(printf '%q' "${inner_cmd}")
    "
  else
    ssh "${REMOTE_HOST}" "export PATH=\"\$HOME/.local/bin:\$PATH\" && cd ${REMOTE_DIR} && ${inner_cmd}"
  fi
}

cmd="${1:-probe}"
shift || true

case "${cmd}" in
  probe)
    ssh "${REMOTE_HOST}" '
      set -e
      echo "=== OS & CPU ==="
      uname -a
      lscpu | grep -E "Model name|CPU\(s\):" || true
      echo -n "SIMD flags: "; grep -oE "avx2|avx512f|fma" /proc/cpuinfo | sort -u | tr "\n" " "; echo ""
      echo "=== MEMORY & DISK ==="
      free -h
      df -h / /data /mnt 2>/dev/null | uniq
      echo "=== GPUS & INTERCONNECT TOPOLOGY ==="
      nvidia-smi
      nvidia-smi topo -m
      echo "=== CUDA & CONTAINER RUNTIME ==="
      which nvcc docker uv nsys ncu 2>&1 || true
      nvcc --version 2>/dev/null || true
      docker --version 2>/dev/null || true
    '
    ;;

  sync)
    sync_repo
    ;;

  setup)
    sync_repo
    ssh "${REMOTE_HOST}" "
      set -e
      if ! command -v uv >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
      fi
      export PATH=\"\$HOME/.local/bin:\$PATH\"
      cd ${REMOTE_DIR}
      uv sync --python 3.12
      uv run python -c 'import torch, triton; print(\"CUDA:\", torch.cuda.is_available(), torch.cuda.device_count(), \"Triton:\", triton.__version__)'
    "
    ;;

  lab0|lab1|lab2|lab3|lab3b)
    sync_repo
    case "${cmd}" in
      lab0) lab_script="labs/lab0_sm86_roofline.py" ;;
      lab1) lab_script="labs/lab1_loop_and_convmix.py" ;;
      lab2) lab_script="labs/lab2_sm86_rel_attention.py" ;;
      lab3) lab_script="labs/lab3_sm86_nvfp4_mxfp8_moe.py" ;;
      lab3b) lab_script="labs/lab3b_sm86_batched_moe.py" ;;
    esac
    if [ "${RUNTIME}" = "docker" ]; then
      run_remote "python3 ${lab_script} $*"
    else
      run_remote "uv run python ${lab_script} $*"
    fi
    ;;

  test)
    sync_repo
    if [ "${RUNTIME}" = "docker" ]; then
      run_remote "pytest -v $*"
    else
      run_remote "uv run pytest -v $*"
    fi
    ;;

  run)
    sync_repo
    run_remote "$*"
    ;;

  vllm-probe)
    sync_repo
    ssh "${REMOTE_HOST}" "
      set -e
      cd ${REMOTE_DIR}
      echo '>>> Pulling vllm/vllm-openai:inkling-2-mini and testing SM86 bring-up patch inside container...'
      docker pull vllm/vllm-openai:inkling-2-mini
      docker run --rm --gpus all --ipc=host \
        -v \$HOME/.cache/huggingface:/root/.cache/huggingface \
        -v \$(pwd):/workspace/inkling2-sm86-kernels \
        --entrypoint bash \
        vllm/vllm-openai:inkling-2-mini -c '
          set -e
          python3 /workspace/inkling2-sm86-kernels/vllm_sm86/apply_sm86_patch.py --installed
          python3 -c \"
import torch
from vllm.models.inkling2.quant_config import Inkling2QuantConfig
from vllm.models.inkling2.nvidia.ops import supports_pdl
print(\\\"GPUs:\\\", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print(\\\"Inkling2QuantConfig.get_min_capability():\\\", Inkling2QuantConfig.get_min_capability())
print(\\\"supports_pdl():\\\", supports_pdl())
\"
        '
    "
    ;;

  *)
    echo "Unknown subcommand: ${cmd}"
    echo "Available: probe | sync | setup | lab0 | lab1 | lab2 | lab3 | lab3b | test | vllm-probe"
    exit 1
    ;;
esac
