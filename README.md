# Inkling-2-Mini on Ampere (SM86 / Dual RTX 3090) in Upstream vLLM V1

Native **NVFP4/MXFP8 MoE + Vision (FA2) + Tool Calling + 7-Token DSpark Speculative Decoding** for [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on **Ampere (SM80/SM86)** GPUs inside **upstream vLLM V1** — **zero C++ compilation required**.

![Inkling-2-Mini SM86 Benchmark Summary](benchmarks/inkling2_sm86_summary.png)

## Summary of Verified Results (2× RTX 3090, 225W Cap, NVLink, TP=2)

Tested on **2× NVIDIA GeForce RTX 3090** (`SM86`, 24 GiB GDDR6X each, 225 W power limit, NV4 NVLink bridge measuring `52.7 GB/s` unidirectional P2P and `840.4 GB/s` GDDR6X bandwidth), driven by an AMD FX-8350 (2012) host sharing the GPUs with ~2.2 GiB/GPU of co-resident services.

- **Single-user decode (Base / no spec decode):** `114.7 ± 0.1 tok/s` (1K) → **`99.4 ± 0.1 tok/s`** (61.2K uncached context, 0% prefix hits)
- **Single-user decode (DSpark, 7 draft tokens):** **`166–178 tok/s`** from 1K to 33.5K and **`170.2 ± 21.3 tok/s`** at **61.2K uncached context**; **`209.9 tok/s`** on structured JSON and **`202.4 tok/s`** on code
- **Uncached prefill:** **`4,918–5,709 tok/s`** (1K–16K) and **`4,954 tok/s`** at 61.2K (`12.36 s` TTFT)
- **Multi-user concurrency (`~4K` uncached prompt/user, unified paged KV):** DSpark scales monotonically **`190.6 → 277.9 → 352.6 → 383.4 tok/s`** aggregate (`c = 1, 2, 4, 8`), beating Base (`113.5 → 173.9 → 249.0 → 328.1 tok/s`) at every concurrency level; short-prompt coding reaches **`697.1 tok/s`** aggregate (`c = 8`)
- **Unified Hybrid KV Cache & `n`-Parallel Sampling:** Native vLLM V1 `HybridKVCacheManager` (`81,966` unified KV tokens across `5,712` paged blocks and `max_num_seqs = 32`), Automatic Prefix Caching (`99.2%` block reuse on shared 6.2K prompts), and OpenAI `n > 1` parallel sampling (`n = 4` verified at `358.1 tok/s` aggregate)
- **Accuracy on SM86 DSpark:** **`97.56%`** HumanEval (`160/164`; `96.34%` raw before fixing a harness helper-prefix bug), **`98.0%`** GSM8K (`196/200`; `95.0%` strict raw), **`30/30`** long-context needle recall (1K–61.2K), **`cos = 0.999998`** vision FA2 numerical parity vs FP32 SDPA, and `2/2` structured tool calling

---

## Why Stock `vllm/vllm-openai:inkling-2-mini` Fails on SM86 (and the Fix)

1. **SM100-only compiled extensions in the stock Docker image.**
   Inspecting `vllm/vllm-openai:inkling-2-mini` (`sha256:c2dc759f5984...`) reveals it was built with `torch_cuda_arch_list="10.0+PTX 10.3+PTX"`. Over 99% of kernels in `_C_stable_libtorch.abi3.so` and `_moe_C_stable_libtorch.abi3.so` (including `moe_align_block_size`, `reshape_and_cache`, `rms_norm`, and `rotary`) lack `sm_80`/`sm_86` cubins, throwing `cudaErrorNoKernelImageForDevice`.
   - **Fix:** Overlay vLLM's matching multi-arch `cp38-abi3` nightly wheel (`vllm-0.34.0rc1.dev1126+g7e792e6e1`, `torch==2.14.0+cu130`, shipping `sm_75/80/86/89/90/100/120` cubins + PR #69028).

2. **Hopper-only PTX & guards across `vllm/models/inkling2` (13 Python files).**
   `humming 0.1.20` already JIT-compiles native `kernel.sm_86.cubin` kernels for NVFP4 and MXFP8 routed MoE, `ops/attention_sm90.py` runs pure Triton relative-bias paged attention, `ops/allreduce.py` already self-disables Hopper multicast on `< SM90`, and `vllm_flash_attn` already ships `sm_80` FA2. [`vllm_sm86/apply_sm86_patch.py`](vllm_sm86/apply_sm86_patch.py) patches only the five remaining Hopper assumptions:
   - `quant_config.py`: `get_min_capability()` `90 → 80`.
   - `nvidia/ops/__init__.py` + 9 `ops/*.py` Triton files: gate Hopper Programmatic Dependent Launch (`launch_pdl`, `tl.extra.cuda.gdc_wait()`, `tl.extra.cuda.gdc_launch_dependents()`) behind fork-safe `current_platform.has_device_capability(90)` with no-op `@triton.jit` helpers on `< SM90`.
   - `nvidia/ops/linear.py`: disable Hopper TMA `TensorDescriptor` on `< SM90` so Ampere uses `_k_group_kernel`.
   - `nvidia/ops/conv_mix.py`: route `< SM90` to the existing Triton `short_conv` + `attn_res` fallback (same path SM120 uses).
   - `nvidia/vision.py`: select FlashAttention-2 (`fa_version=2`) on `< SM90` instead of Hopper FA3.

---

## 3-Step Quickstart (Reproduce in ~2 Minutes)

### 1. Build the Zero-C++-Rebuild SM86 Image
```bash
git clone https://github.com/agent2x0r/inkling2-sm86-kernels.git
cd inkling2-sm86-kernels

docker build -f vllm_sm86/Dockerfile.sm86 -t vllm-inkling2-sm86:latest .
```

### 2. Launch Persistent Serving on Dual RTX 3090 (`TP=2`, 65K Context, DSpark)
Download `thinkingmachines/Inkling-2-Mini` locally (e.g. `/data/models/thinkingmachines/Inkling-2-Mini`), then run:
```bash
MODEL_DIR=/data/models/thinkingmachines/Inkling-2-Mini \
HOST_PORT=8001 \
ENABLE_DSPARK=1 \
./vllm_sm86/serve_inkling2_sm86.sh start

./vllm_sm86/serve_inkling2_sm86.sh status
```
Set `ENABLE_DSPARK=0` to benchmark Base (no speculative decoding) with identical flags.

### 3. Reproduce Benchmarks & Chart
```bash
# Functional + speed suite (vision, tools, 12 greedy prompts, 1K..60K scaling, categories, c=1..8)
python3 benchmarks/suite_functional_and_speed.py \
  --mode dspark \
  --base-url http://127.0.0.1:8001 \
  --out benchmarks/results_dspark.json

# Accuracy suite (HumanEval 164/164, GSM8K 200, MMMU-Pro 100)
python3 benchmarks/suite_accuracy.py --base-url http://127.0.0.1:8001 --out benchmarks/results_accuracy.json

# Regenerate the PNG summary chart directly from the JSON artifacts
uv run --with matplotlib python3 benchmarks/plot_summary.py
```

---

## Full Benchmark Results (from Checked-In JSON Artifacts)

All speed tests use real Python 3.12 stdlib source files with a unique nonce per trial and **verified `prefix_cache_hits = 0`** via `/metrics` (`512` output tokens, `temperature = 0.6`, `reasoning_effort = "low"`, 3 trials ± 1σ). Raw JSON outputs live in [`benchmarks/`](benchmarks/).

### 1. Context-Length Scaling (`results_base.json` vs `results_dspark.json`)

| Context | Avg Prompt | Prefill (tok/s) | Base TTFT | Base Decode | DSpark TTFT | DSpark Decode | Speedup | DSpark Acc | Needles |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **1K** | `928` | `4,918 ± 37` | `0.22 ± 0.05s` | `114.7 ± 0.1 /s` | `0.19 ± 0.00s` | **`172.2 ± 16.7 /s`** | `1.50×` | `2.83` | `6/6` |
| **4K** | `3,858` | `5,575 ± 197` | `0.72 ± 0.08s` | `113.6 ± 0.2 /s` | `0.69 ± 0.03s` | **`166.0 ± 4.2 /s`** | `1.46×` | `2.75` | `6/6` |
| **16K** | `16,694` | `5,709 ± 4` | `2.97 ± 0.08s` | `110.0 ± 0.1 /s` | `2.92 ± 0.00s` | **`178.3 ± 11.4 /s`** | `1.62×` | `3.01` | `6/6` |
| **32K** | `33,506` | `5,439 ± 7` | `6.16 ± 0.01s` | `105.6 ± 0.1 /s` | `6.16 ± 0.01s` | **`177.9 ± 13.7 /s`** | `1.69×` | `3.09` | `6/6` |
| **60K** | `61,211` | `4,954 ± 7` | `12.38 ± 0.02s` | **`99.4 ± 0.1 /s`** | `12.36 ± 0.02s` | **`170.2 ± 21.3 /s`** | `1.71×` | `3.06` | `6/6` |

### 2. Output-Category Decode Speed (Workload Entropy)

| Category | Base Decode | DSpark Decode | Speedup | DSpark Mean Acc |
| :--- | ---: | ---: | ---: | ---: |
| **Structured JSON** | `115.7 ± 0.1 /s` | **`209.9 ± 6.8 /s`** | `1.81×` | `3.47 ± 0.17` |
| **Code (Python/CUDA)** | `115.6 ± 0.1 /s` | **`202.4 ± 30.4 /s`** | `1.75×` | `3.31 ± 0.52` |
| **Math Reasoning** | `115.8 ± 0.1 /s` | **`177.6 ± 10.4 /s`** | `1.53×` | `2.79 ± 0.17` |
| **Technical Prose** | `115.5 ± 0.0 /s` | **`135.5 ± 2.6 /s`** | `1.17×` | `2.21 ± 0.03` |

### 3. Multi-User Concurrency (`~4K` Unique Uncached Prompt per User, 0% Prefix Cache)

| Concurrency | Base Per-User | Base Aggregate | DSpark Per-User | DSpark Aggregate | Speedup | DSpark Acc |
| :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `c = 1` | `113.4 /s` | `113.5 /s` | `190.5 /s` | **`190.6 /s`** | `1.68×` | `3.17` |
| `c = 2` | `87.0 /s` | `173.9 /s` | `147.2 /s` | **`277.9 /s`** | `1.60×` | `3.10` |
| `c = 4` | `69.5 /s` | `249.0 /s` | `109.3 /s` | **`352.6 /s`** | `1.42×` | `3.25` |
| `c = 8` | `50.9 /s` | `328.1 /s` | `67.8 /s` | **`383.4 /s`** | `1.17×` | `3.08` |

### 4. Accuracy & Functional Validation (`results_accuracy_text.json`, `results_accuracy_mmmu.json`)

- **HumanEval (164/164, isolated subprocess tests, `reasoning_effort=low`):** **`158/164` (`96.34%`)** raw → **`160/164` (`97.56%`)** after restoring prompt helper functions (`encode_cyclic`, `encode_shift`) stripped by the harness on `/38` & `/50`; `697.1 tok/s` aggregate (`c=8`).
- **GSM8K (200 test items, `reasoning_effort=low`):** **`190/200` (`95.0%`)** strict raw → **`196/200` (`98.0%`)** unit-normalized (`160 minutes`, `60\%`, etc.); `564.6 tok/s` aggregate (`c=8`).
- **Vision (SM86 FA2):** `max_abs = 0.0018`, `mean_abs = 0.00010`, **`cos = 0.999998`** vs FP32 SDPA; synthetic `480×240` shapes & `1024×1024` grid `2/2 PASS`. On a 100-item stratified MMMU-Pro Standard 10 slice (`reasoning_effort=medium`, `max_tokens=4096`), raw accuracy was `58/100` due to ~25% of failures hitting the `4096` token budget mid-reasoning (`pred=null`) and 3 HTTP 400s from an oversized >4096 px image and a 5-image item; reproducing the published `74.9%` reference requires `reasoning_effort=high`, `12K+` token budget, and resizing images to `<= 4096 px`.
- **Tool Calling (`--tool-call-parser inkling`):** `2/2 PASS` with exact JSON arguments.

---

## Repository Structure

```text
vllm_sm86/
  apply_sm86_patch.py          # Idempotent ~50-line SM86 patcher for vllm/models/inkling2
  Dockerfile.sm86              # Pinned zero-C++-rebuild Ampere Docker image
  serve_inkling2_sm86.sh       # Persistent TP=2 vLLM + DSpark launcher
benchmarks/
  suite_functional_and_speed.py
  suite_accuracy.py
  plot_summary.py              # Regenerates inkling2_sm86_summary.png from JSONs
  inkling2_sm86_summary.png
  results_base.json            # Full paired Base benchmark output
  results_dspark.json          # Full paired DSpark benchmark output
  greedy_dspark.json           # 12-prompt greedy parity capture
  results_accuracy_text.json   # Full HumanEval (164) + GSM8K (200) per-item results
  results_accuracy_mmmu.json   # 100-item stratified MMMU-Pro per-item results
labs/ & csrc/                  # Standalone SM86 CUDA C++ & Triton kernel labs (Roofline, ShortConv, Sheared-Bias Attention, PTX lop3 NVFP4/MXFP8 MoE)
tests/                         # Unit tests for reference numerics & SM86 vLLM patcher
```

## License

MIT — Copyright (c) 2026 Nate Brown `<agent2x0r.xsfmy@passmail.net>`.
