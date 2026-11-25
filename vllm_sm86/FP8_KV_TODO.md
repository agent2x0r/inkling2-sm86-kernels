# Software FP8 (`E4M3FN`) KV Cache for Inkling-2-Mini on SM86

## Status: Complete — Live Full-Server Verified on Dual RTX 3090 (SM86, TP=2, DSpark)

All items are implemented in [`apply_sm86_patch.py`](apply_sm86_patch.py) (idempotent against both fresh upstream wheels and previously SM86-patched images) and [`serve_inkling2_sm86.sh`](serve_inkling2_sm86.sh) (`KV_CACHE_DTYPE=fp8`).

## Hardware Blockers Found & Solved
1. **Triton rejects `fp8e4nv` kernel arguments on `< SM89`:**
   Even passing a `torch.float8_e4m3fn` pointer to any Triton kernel on SM86 raises `ValueError("type fp8e4nv not supported in this architecture")` at AST→TTIR lowering.
   - **Fix (`nvidia/attention.py`):** Keep vLLM's native `torch.uint8` byte storage (`kv_cache` without `.view(torch.float8_e4m3fn)`) when `not supports_fp8_kv_ptx()` (`< SM89`). Both `attention_prologue` (`cache.dtype.element_ty == tl.uint8`) and `_rel_attention_kernel` (`primitive_bitwidth == 8`) dispatch cleanly at compile time with zero impact on BF16 KV cache.
2. **SM86 KV-store path (`nvidia/ops/sconv.py`):**
   `_kv_conv_tile` stores L2-clipped K (`<= 22.627`) and RMS-normalized V (`RMS = 1.0`) directly in unit-scale E4M3FN (`[-448, 448]`).
   - **Fix:** Added `_fp32_to_e4m3fn_u8` — pure Triton exact IEEE-754 round-to-nearest-even FP32→E4M3FN (`uint8`) conversion (normal RNE drop-20-bits + subnormal magic `+ 16384.0` hardware RNE add carrying `7.5 * 2^-9 -> 0x08`), verified **200,125 / 200,125 bit-exact** against `torch.float8_e4m3fn`.
3. **SM86 KV-load dequant (`nvidia/ops/attention_sm90.py`):**
   Upstream `_load_fp8_kv` emits SM89+ `cvt.rn.f16x2.e4m3x2`.
   - **Fix:** Replaced on `< SM89` with a 5-op SM80/SM86 PTX unpack (`prmt.b32 0x1707` + `and` + `shr 1` + `lop3 0xEC` + non-FTZ `mul.rn.f16x2` by `256.0` = `0x5C005C00`), verified **bit-exact across all 254 finite E4M3FN values** (normals, subnormals, and `±0.0`) and **`0.00e+00` max diff** vs dequantized BF16 KV across `rel_attention` and split-KV `rel_decode_attention`.
4. **Capability & DSpark Draft Guards (`nvidia/attention.py`, `speculator_config.py`, `v1/worker/gpu/spec_decode/speculator_config.py`):**
   - Enabled `("float16", "fp8", "fp8_e4m3")` in `Inkling2Attention` on `< SM90` and overrode `Inkling2AttentionBackend.supports_kv_cache_dtype`.
   - Extended `Inkling2DSparkSpeculatorConfig.adapt_draft_model_config` to `< SM90` so the 3-layer DSpark draft stays in BF16 KV (`spec.kv_cache_dtype = "auto"`).
   - Fixed upstream `_resolve_checkpoint_config_class` returning base `DSparkSpeculatorConfig` instead of draft-declared `Inkling2DSparkSpeculatorConfig` when `method="dspark"` is passed explicitly (which previously caused `adapt_draft_model_config` to be skipped!).

## Verified SM86 (RTX 3090) Kernel Microbenchmarks (L2-Flushed, TP=2 Shape `H=12, HKV=4, D=128`)

Measured on an idle RTX 3090 via [`benchmarks/bench_fp8_kv_kernel.py`](../benchmarks/bench_fp8_kv_kernel.py) (checked-in artifact: [`benchmarks/results_kernel_fp8kv.json`](../benchmarks/results_kernel_fp8kv.json)).

| Attention Regime | BF16 KV | Software FP8 KV | Speedup vs BF16 | Eff. FP8 Read BW | Max Diff vs Dequant BF16 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Decode (`q=1`, 16K KV, 20 splits)** | `66.6 us` | **`55.3 us`** | **`1.20x`** | `303.4 GB/s` | `0.0e+00` |
| **Decode (`q=1`, 64K KV, 20 splits)** | `180.2 us` | **`121.9 us`** | **`1.48x`** | `550.7 GB/s` | `0.0e+00` |
| **Decode (`q=1`, 128K KV, 20 splits)** | `340.0 us` | **`208.9 us`** | **`1.63x`** | `642.5 GB/s` | `0.0e+00` |
| **DSpark Verify (`q=8`, 16K KV, 20 splits)** | `71.8 us` | **`61.4 us`** | **`1.17x`** | `273.1 GB/s` | `0.0e+00` |
| **DSpark Verify (`q=8`, 64K KV, 20 splits)** | `185.3 us` | **`150.5 us`** | **`1.23x`** | `445.8 GB/s` | `0.0e+00` |
| **DSpark Verify (`q=8`, 128K KV, 20 splits)** | `340.0 us` | **`285.7 us`** | **`1.19x`** | `469.8 GB/s` | `0.0e+00` |
| **Prefill (`q=256`, 32K KV, BM128)** | `2,798.6 us` | `3,072.0 us` | `0.91x` | compute-bound | `0.0e+00` |

End-to-end patched vLLM `attention_prologue` + `rel_attention` + `rel_decode_attention` parity on RTX 3090:
- **SWA (`W=1024`, static conv):** `max_diff = 0.00e+00` vs dequant BF16; `cos = 0.9993` (prefill) / `0.9991` (split-KV verify) vs unquantized BF16.
- **Full (`W=0`, dynamic conv):** `max_diff = 0.00e+00` vs dequant BF16; `cos = 0.9994` (prefill) / `0.9993` (split-KV verify) vs unquantized BF16.

## Live Full-Server Results on 2× RTX 3090 (TP=2, 250W, DSpark `num_spec=7`, `gpu_memory_utilization=0.98`)

### 1. Unified KV Cache Capacity (Why It Jumped **2.02×**)
- **BF16 KV (`auto`):** `7,712` blocks (`8.47 GiB/GPU`), **18 padding layers** across hybrid groups (up to 350% waste in the draft group) → **`239,061` tokens** (`1.12×` at `212,992` max len).
- **Software FP8 KV (`fp8`) + BF16 DSpark draft:** Target FP8 (`block_size=128`, 1 B/elem) matches BF16 DSpark draft (`block_size=64`, 2 B/elem) 1:1 in page size → **0 padding layers**, `67,664` blocks (`8.26 GiB/GPU`) → **`482,810` unified KV tokens** (**`2.02×`** capacity, `1.13×` at **`425,984` / `416K`** single-request `max_model_len`, `7s` / `0.34 GiB` CUDA graph capture).

### 2. Long-Context Scaling & Needle Recall (0% Prefix Hits — `results_dspark_fp8kv.json`, `results_longctx_fp8kv.json`)

| Bucket | Prompt Tokens | TTFT | Prefill | Decode (DSpark) | Mean Acc | Needle |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| **1K** | `1,144` | `0.27s` | `4,273 tok/s` | `182.4 tok/s` | `2.92` | `3/3` |
| **4K** | `4,335` | `0.81s` | `5,424 tok/s` | `193.3 tok/s` | `3.13` | `3/3` |
| **16K** | `16,997` | `2.91s` | `5,837 tok/s` | `177.8 tok/s` | `2.91` | `3/3` |
| **32K** | `34,017` | `6.18s` | `5,501 tok/s` | `198.5 tok/s` | `3.34` | `3/3` |
| **60K** | `63,296` | `12.93s` | `4,896 tok/s` | `157.0 tok/s` | `2.74` | `3/3` |
| **64K** | `67,163` | `13.75s` | `4,886 tok/s` | `148.7 tok/s` | `2.59` | `1/1` |
| **128K** | `130,946` | `33.45s` | `3,914 tok/s` | `146.5 tok/s` | `2.76` | `1/1` |
| **200K** | `204,739` | `64.42s` | `3,178 tok/s` | `125.4 tok/s` | `2.59` | `1/1` |
| **256K** | `265,665` | `96.48s` | `2,754 tok/s` | `111.5 tok/s` | `2.46` | `1/1` |
| **320K** | `336,567` | `140.80s` | `2,390 tok/s` | **`98.1 tok/s`** | `2.34` | `1/1` |

### 3. Accuracy & Functional Parity (`results_accuracy_text_fp8kv.json` vs BF16 Baseline)
- **HumanEval (164/164):** **`163/164` (`99.39%`)** at `597.7 tok/s` agg (`c=8`) — vs BF16 `160/164` (`97.56%`).
- **GSM8K (200):** **`195/200` (`97.5%`)** at `552.4 tok/s` agg (`c=8`) — vs BF16 `196/200` (`98.0%`).
- **Vision (SM86 FA2):** `2/2 PASS` (`480×240` shapes + `1024×1024` grid).
- **Tool Calling (`--tool-call-parser inkling`):** `2/2 PASS` exact JSON args.
- **Needle Recall (1K–337K):** **`20/20` (`100%`)** with `prefix_cache_hits = 0`.
- **Concurrency (`c=8`, ~4K uncached prompt/user):** **`384.4 tok/s`** aggregate (matches BF16 `383.4 tok/s`).

