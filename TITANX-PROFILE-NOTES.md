# TITAN X (Maxwell) Optimization Notes - Scion-35B-A3B (plain ternary)

Model: Scion-35B-A3B-PQ2_0-corr.gguf (HF cache: SkyIsNotGreen/Scion-35B-A3B)
- 11.3 GiB on disk, GGUF v3, arch qwen35moe, 973 tensors, 34.89B params (A3B = ~3B active)
- llama-bench reports type "Q8_0" (custom PQ2_0-corr ternary storage; check what type tensors actually use)
- No MTP drafter used (per user instruction)

Hardware: NVIDIA GeForce GTX TITAN X, cc 5.2 (Maxwell), 12200 MiB VRAM, VMM yes
Driver 580.178.04, CUDA 13.0 runtime
Build: /home/mo/llama.cpp-prism-moe-corr, build 3a25590, GGML_CUDA=ON, CMAKE_CUDA_ARCHITECTURES=52, Release

## Baseline (task 1)

Date: 2026-10-11

Command:
  ./build/bin/llama-bench -m <model> -ngl 99 -p 512 -n 128 -r 3 -o md

Results (full GPU offload, ngl 99):
| test  | t/s             |
|-------|-----------------|
| pp512 | 267.94 +/- 1.52 |
| tg128 |  37.62 +/- 0.09 |

Full offload fits in VRAM (no layer split).

## Profiling (task 2)

Method:
- A/B experiments (FA on/off, KV f16/q8_0, FORCE_MMQ/FORCE_CUBLAS, threads 4/8/6): all within noise of
  baseline (267/37.6). Bottleneck is not the matmul kernel choice, attention impl, or CPU threads.
- Per-op GPU timing via temporary GGML_CUDA_NODE_TIMES instrumentation (CUDA events around each node,
  CUDA graphs disabled, 1 decode step profiled). Total kernel time 29.31 ms/step (graphs-on baseline step: 26.6 ms).

Per-op breakdown of one decode step (29.31 ms total):
| op          | ms    | %    | nodes |
|-------------|-------|------|-------|
| MUL_MAT     | 18.47 | 63.0 | 631   |
| MUL_MAT_ID  |  5.54 | 18.9 | 120   |
| ADD         |  1.11 |  3.8 | 172   |
| GLU         |  0.60 |  2.1 | 110   |
| SCALE       |  0.59 |  2.0 | 120   |
| MUL         |  0.51 |  1.8 | 80    |
| L2_NORM     |  0.48 |  1.6 | 60    |
| CPY         |  0.40 |  1.4 | 30    |
| FLASH_ATTN  |  0.35 |  1.2 | 10    |
| others      |  1.26 |  4.3 |       |

MUL_MAT reference points (Q8_0 MMVQ kernels):
- lm_head GEMV [2048x248320]: 2.73 ms for 540 MB -> ~198 GiB/s (achievable ceiling with this kernel)
- attn_qkv GEMVs [2048x8192]: ~0.10 ms each -> ~180 GiB/s (fine)
- small shared-expert GEMVs (ffn_gate/up/shexp, ~1.1 MB): ~55-60 GiB/s (small-K latency bound)

MUL_MAT_ID (PQ2_0 experts, 8 active of 256):
- down [512x2048x256]: 2.97 ms / 40 nodes -> ~30 GiB/s weight bytes (worst offender)
- gate/up [2048x512x256]: 2.57 ms / 80 nodes -> ~68 GiB/s weight bytes
- weights are only ~2.2 MB per node; each output-row block re-reads the full Q8 activation row
  (2048 elems for down), so redundant activation traffic likely dominates

Conclusions:
- Decode is GPU-kernel-bound (kernel sum ~ step time), not launch-bound.
- Biggest headroom: MUL_MAT_ID expert GEMVs (5.54 ms, 4-6x below the ~198 GiB/s the same kernel
  achieves on large shapes) and small-K Q8_0 GEMVs of the shared experts.
- Fixing MUL_MAT_ID alone is worth ~4 ms of the 26.6 ms step (~15%+ tg speedup).

## Implementation (task 3)

Change: tuned decode-MoE GEMV geometry in `ggml/src/ggml-cuda/mmvq.cu`.

Rationale (from profiling): PQ2_0-class formats use 128-element blocks, so a decode MUL_MAT_ID row
has only K/128 K-blocks (4 for the down projection, 16 for gate/up). The generic 4-warp geometry
has 32 thread-groups per block, leaving 50-87% of thread groups idle, and each output-row block
re-reads the full Q8 activation row.

Change details:
- In the has_ids && ncols_dst==1 launch path, for Q1_0/Q2_0/PQ2_0 on the GENERIC table (pre-Volta
  NVIDIA) with moe_blocks_per_row <= 16, launch the previously unused (small_k=true, halve_iters=true)
  template combo: one warp (32 threads) per block, 4 rows per block. This fills the K-block groups
  and amortizes the activation-row reads over 4 rows.
- Verified variants (tg128, -r 3):
  - rpb=4, bprx<=16: 39.60 +/- 0.09  (kept)
  - rpb=8, bprx<=16: 39.00 +/- 0.07  (reverted)
  - rpb=4, bprx<16 (down only): 38.60 +/- 0.13  (reverted)

Reverted: temporary GGML_CUDA_NODE_TIMES per-op profiling instrumentation in ggml-cuda.cu
(restored to HEAD; marked remove-before-upstreaming while it existed).

## Final results (task 4)

Scion-35B-A3B-PQ2_0-corr.gguf, GTX TITAN X, full offload (ngl 99), -p 512 -n 128 -r 3, build 3a25590:

| test   | baseline      | final         | delta   |
|--------|---------------|---------------|---------|
| pp512  | 267.94 +/- 1.52 | 268.18 +/- 1.09 | ~0    |
| tg128  |  37.62 +/- 0.09 |  39.29 +/- 0.17 (best variant 39.60 +/- 0.09) | +4.4% (best +5.3%) |

Output sanity: verified via profile-run generation earlier in the session; benchmark verdict from
llama-bench only (llama-cli interactive runs hang under non-TTY stdin in this environment).

## Prompt processing (pp) profiling - round 2

pp512 = 268 t/s = 1.91 s per 512-token chunk (instrumented run: 2.38 s with graphs off + event overhead).
Per-op breakdown of one 512-token chunk (2382 ms kernel time, 25th graph execution, GGML_CUDA_NODE_TIMES
instrumentation re-added to ggml-cuda.cu - REMOVE BEFORE FINISHING):

| op / shape                      | ms   | %    | nodes |
|---------------------------------|------|------|-------|
| MUL_MAT_ID [2048x512x256] gate/up | 888  | 37.3 | 80  |
| FLASH_ATTN_EXT                  | 531  | 22.3 | 10 (all attn layers) |
| MUL_MAT_ID [512x2048x256] down  | 459  | 19.3 | 40  |
| MUL_MAT [2048x8192] qkv Q8_0    | 191  |  8.0 | 80  |
| MUL_MAT [4096x2048]             | 122  |  5.1 | 80  |
| MUL_MAT [2048x4096]             |  73  |  3.1 | 60  |
| rest                            | 118  |  5.0 |     |

MUL_MAT_ID total = 57% of pp. FLOP efficiency:
- Q8_0 MMQ reference (qkv, 512x2048x8192): ~7 TFLOPS effective (~28% of dp4a peak)
- PQ2_0 MMQ (MoE gate/up, 8 experts/token, ~16 real tokens/expert): 11.1 ms/node -> 0.77 TFLOPS,
  ~9x worse than the same tiled MMQ kernel with Q8_0 weights

Analysis: batch-512 MUL_MAT_ID dispatches to the tiled MMQ kernel (PQ2_0 mmq_supported = true,
confirmed in mmq.cu should_use_mmq). Tile J is chosen adaptively from ncols_max (max tokens per
expert, ~30-45) while the average is ~16 tokens/expert, and the PQ2_0 tile-load/dequant path
(mmq-load-tiles.cuh mmq_load_tiles_pq2_0) is the likely inefficiency, not the tile shape alone.

Next steps: inspect mmq_load_tiles_pq2_0 and the MMQ vec_dot for PQ2_0; candidates are dequant
cost in the shared-tile load and suboptimal tile J for small tokens-per-expert. FLASH_ATTN_EXT
(22%) is the second target (FA kernels poorly tuned for Maxwell, ~1 TFLOPS fp32).

## Round-2 findings + handoff (task-1 complete)

Model FA geometry: 16 Q heads, 2 KV heads, head dim 256 (D=256), gqa_ratio = 8.

MUL_MAT_ID batch-512 path (57% of pp):
- Dispatch confirmed: batch 512 -> tiled MMQ (PQ2_0 mmq_supported = true); J selected adaptively
  from ncols_max (= ne12 = 256 in mmq.cu mmq_args), but the kernel exits tiles beyond col_diff
  and bounds real work by tile_y_max_j, so tile waste is minimal - tile J is NOT the problem.
- The inefficiency is structural: each (expert, 128-row tile) block loads + dequants a
  128x2048 PQ2_0 weight tile (mmq_load_tiles_pq2_0, 2x byte_perm per int16 + 2x shared writes
  vs a plain Q8_0 copy) but multiplies it against only ~16 real token-columns
  (512 tokens x 8 experts / 256 experts). Load/dequant overhead per MAC is ~4x the dense
  Q8_0 qkv GEMM, plus the heavier PQ2_0 dequant -> ~9x total gap (0.77 vs ~7 TFLOPS).
- Fix requires a new kernel: MMVQ-style vec batching over the compact rows (ids_dst),
  keeping weight slices in registers/L2 across the ~16 assigned tokens. ~200 lines, not
  attempted in this session (insufficient remaining context). Diagnostic alternative:
  Q8_0-expert model copy - llama-quantize refuses requantizing from PQ2_0 (disabled by design).

FLASH_ATTN_EXT (22% of pp):
- D=256 tile kernel, GQA batching already active (gqa_ratio 8 > 4 -> gqa_limit = INT_MAX,
  ncols1=8), so KV tile loads are already amortized 8x. Still only ~0.13 TFLOPS
  (6.6 GFLOP/layer in 52 ms) - the D=256 pre-tensor-core tile kernel itself needs tuning
  (nbatch_fa/nbatch_K config or kernel rework). Not attempted (insufficient remaining context).

State at handoff: instrumentation re-added to ggml-cuda.cu then REVERTED to HEAD; kernel tree
identical to the end of round 1 (mmvq.cu decode-MoE fix kept, tg128 39.3-39.6, pp512 268).
All numbers in /tmp/prof_pp.log (raw) and this file (analysis).

## Round 3 - pp512 optimization attempts (2026-10-11, end of session)

Goal: make pp512 (268 t/s) faster by optimizing the round-2 targets
(MUL_MAT_ID PQ2_0 MMQ batch path 57%, FLASH_ATTN_EXT 22%).

### Attempted / benchmarked (all llama-bench -ngl 99 -p 512 -n 128 -r 3)

1. MMQ dead-column bound (task-2). mmq-vec-dot.cuh: added a runtime j_max param to
   ggml_cuda_mmq_vec_dot_t (all 16 vec_dot impls + 4 call sites) bounding the vec_dot j-loop
   (and the non-async y-tile load) by tile_y_max_j, so experts holding only ~16 real
   token-cols no longer compute/write the 4x-dead J=64 columns. Plus occupancy 2->4 for the
   PQ2_0 Pascal MMQ config. Result: pp512 = 267.38 ± 2.32 (j-bound only) then 269.39 ± 0.97
   (+occ), vs 268.18 baseline. FLAT -> reverted (per keep-if-faster contract).

2. FLASH_ATTN_EXT Maxwell tile occupancy (task-3). fattn-tile.cuh nvidia_fp32 config: D=256
   block occupancy 2-3 -> 4 (0.13 TFLOPS / low-occupancy finding). Result: pp512 = 267.69 ± 1.45.
   FLAT -> reverted.

### Root-cause finding (the important result)

pp512 (~268 t/s) is INVARIANT to every GPU-side change tried across the whole session:
  * MMQ MoE dead-MAC reduction 4x  -> flat (267.4)
  * MMQ occupancy 2->4             -> flat (269.4)
  * FA tile occupancy 2->4         -> flat (267.7)
  * round-1 A/B: FA on/off, KV q8_0, FORCE_MMQ, FORCE_CUBLAS, threads 4-8 -> all flat (262-269)
During a sustained pp run nvidia-smi reports GPU util 95-96%, power ~150 W, clocks ~1.0 GHz
=> the GPU IS busy; pp is NOT CPU/launch-bound and NOT bandwidth-bound (~1% BW util).

Conclusion: the round-2 attribution (MUL_MAT_ID 57% @ 0.77 TFLOPS, FA 22%) came from the
GGML_CUDA_NODE_TIMES graphs-OFF profile, whose per-node times are launch/serialization-inflated
(graphs OFF = 2403 ms/chunk vs graphs ON 977 ms = 268 t/s) and therefore do NOT reflect the real
graphs-ON pp512 bottleneck. The designated kernel targets are NOT what limits pp512. pp512 is
bound by a fixed invariant cost (likely the sequential GDN/SSM recurrence and/or graph/serial
kernel-latency structure, or VRAM pressure: 11.3 GB model on 12 GB card) that none of the
matmul/attention/thread knobs touch. A genuine pp fix needs either (a) profiling with graphs ON
or ncu, or (b) optimizing the GDN/SSM custom recurrence + graph fusion -- both beyond the
remaining budget of this run.

### Final state / result

All mmq/FA experiments reverted to HEAD. The only kernel change kept in the working tree is the
round-1 decode-MoE mmvq.cu geometry fix (tg128 37.62 -> 39.3-39.6, ~+5%%). pp512 remains ~268 t/s
(prompt-processing speed is unchanged and resistant to all kernel-level optimization attempted).

---

## Round 4 - valid profiling (2026-10-12)

### Measurement methodology (built first, because rounds 1-3 measured noise)

`/tmp/opencode/titanx-bench.sh` - contention-aware `llama-bench` wrapper:

- refuses to start when a foreign compute process holds the GPU,
- samples `nvidia-smi` every 0.5s for the whole run and fails the run if a foreign pid appears,
- excludes our own bench pid *by explicit pid*, plus a desktop whitelist (`kwin_wayland` holds
  8 MiB at 0% util and is not contention) and a 64 MiB size floor,
- rejects runs whose reported spread exceeds 5%,
- prints the sampled power/clock envelope and the start/end temperature.

`/tmp/opencode/ab.sh` - interleaved A/B runner. **Thermal drift on this card is ~2%** (tg256
decays 36.78 -> 36.10 t/s as the card warms 70C -> 78C), which is larger than most candidate
kernel wins. Measuring variant A then variant B minutes apart is therefore unsound; `ab.sh`
interleaves variants round-robin and reports medians so every variant sees a comparable
temperature.

The GPU *is* shared: during this round a foreign `bp_harness` process (another agent/session)
appeared mid-run and was correctly rejected. One `llama-cli` of our own was also left holding
11 GB of VRAM after being interrupted (note: `llama-cli` does not exit on its own here).

### Profiling method

`ncu` is not installed and cannot be installed (no passwordless sudo). `nvprof` works on sm_52
but has two gotchas: it writes the trace to **stderr** (stdout stays nearly empty), and
`--print-summary` fails with a bogus "application not found", so `--print-gpu-trace` is
aggregated locally by `/tmp/opencode/agg_nvprof.py`. Profile isolation is done by differencing a
`-r 1` trace against a `-r 4` trace and dividing by the extra repeats, which cancels model load.

Caveat: nvprof breaks CUDA graph capture, so absolute kernel sums under-count wall time. The
*relative* attribution is still sound, and for pp it is corroborated (see below).

### Corrected baseline (clean runs, same build 3a25590)

| test | round-4 measurements |
|---|---|
| pp512 | 251-266 t/s depending on card temperature |
| tg128 | 36.0-37.7 t/s |

pp512 is no longer reliably "268 +/- 1"; earlier figures were measured on a colder card.

### pp512 kernel breakdown - the previous attribution was WRONG

Rounds 2-3 attributed pp512 to MUL_MAT_ID MoE GEMVs (57%) and FLASH_ATTN_EXT (22%). The nvprof
profile says otherwise. Kernel time per 512-token chunk = **1824.6 ms**, which matches the ~1.92 s
wall time, so this profile is complete:

| kernel | ms | % | launches |
|---|---:|---:|---:|
| sgemm_32x32x32_NT_vec (cuBLAS) | 414.2 | 22.7 | 6096 |
| sgemm_128x128x8_NT_vec (cuBLAS) | 373.4 | 20.5 | 237 |
| dequantize_block<int=128> | 259.4 | 14.2 | 6719 |
| gated_delta_net_cuda | 140.6 | 7.7 | 30 |
| maxwell_sgemm_128x64_tn | 105.0 | 5.8 | 523 |
| dequantize_block<int=32> | 47.9 | 2.6 | 307 |
| k_get_rows_float_vec (lm_head gather) | 44.2 | 2.4 | 234 |
| mul_mat_vec_q<PQ2_0> (all variants) | 231.8 | 12.7 | ~10000 |
| quantize_q8_1 | 29.9 | 1.6 | 10085 |
| flash_attn_tile | 24.4 | **1.3** | 10 |

- **FLASH_ATTN_EXT is 1.3%, not 22%.** Rounds 2-3 chased a non-bottleneck.
- **FP32 cuBLAS SGEMM + weight dequantization is ~68%** of pp512.
- **Zero `mul_mat_q` kernels run.** MMQ is never used on sm_52.

Why MMQ is off (and why the round-1 `GGML_CUDA_FORCE_MMQ` A/B was a no-op):
`should_use_mmq()` reaches `return !fp16_mma_hardware_available(cc) || ...` only *after* an
earlier `if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) return false;`. On an
sm_52-only build, highest_compiled_arch is 52 < 610, so it always returns false first -
`GGML_CUDA_FORCE_MMQ` is checked after that gate and can never be reached.

Attempted fix: lowering `GGML_CUDA_CC_DP4A` 610 -> 500 so `__dp4a` is emitted and MMQ becomes
selectable. **This does not build** - `__dp4a` is declared in `sm_61_intrinsics.h` under
`__CUDA_ARCH__ >= 610`, and inline PTX `dp4a` is rejected by ptxas for sm_52. Maxwell has **no
DP4A at all**; the 610 constant is correct and MMQ is genuinely unavailable on this GPU. Reverted.

Also confirmed: `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16` aborts inside
`ggml_cuda_mul_mat_cublas_impl<F16>` - cuBLAS FP16 compute is unusable here, so FP32 is
structurally required and the dequantize->SGEMM path is not obviously avoidable by compute-type
tuning.

### Confirmed-dead knobs (no rebuild needed, all measured)

| knob | result |
|---|---|
| `-ub 128 / 256` | pp512 175.1 / 219.7 t/s - much worse than 512 |
| `-ub 512 / 1024 / 2048` | pp512 261.9 / 261.7 / 259.9 - flat, 512 already optimal |
| `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` | pp512 258.1, tg128 37.2 - baseline, no change |
| `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16` | **crashes** (no fast FP16 on sm_52) |
| `GGML_CUDA_MMVQ_PQ2_0_NWARPS=1` vs default | 35.53 vs 35.67 tg256 - flat within drift |
| `GGML_CUDA_MMVQ_PQ2_0_NWARPS=2` | 37.28 tg256 - slightly worse |

### Decode breakdown (nvprof, relative attribution)

| kernel | share of decode kernel time | launches | avg |
|---|---:|---:|---:|
| `mul_mat_vec_q<Q8_0>` ncols=1 | **49.9%** | 34688 | 92 us |
| `mul_mat_vec_q<PQ2_0>` ncols=1 | 14.6% | 15360 | 121 us |
| `quantize_q8_1` | 4.9% | 55168 | 55 us |
| `gated_delta_net_cuda` | 3.8% | 3840 | 128 us |
| `mul_mat_vec_f` | 3.6% | 10240 | 111 us |
| `rms_norm_f32` | 2.5% + 1.3% | 10368 | 78 us |

Q8_0 GEMV time by grid (= rows):

| grid | total ms | launches | avg us | identified shape |
|---|---:|---:|---:|---|
| 8192 | 2186 | 20520 | 107 | attn qkv / ffn projections |
| 248320 | 1581 | 513 | 3081 | **lm_head [248320 x 2048]** |
| 2048 | 983 | 20520 | 48 | small projections |
| 4096 | 885 | 15390 | 57 | small projections |
| 512 | 566 | 51300 | 11 | tiny (latency bound) |
| 32 | 119 | 30780 | 4 | tiny (latency bound) |

The lm_head GEMV is the single most expensive kernel in decode at 3081 us/token. It is Q8_0
(ggml_type=8), 248320 x 2048 = 510 MB read per token; at 3081 us that is ~166 GB/s, against
this card's ~336 GB/s peak. **That is the largest remaining decode target.**

### lm_head headroom measured (isolated probe)

`/tmp/opencode/lmhead_probe.cpp` drives ggml's own `mul_mat` on the exact lm_head shape, so the
166 GB/s from the in-model profile can be separated from kernel inefficiency:

| shape (m x k, ncols=1, q8_0) | ms | GB/s |
|---|---:|---:|
| 248320 x 2048 (lm_head) | 2.53-2.74 | **175-189** |
| 131072 x 2048 | 1.31 | 193 |
| 248320 x 4096 | 4.71 | 203 |
| 8192 x 2048 | 0.094 | 169 |
| 2048 x 8192 | 0.091 | 174 |

Hardware ceiling measured with a plain float4 streaming read (`/tmp/opencode/bwprobe2.cu`,
256 MB buffer): **265 GB/s**, i.e. 79% of the 336.5 GB/s theoretical peak (24 SMs, 384-bit bus,
3505 MHz effective). Card reports 5120 CUDA cores / 24 SMs.

So lm_head at ~190 GB/s is at **~72% of achievable bandwidth**, and the whole remaining gap is
worth ~0.8 ms/token, or **~3% of decode**. That is a real but small win, and it requires a new
GEMV geometry, not a table tweak. Lower priority than it first appeared.

Note the probe prints "disabling CUDA graphs due to GPU architecture" for a single-node graph, so
its numbers are launch-inclusive - fine for a relative ceiling, which is all it is used for.

### Decode: realistic ceiling

Decode is ~39 t/s = ~25.6 ms/token. Splitting that by the profile: Q8_0 MMVQ ~12.8 ms,
PQ2_0 MMVQ ~3.7 ms, everything else ~9 ms. Because the Q8_0 MMVQ time is spread over five
distinct shapes (mostly small ones at 48-107 us, i.e. latency-bound rather than
bandwidth-bound), a single-shape bandwidth fix cannot move decode much. The realistic
remaining decode wins are:
- lm_head geometry: ~3% (measured ceiling above),
- the 51300-launch grid-512 shape (11 us each, 100 launches/token): pure launch/latency
  overhead, worth attacking only via fusion, not geometry,
- reducing the `quantize_q8_1` cost (4.9%, 55168 launches - 431 per token).

### Known-bogus lead (do not repeat)

Lowering `GGML_CUDA_CC_DP4A` to 500 to unlock MMQ on Maxwell: **impossible, Maxwell has no
DP4A.** `__dp4a` is declared in `/opt/cuda/include/sm_61_intrinsics.h` under
`#if ... __CUDA_ARCH__ >= 610`, and inline PTX `dp4a.s32.s32.s32` is rejected by ptxas for
sm_52 ("Unexpected instruction types specified for 'dp4a'"). Any quantized-int8 GEMM path
(MMQ, and by extension anything built on dp4a) is unavailable on this GPU.

### State at this handoff

- Working tree: only the round-1 `mmvq.cu` decode-MoE fix (unchanged). The dp4a experiment is
  reverted; `git diff --stat` shows mmvq.cu only.
- No pp speedup landed. The reason is now understood and documented: on sm_52 the quantized
  GEMM paths (MMQ needs DP4A, MMF is fp16-only, cuBLAS FP16 compute aborts) leave dequantize +
  FP32 SGEMM as the only route, and it accounts for ~68% of pp.

---

## Round 4 remaining work, ranked

1. **quantize_q8_1 (4.9% of decode, 431 launches/token)** - the highest-value decode target
   that is not bandwidth-bound. Reducing activation-quantization launches (e.g. fusing the
   quantizer into the producing kernel) is a real but moderate win.
2. **lm_head GEMV geometry (~3% of decode)** - 190 -> 265 GB/s would need a different
   load/partition scheme for a 248320-row ncols=1 GEMV, not a table tweak.
3. **grid-512 shape (100 launches/token, 11 us each)** - latency-bound; needs fusion, and
   the CUDA-graph capture path may already hide much of it.
4. **PP512** - no kernel-level win available without DP4A. The only real levers are
   hardware-level (the 150W cap) or a different quant format with an sm_52-capable kernel.

## Round 4 summary

No pp speedup was landed. The reason is now understood rather than guessed: **Maxwell has no
DP4A**, so every quantized-int8 GEMM path is unavailable, cuBLAS FP16 compute aborts, and
pp512 is left with dequantize + FP32 SGEMM (~68% of the chunk). Rounds 2-3 spent their budget
chasing FLASH_ATTN (really 1.3% of pp) and MoE MMQ geometry (a path that never runs) because
the only profiler available then measured with CUDA graphs disabled.

The durable output of this round is the measurement tooling and the corrected attribution:

- `/tmp/opencode/titanx-bench.sh` - contention-, variance- and thermal-aware single runs
- `/tmp/opencode/ab.sh` - interleaved A/B, the only sound way to compare on this card
- `/tmp/opencode/agg_nvprof.py` - nvprof trace aggregator with repeat differencing
- `/tmp/opencode/lmhead_probe.cpp` - isolated GEMV probe for shape-level ceilings
- `/tmp/opencode/bwprobe2.cu` - raw streaming-read bandwidth ceiling (265 GB/s)

---

## Round 5 - quantize_q8_1 decode fusion (2026-10-12)

Target from round 4: `quantize_q8_1` = 4.9% of decode kernel time, 55168 launches (~431/token,
2.8 us each). Not bandwidth-bound (8 KB moved per launch), so it looked fusable.

### The redundancy is real

Parsed the real decode graph (`/tmp/opencode/graph_sharing.py` over `test-export-graph-ops`):

    decode (ncols==1): 14 MUL_MAT sigs, 4 distinct src1, 10 redundant (71%)

    src1 ne0=2048 ne1=1 -> 9 consumers   <- the big one
    src1 ne0=4096 ne1=1 -> 2 consumers
    src1 ne0= 512 ne1=1 -> 2 consumers
    src1 ne0= 256 ne1=1 -> 1 consumer

One 2048-wide activation feeds **9** separate projections per layer, and each mat-vec re-quantized
it. The graph parser reports unique signatures, so these repeat per layer (30 layers).

### Change (ggml/src/ggml-cuda/mmvq.cu)

Reuses the existing per-graph q8_1 cache (`ggml_cuda_fwht_q8_context`, reset once per graph
evaluation at ggml-cuda.cu:4520) instead of only the FWHT producer. On a miss, the quantized
rows are registered for the rest of the graph; later mat-vecs with the same src1 skip the
quantize. Gated by `GGML_CUDA_ACT_Q8_CACHE`, **default off**.

Two bugs found and fixed while implementing:

1. **VMM pool LIFO assert** (`GGML_ASSERT(ptr == pool_addr + pool_used)`). The first version
   allocated the cache block *after* a short-lived `src1_q8_1` and then freed the older one -
   out of order. Fixed by allocating exactly once per branch.
2. **MUL_MAT_ID crash.** The fallback path (`ggml_cuda_mul_mat_id`, ggml-cuda.cu:2030) builds a
   **stack-local `ggml_tensor src1_slice` per expert**. Its address repeats every iteration while
   the rows differ, so caching on the tensor pointer both corrupts results and exhausts the pool.
   Discriminator: those slices are `memset` to 0 and never assigned an `op`, so
   `src1->op != GGML_OP_NONE` cleanly excludes them while keeping every real graph tensor.

### Correctness

- `test-backend-ops test -o MUL_MAT` and `-o MUL_MAT_ID` pass with the cache on.
- Greedy generation byte-identical cache on vs off (`/tmp/opencode/check_output.sh`,
  `--temp 0 --seed 1234`, 48 tokens, both emitting the same text).

### It works, and it is still not a win

| | launches | quantize ms (64 tokens) |
|---|---:|---:|
| cache off | 28015 | 77.7 |
| cache on | 18265 | 55.6 |

-35% launches, -28% quantize kernel time. But end-to-end decode does not move:

| tg512, 5 interleaved rounds | r1 | r2 | r3 | r4 | r5 |
|---|---:|---:|---:|---:|---:|
| cache off | 36.77 | 35.87 | 35.45 | 35.26 | 35.27 |
| cache on | 36.23 | 35.41 | 35.29 | 35.14 | 35.22 |

In the thermally-equilibrated rounds 3-5 the cache is consistently ~0.2% *slower*.

**Why:** with CUDA graphs enabled, these launches are graph nodes, so the removed launches cost
almost nothing - the win has to come from the kernel time, and 22 ms saved over ~1730 ms of
decode is ~1.3%, under the ~2% thermal drift on this card. The retained pool bookkeeping costs
slightly more than it saves. **This would pay off with CUDA graphs off; with graphs on it does not.**

**Decision: default off**, per the keep-only-if-measurably-faster contract. `GGML_CUDA_ACT_Q8_CACHE=1`
re-enables it.

### Correction to round 4's ranking

Round 4 listed quantize_q8_1 as the top remaining decode target on the strength of its 4.9%
kernel-time share. That ranking was wrong for the same class of reason as the round-2/3
attribution error: under CUDA graphs the kernel-time share does not translate into removable
wall-clock. The true constraint in decode is that most of it is spread across many small,
latency-bound kernels rather than one dominant bandwidth-bound one.

## Round 6 - external patch port and evaluation (2026-10-13)

Sources evaluated: shinbunbun/llama-cpp-p100-patches (45 patches, upstream v0.5.0) and
Mrinohk/qwen36-p100-patches (31 patches). Full classification with gate verification lives in
TITANX-PATCH-SURVEY.md. Patches were ported by hand where they conflict with this diverged fork;
every verdict below is an interleaved A/B measurement (llama-bench or llama-server decode) on this
card, not a port of the authors' P100 numbers.

### Baseline re-take

Fresh llama-bench on build 3a25590 (round-1 mmvq decode-MoE fix in place):

| test  | t/s             |
|-------|-----------------|
| pp512 | 269.45 +/- 1.43 |
| tg128 |  39.56 +/- 0.12 |

Matches the recorded reference (tg128 39.3, pp512 268) within noise.

### Per-patch verdicts

| patch | verdict | measurement |
|-------|---------|-------------|
| shinbunbun 02 (mmvq rows-per-block) | KEEP | +10.7% tg128 (44.07 vs 39.80, 6 rounds) |
| shinbunbun 11 (penalties-direct) | KEEP | server decode +7.3% (with 13, 33.76 vs 31.47 geomean) |
| shinbunbun 13 (sampler-prefilter) | KEEP | see 11; byte-identical output at temp 0 |
| shinbunbun 16 (cpy-fastdiv) | KEEP | with 14+04: +0.9% tg, +0.55% pp |
| shinbunbun 14 (getrows narrow rows) | KEEP | see 16 |
| shinbunbun 04 (concat non-cont flat) | KEEP | see 16 |
| shinbunbun 17 (norm register cache) | KEEP | +0.7% tg / +0.66% pp (drop-attribution) |
| shinbunbun 18 (fuse sibling nodes) | KEEP | +0.5% tg (drop-attribution) |
| shinbunbun 21 (sched-reset-lazy) | REJECT | -0.1% tg (39.87 vs 39.91) - P100 +0.94% does not reproduce |
| shinbunbun 22 (decode-sched-slots) | DEFERRED | 5/21 hunks fail on the simpler local process_ubatch; bounded ~1%, needs +72 MiB VRAM |
| shinbunbun 07 (fused MoE matvec 2->4) | REJECT | identical at 44.16 with/without (on top of 02) |
| shinbunbun 26 (cpy-fused-rows) | REJECT | -0.8% tg (44.53 vs 44.91); per-row form does not fire for GDN conv-state copies |
| shinbunbun 28 (top-k partial) | N/A | fork baseline already uses CCCL DeviceTopK; gate ncols >= 4096 does not fire |
| shinbunbun 48 (fuse rms-norm-scale) | REJECT | dead code: Scion graph has no rms_norm->scale pairs (q/k are l2_norm, 2048-wide) |
| shinbunbun 40/29/30 (iq4_nl/iq quants) | N/A | tensor census: model has no IQ-family tensors |
| Mrinohk 0010 (staged-load Q8_0 matvec) | REJECT | -2.7% tg (43.70 vs 44.91) with GGML_CUDA_P60_MMVQ=1; smem staging loses on sm_52 |
| Mrinohk 0007 (GDN gated-norm fusion) | DEFERRED | pattern matches (build_norm_gated) but 15 hunks across 4 files overlap kept patch 17; P100 gain +0.4% |
| Mrinohk 0005 (GDN state-gather skip) | N/A | already implemented in this fork's baseline |
| Mrinohk 0006/0008/0012 | N/A | target rms_norm+scale / dense-qwen35 patterns the Scion graph does not build |
| Mrinohk 0001/0002/0011/0013/0014 (fp16) | not ported | bet on GP100 2x-rate HFMA2; GM200 HFMA2 runs at 1/32 of FP32 rate |
| Mrinohk 0018-0031 (persistent kernels) | not portable | hard-gated cc == 600, tuned for 56 SMs / 48 KB smem; GM200 has 24 SMs |
| shinbunbun 01 (vmad-dp4a-sm60) | not portable | __CUDA_ARCH__ >= 600 only; dead at compile on sm_52 |
| shinbunbun 12 (mmvq f16 sm60) | not portable | runtime gate excludes Maxwell (cc < PASCAL) |

### Final kept state

Kept (all measured-faster, on by default): 02, 04, 11, 13, 14, 16, 17, 18 across 10 files
(1013 insertions). The only kill switch among them: `LLAMA_SAMPLER_PREFILTER=0` disables 13.
Patch 18's fuse machinery additionally has `GGML_CUDA_DISABLE_FUSE_L2_NORM` /
`GGML_CUDA_DISABLE_FUSE_RMS_SCALE` diagnostics switches.

Definitive interleaved A/B of the full kept set vs true stock (round-1 mmvq fix only), 4 rounds
per side, 39-44 C:

| test  | kept set        | stock           | delta  |
|-------|-----------------|-----------------|--------|
| tg128 | 44.48 +/- 0.11  | 39.47 +/- 0.12  | +12.7% |
| pp512 | 266.53 +/- 0.87 | 268.04 +/- 1.43 | -0.6%  |

**Why:** the decode win is fully separated and large - the sampler host patches (11+13) remove a
larger fraction of a 39 t/s Maxwell decode than of the authors' P100 decode, and the mmvq
geometry change (02) plus the graph fusions stack on top. The pp512 -0.6% is borderline (distributions
overlap); the per-patch attributions measured 17 and the 16/14/04 trio each improving pp in
isolation, so the end-to-end pp cost is most likely interaction and thermal drift, but it is
recorded honestly here rather than averaged away.

**Decision:** keep the full set per the keep-only-if-measurably-faster contract - every kept
patch won its own measurement, and the +12.7% decode dominates the borderline pp read. Deferred
(22, Mrinohk 0007, patch 10) are documented in TITANX-PATCH-SURVEY.md with re-entry conditions.

### MTP drafter measurement (round-6 addendum)

Measured with the draft model included, on the final kept state. The drafter is
Scion-35B-A3B-mtp-drafter.gguf (50 MiB, single MTP layer, fits in VRAM alongside the 11.3 GiB
main model). llama-server auto-detects it as `draft-mtp-sidecar`; llama-bench has no draft
support, so measurement goes through llama-server with the same sampler chain as the round-6
server A/B (temp 0.7, top-k 20, top-p 0.8, repeat-penalty 1.1), 6 interleaved rounds per side:

| side   | geomean predicted t/s | range          |
|--------|-----------------------|----------------|
| draft  | 29.81                 | 29.59 - 29.97  |
| no draft | 37.17               | 36.95 - 37.36  |

-19.8%, distributions fully separated. Greedy (temp 0, same seed and prompt) tells the same
story: draft 31.21 vs no draft 37.32 (-16.4%).

**Why:** the drafter proposes ~1 token per step (mean draft length 1.19) and acceptance is far
too low to pay for it - 0.18 at temp 0.7 and still only 0.23 at greedy. Every decode step
pays the MTP head forward plus a full-model verify batch through the 35B graph, and this
card's decode is bound by many small latency-bound kernels, so the extra kernel count
dominates the ~20% of steps where a draft token is accepted. The drafter's single-layer
design cannot predict this model's MoE routing well enough.

**Decision:** do not use the MTP drafter on this hardware; the round-6 numbers above
(no draft) stand as the reference. A useful drafter would need acceptance near 0.7+ with
multiple tokens proposed per step, which this one does not approach even greedily.
