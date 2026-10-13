# Patch survey: P100 patch repos vs GTX TITAN X (sm_52)

Working doc for round 6. Both repos were developed and measured on Tesla P100
(GP100, sm_60). Target here: GeForce GTX TITAN X (GM200, sm_52, 12 GiB), build
`moe-corr-runtime-b10777-3a25590`, Scion-35B-A3B-PQ2_0-corr (arch qwen35moe,
custom ternary quant, no MTP head), full GPU offload.

Key facts that decide applicability:

- Maxwell sm_52 has no DP4A (`GGML_CUDA_CC_DP4A = 610`), so every gate of the
  form `cc >= GGML_CUDA_CC_DP4A` treats the TITAN X like a P100. The local
  build is compiled only for sm_52, so `ggml_cuda_highest_compiled_arch`
  routes every card to the "pascal_older" tables - patches scoped "no DP4A"
  fire here directly.
- Maxwell is `pre-Volta` and `pre-Turing`: runtime cc gates of `cc < VOLTA`
  / `cc < TURING` also fire here.
- Maxwell (GM200) does NOT have GP100's full-rate fp16. HFMA2 runs at ~1/32
  of fp32 rate, so every fp16-arithmetic kernel tuned for GP100 is expected
  to lose on this card. Patches whose gate already excludes Maxwell are
  correctly excluded; patches gated only on "no DP4A" that use fp16 math
  compile and run but are expected to measure slower (test, do not assume).
- The local tree diverges from upstream v0.5.0 (custom qwen35moe arch,
  PQ2_0-corr storage, FWHT q8 context, round-1 mmvq.cu MoE GEMV fix), so
  `patch -p1 -F0` is expected to fail on touched files; every applied patch
  needs a manual port.
- Decode baseline: 39.3 t/s tg128 (kernel-bound, many small latency-bound
  kernels under CUDA graphs). pp512: 268 t/s (dequantize + FP32 SGEMM, no
  DP4A). No MTP: single-token decode is the hot path, verify-batch patches
  matter less.

## shinbunbun/llama-cpp-p100-patches (45 patches)

### Applies - port and A/B

| # | Patch | Scope | Why it applies |
|---|---|---|---|
| 02 | mmvq-rows-per-block-sm60 | pre-Turing | runtime cc gate fires on sm_52; +23% tg32 on P100. Check hunk dependency on patch 01 |
| 05 | mmvf-f32-pascal | pre-Turing | raises F32 vector path cap ne11<=3 -> 8 for pre-Turing; helps batched decode (n>3) |
| 07 | mmvq-moe-rows-sm60 | all archs | hardcoded rows_per_block 2->4, ungated constant; check interaction with round-1 mmvq.cu fix |
| 08 | mmvq-mmid-batch-sm60 | pre-Volta | table keyed cc < VOLTA; Maxwell falls to sorted-gather past the cap like sm_60 |
| 09 | mmvq-nwarps-small-k-sm60 | pre-Turing | small-K warp sizing, +1.29% MoE decode on P100; not bit-identical where need=3 |
| 11 | penalties-direct | host | arch-independent, bit-identical, +5.3% on P100; host share grows as GPU-bound decode shrinks |
| 13 | sampler-prefilter | host | arch-independent, falls back safely |
| 21 | sched-reset-lazy | host | general llama.cpp improvement, bit-identical |
| 22 | decode-sched-slots | host | +0.97% but +72 MiB VRAM; model is 11.3 GiB of 12 GiB - test carefully, default 4 slots |
| 04 | concat-non-cont-flat | CUDA | generic kernel fix; fires on GDN conv-state concat (model is qwen35moe) |
| 10 | mmvq-q8-1-activation-cache | CUDA | generic, on by default; note local tree already has an ACT_Q8_CACHE (round 5, default off) - avoid duplication |
| 14 | getrows-narrow-rows | CUDA | generic launch-geometry fix |
| 16 | cpy-fastdiv | CUDA | generic, +0.82% |
| 17 | norm-register-cache | CUDA | l2_norm half expected to fire: prior profile shows L2_NORM nodes in local decode (custom arch still uses l2_norm) |
| 18 | fuse-sibling-nodes | CUDA | generic CPY/L2_NORM sibling fusion, +1.06% |
| 26 | cpy-fused-rows | CUDA | generic, 1.9x kernel |
| 29 | mmvq-iq3xxs-grid-smem | CUDA | bit-identical; only fires if IQ3_XXS tensors exist (PQ2_0-corr: verify tensor types) |
| 30 | mmvq-ksigns-smem | CUDA | bit-identical; same tensor-type caveat as 29 |
| 37 | getrows-narrow-batched | CUDA | off by default (env); generalizes 14 to batched gathers, -9.4% prefill on P100 |
| 40 | mmq-iq4-nl-threads | no DP4A | fires directly on sm_52-only builds; only relevant if IQ4_NL tensors exist (verify) |
| 48 | fuse-rms-norm-scale | CUDA | fires on RMS_NORM->SCALE runs; verify local arch still has these (if l2_norm, may not fire) |

### Test only if time permits / conditional

| # | Patch | Scope | Caveat |
|---|---|---|---|
| 19 | fuse-pre-add-rms-norm | CUDA | off by default, known read-under-write hazard on MoE graphs; full-GPU offload but MoE graph; measure with output check |
| 20 | fuse-add-unary-mul | CUDA | same hazard class as 19, off by default |
| 28 | top-k-partial | CUDA | CUDA 12.x lacks CCCL 3.2 DeviceTopK; local runtime is CUDA 13.0 - check whether CUB_TOP_K_AVAILABLE already covers it |
| 31 | fattn-f16-kv-chunk | pre-Turing | only pays with quantized KV at long context; local tests are short-ctx F16 KV |
| 38 | getrows-q4-0-block | CUDA | off by default; needs q4_0 KV gather workload (we use F16 KV) |
| 41 | top-k-radix-select | CUDA | off by default; for large-k sparse-attention indexer, N/A for this model |
| 35 | mmvq-mmid-batch-cap | CUDA | off by default, for resident expert cache + wide verify batches; no MTP here |
| 36 | mmvq-chunk-large-batch | CUDA | off by default, for quantized LM head + 8<ne11<=32 batches; no MTP here |
| 47 | sched-split-inputs-cap | host | off by default; for sparse-attention offloaded MoE at 262k ctx |

### Does not apply

| # | Patch | Reason |
|---|---|---|
| 01 | vmad-dp4a-sm60 | `#if __CUDA_ARCH__ >= 600` - dead code on sm_52 (Maxwell has no DP4A to replace) |
| 12 | mmvq-f16-sm60 | gate `cc < GGML_CUDA_CC_PASCAL` excludes Maxwell; also fp16 is 1/32-rate here, correctly gated out |
| 15 | mtp-draft-vocab | model has no MTP head |
| 32, 33 | sched prefetch | for experts in host memory; run is full GPU offload |
| 34 | mul-mat-id-negative-ids | enables 46 only |
| 39 | gdn-lanes-per-column | applies in principle (qwen35moe has GDN) but off by default, tuned S_v==128; verify head size, low priority |
| 42, 44, 45, 46 | qwen4exp-* | arch-specific (qwen4exp), not qwen35moe |

## Mrinohk/qwen36-p100-patches (31 patches, on top of shinbunbun)

### Applies - port and A/B

| # | Patch | Why it applies |
|---|---|---|
| 0002 | fused swiglu prep + token count policy | gate cc < DP4A fires on Maxwell; fusion itself is not fp16-dependent |
| 0005 | skip GDN state gather | GDN fusion, env-gated; model is qwen35moe |
| 0006 | GDN q/k norm + scale fusion | same |
| 0007 | GDN gated output norm fusion | same |
| 0008 | GDN alpha + dt bias fusion | same |
| 0009 | GDN conv-state shuffle folded into ssm_conv | same |
| 0012 | GDN node reorders for dense qwen | depends on 0003/0006 fusions |
| 0015, 0016 | cuBLAS ALGO6 choice | host-side algo selection; verify arch gate, cheap to A/B |

### Conditional - fp16 kernels gated cc < DP4A (fire on Maxwell, expected to lose)

| # | Patch | Caveat |
|---|---|---|
| 0001 | fp16 decode matvec + fused rms_norm prep | HFMA2 at ~1/32 rate on GM200; the fp16 matvec half expected to lose. The fused rms_norm prep (q8 activation prep fill) is fp16-independent - extract and test that half separately if the whole patch loses |
| 0013 | fp16 multi-column matvec | same expectation |
| 0003, 0004 | fold residual adds / write q8_1 from fused rms_norm | depend on the P60 mmvq path and fp16 activation prep; test after 0001 verdict |

### Depends on MMQ / persistent kernels

| # | Patch | Caveat |
|---|---|---|
| 0010, 0014 | staged-load Q8_0 / Q6_K matvec (bit-exact) | gate: no-DP4A + GENERIC table -> fires on Maxwell. Model weights are Q8_0-class: highest-interest candidate. Depends on 0001's harness |
| 0011 | MoE MMQ tiles by tokens per expert | MMQ path never runs in local MoE decode (round-2/3 finding); likely N/A |
| 0017 | MTP reduced draft vocab | no MTP head, N/A |
| 0018-0025 | persistent single/multi-token layer halves | hard gate `cc == 600`; tuned for 56 SMs / 48 KB smem / full-rate fp16. Cannot apply as-is; a port is a retune for 24 SMs / 96 KB smem / 1/32-rate fp16 - large effort, expected reject |
| 0020 | MTP verify output-layer + FA kernels | no MTP, N/A |
| 0026-0031 | pp fp16 expert GEMM / pp GDN / pp FA | fp16 grouped GEMM relies on full-rate fp16; expected to lose on Maxwell. pp has no kernel-level win without DP4A (round-4 finding) |

## Priority order for measurement

1. Host patches (11, 21, 13, 22): port as-is, A/B decode.
2. mmvq geometry (02, 07, 09, 08): decode MoE/matvec is the hot path.
3. Staged-load Q8_0 matvec (Mrinohk 0010): bit-exact, targets the Q8_0 weights this model actually uses.
4. Generic CUDA fixes (16, 14, 04, 26, 18, 17, 10, 29/30 if tensors match).
5. GDN fusions (Mrinohk 0005-0009, 0012).
6. mmvf cap (05) and conditional items.
7. fp16 candidates (0001, 0013): measure to confirm the 1/32-rate expectation, reject with numbers.
8. Persistent kernels (0018-0031): classify as cannot-apply-as-is; port only if time remains and the fp16 measurements justify it.

## Round-6 interim verdicts (2026-10-13)

Baseline re-take (task 2): pp512 269.45 +/- 1.43, tg128 39.56 +/- 0.12
(build 3a25590, GPU cooled to 30 C, -p 512 -n 128 -r 3). Matches the recorded
39.3/268 reference within noise.

- 21 sched-reset-lazy: REJECT. Interleaved A/B, tg128 -p 0 -n 128, 6 rounds
  per side, temps 36-40 C: patched mean 39.87 vs stock 39.91 t/s (-0.1%,
  within noise). The P100's +0.94% does not reproduce on this card; the
  scheduler reset is not a measurable term of the local decode.
- 22 decode-sched-slots: DEFERRED. Patch generated against upstream v0.5.0's
  two-arena gf_res_prev structure; the local tree still uses the simpler
  single-arena process_ubatch, so 5 of 21 hunks fail and the port rewrites
  core scheduling. Bounded expected value: the mechanism it builds on
  (patch 21) measured ~0 here because decode is GPU-kernel-bound. Port last
  if time remains.
- 11 penalties-direct + 13 sampler-prefilter: KEEP. Server decode A/B, 3
  fixed prompts x 256 tokens, chain penalties;top_k;top_p;temperature,
  4 interleaved rounds per side: patched geomean ~33.76 vs base ~31.47 t/s
  (+7.3%, distributions fully separated: min 33.60 vs max 31.56). Bigger
  than on P100 because the realistic sampler chain is a larger share of a
  39 t/s decode. Output byte-identical at temp 0, same seed. Port note:
  patch 13 hunk 1 (include <cstdlib>) failed; added manually.

Measurement tooling rebuilt for this round: /tmp/p100/ab.sh (interleaved
llama-bench A/B over swapped library sets), /tmp/p100/serverbench.sh
(server decode geomean over 3 prompts), build sets under /tmp/p100/ab/.

Round-6, task 4 (CUDA patches), continued:

- 02 mmvq-rows-per-block-sm60: KEEP (+10.7% tg128 decode). Ported onto the
  local calc_rows_per_block (preserving the round-1 tiny-K branch and the
  PTQ1_0 branch). Interleaved A/B vs stock, 6 rounds per side, 37-41 C:
  patched 43.96-44.17 (mean 44.07) vs stock 39.56-39.98 (mean 39.80).
  Distributions fully separated. The Q8_0 attn_qkv/lm_head/expert GEMVs all
  amortize activation-row reads across 4 output rows now.
- 07 mmvq-moe-rows-sm60: REJECT. Attribution A/B (02 only vs 02+07,
  4 rounds per side): 44.16 vs 44.16 - identical. The fused MoE matvec
  constant 2->4 does nothing here; the local expert GEMVs run the round-1
  custom path (small_k=true, 1 warp), not the launcher patch 07 retunes.
  Residue reverted to the upstream line.
- 29/30/40: N/A by tensor census - the Scion quant carries only Q8_0, F32,
  F16 and the fork's custom PTQ1_0/PQ2_0 ternary types; no IQ3_XXS, IQ2 or
  IQ4_NL tensors exist for these kernels to fire on.

Round-6, task 4 (CUDA patches), continued:

- 02 + 17 + 18 + 16 + 14 + 04 batch (on top of 11+13+02): interleaved A/B
  batch1 vs p0207, 4 rounds per side, 35-38 C:
  pp 267.2 vs 265.8 (+0.5%), tg 44.94 vs 44.14 (+1.8%, fully separated).
- Per-patch attribution (drop-one A/Bs vs batch1, 3 rounds per side):
  - 17 norm-register-cache: +0.7% tg / +0.66% pp. KEEP. Register-caches the
    norm row for 1024+ block sizes in both rms_norm and l2_norm kernels.
  - 18 fuse-sibling-nodes: +0.5% tg, pp neutral. KEEP. Fuses runs of
    identically shaped cpy nodes (GDN conv-state snapshots) into one launch.
  - 16+14+04 as a group (cpy-fastdiv, getrows-narrow-rows,
    concat-non-cont-flat): +0.9% tg / +0.55% pp. KEEP.
  Port notes: 04 hunk 2 (launch routing) ported manually into the diverged
  concat.cu (DGX_SPARK kernel present locally); 16+18 kernels are adjacent
  in cpy.cu and 18's fused kernel had to be re-inserted by hand after the
  16 revert; patch 18 depends on patch 16 (its fused kernel reuses the
  fastdiv machinery but compiles standalone).
- Running best: pp512 ~267 t/s, tg128 ~44.9 t/s (llama-bench), server
  decode ~+7.3% over stock samplers.

Round-6, task 4 (CUDA patches), continued:

- 26 cpy-fused-rows: REJECT. Ported both hunks onto the local ggml_cuda_cpy_fused
  (18's fused machinery present); interleaved A/B p26 vs batch1, 3 rounds per
  side: pp 267.07 vs 267.41 (neutral), tg 44.53 vs 44.91 (-0.8%, consistent).
  The per-row form does not fire for the local GDN conv-state copies. Reverted.
- 48 fuse-rms-norm-scale: REJECT, dead code for this model - verified in the
  graph sources: qwen35moe.cpp uses ggml_l2_norm for the q/k norms (lines
  462-463) and the only rms_norm in the Scion graph is the generic
  LLM_NORM_RMS branch followed by MUL with norm weights; every ggml_scale in
  the graph is fed by GDN delta-rule scalings or MoE gating, not by rms_norm.
  The fusion additionally only targets ncols < 1024 while the local norms are
  2048-wide. Not ported.

Round-6, task 4 (CUDA patches), continued:

- 28 top-k-partial: N/A - the local fork already uses CCCL DeviceTopK
  (DeviceTopK::MaxPairs) for all top-k (committed in the baseline top-k.cu),
  which is the same partial-selection mechanism patch 28 implements by hand;
  patch 28's gate (ncols >= 4096) also does not fire for the model's
  256-expert MoE gating. Not ported.
- 31 fattn-f16-kv-chunk: N/A - the KV cache is F16 and the benchmarks run at
  4k ctx; the patch bounds FA scratch at 81k+ ctx only.
- Mrinohk 0010 staged-load Q8_0 matvec: REJECT with measurement. Ported
  (kernel + opt-in gate, calc_rows_per_block call adapted to the local
  signature) and A/B'd with GGML_CUDA_P60_MMVQ=1, 4 rounds per side,
  38 C: tg128 43.70 vs 44.91 batch1 (-2.7%, consistent), pp neutral.
  The GP100 claim (+30% on the Q8_0 matvecs, +7.3% tg256) does not
  reproduce on sm_52: the smem staging costs more than the uncoalesced
  16-bit weight loads save on this card. First A/B without the env var was
  void (patch is opt-in). Reverted.
- 10 mmvq-q8-1-activation-cache: DEFERRED (not ported). The mechanism saves
  one q8_1 quantize per gate/up pair; bounded value here (~0.5% max, same
  class as the local round-5 ACT_Q8_CACHE which measured ~0.2% slower under
  CUDA graphs), and the port requires manual work in the diverged
  mul_mat_vec_q. Port only if time remains.

Round-6, task 5 (fp16 and sm_60-only kernels):

- shinbunbun 01-vmad-dp4a-sm60: dead code at compile time - the kernel body is
  __CUDA_ARCH__ >= 600 and the build compiles for sm_52 only. Not ported.
- shinbunbun 12-mmvq-f16-sm60: runtime gate (cc < GGML_CUDA_CC_PASCAL or
  cc >= GGML_CUDA_CC_DP4A excluded) fires only for Pascal/Volta; Maxwell
  sm_52 is below GGML_CUDA_CC_PASCAL so the gate excludes it. Not portable.
  (Its benefit would be full-rate HFMA2 anyway, which GM200 lacks: HFMA2 runs
  at 1/32 of the FP32 rate on GM200 per the profile notes, so fp16 matvec is
  architecturally dead here.)
- Mrinohk 0001/0002/0011/0013/0014 (fp16-activation family): gated
  cc < GGML_CUDA_CC_DP4A so they compile for Maxwell, but they all bet on
  GP100's 2x-rate HFMA2; GM200's 1/32-rate HFMA2 makes fp16 matvec a
  large loss by architecture (same established fact). Not ported.
- Mrinohk 0018-0031 (persistent/mega kernels): hard-gated cc == 600 and tuned
  for 56 SMs / 48 KB smem; GM200 has 24 SMs. Cannot apply.
- Mrinohk 0005 (GDN state-gather skip): already implemented locally - the fork
  baseline has ggml_cuda_try_gdn_gather_skip + ctx.gdn_gathers() in
  ggml-cuda.cu and the cache-row read in gated_delta_net.cu (verified in
  sources). Superseded, no port needed.
- Mrinohk 0006 (fuse GDN q/k rms_norm+scale): does not fire on the Scion
  graph - qwen35moe.cpp builds the q/k norms with ggml_l2_norm (lines
  462-463), not rms_norm; Mrinohk's Qwen3.6 graph used rms_norm+scale.
- Mrinohk 0008/0012 (alpha-projection bias fuse, dense qwen35 node reorders):
  0012 targets the dense qwen35 graph which the Scion does not use; 0008 is a
  MUL_MAT->RESHAPE->ADD matcher with +0.2% P100 - bounded value below the
  single-kernel noise floor, matcher machinery diverged locally. Not ported.
- Mrinohk 0007 (fuse GDN gated output norm rms_norm*w*silu(z)): the Scion
  graph HAS build_norm_gated (qwen35moe.cpp line 277, used at 487), so the
  pattern matches. P100 gain +0.4%. Ported and measured below.

- Mrinohk 0007 verdict (from the line above, updated): DEFERRED, not ported.
  The port spans 15 hunks across 4 files; norm.cu hunks rewrite the same
  rms_norm_f32 kernel the kept patch 17 (register cache) modifies, and the
  qwen35moe.cpp node-order change touches the model file. Gain is +0.4% on
  P100 MTP decode with 2 GPUs - below the single-kernel noise floor here,
  and the local fuse machinery (patch 18's matcher) diverged. Revisit only
  if the round-6 wins are merged first.

qwen35moe matcher compatibility check vs the Scion graph: 0005 already in
the fork baseline; 0006 does not fire (l2_norm, not rms_norm); 0007 matches
but deferred; 0008/0012 target patterns the Scion graph does not build.
