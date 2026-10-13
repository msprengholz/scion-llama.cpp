# prism-ml-llama.cpp: `moe-corr-runtime` branch

This branch is the runtime for **Scion-35B-A3B** and for any other GGUF that
uses the Prism `PQ2_0` container or embedded LoRA adapters. It is the default
branch of this fork; `master` is an untouched upstream mirror and **cannot load
those files**.

Model: <https://huggingface.co/SkyIsNotGreen/Scion-35B-A3B>
Write-up and build scripts: <https://github.com/sky-is-green/scion>

## Build

```bash
git clone -b moe-corr-runtime https://github.com/sky-is-green/prism-ml-llama.cpp
cd prism-ml-llama.cpp
./verify-container-support.sh          # must print OK before you build
rm -rf build
cmake -B build -DGGML_CUDA=ON && cmake --build build -j --target llama-cli llama-server
```

- NVIDIA: `-DGGML_CUDA=ON`
- AMD: `-DGGML_HIP=ON` (tested on gfx1100, RX 7900 XT)
- CPU only: no flag

If you already have a checkout of this fork and the model will not load, delete
it and start from the clone above. A build of `master`, a stale `build/`
directory, or an older `llama-cli` earlier on your PATH all produce the same
error, and none of them are fixed by re-running cmake in place.

## If you see "invalid ggml type 142"

```
gguf_init_from_reader: tensor 'blk.0.ffn_down_exps.weight' has invalid ggml type 142. should be in [0, 43)
```

That is not a file problem, an OS problem or a hardware problem. It means the
binary you ran has never heard of the Prism container types:

| tree | `GGML_TYPE_COUNT` | `GGML_TYPE_PQ2_0` |
| --- | --- | --- |
| upstream `master` (also this fork's `master`) | 43 | absent |
| this branch | 144 | `= 142` |

`GGML_TYPE_COUNT` is the upper bound the error prints, so `[0, 43)` is a build of
upstream `master`. Check what you have:

```bash
grep -n "GGML_TYPE_PQ2_0" ggml/include/ggml.h   # expect: GGML_TYPE_PQ2_0 = 142
```

## If you see "unknown model architecture: 'mtp'"

```
llama_model_load: error loading model: unknown model architecture: 'mtp'
common_speculative_init_result: failed to load draft model, '...mtp-drafter.gguf'
```

The drafter is a target-context sidecar head, not a model, so it only loads
through the `draft-mtp-sidecar` path. The log above means the run selected a
draft-model type, most often `--spec-type draft-mtp`, which is the MTP head
inside a full model and a different feature. Run `-md Scion-35B-A3B-mtp-drafter.gguf`
with no `--spec-type`; the sidecar is detected from the GGUF. Current builds
replace an explicit draft-model type with `draft-mtp-sidecar` and print a warning.

## What this branch adds on top of Prism's llama.cpp

Roughly 90 lines, all in the runtime:

- the `ffn_moe_out` LoRA branch target (attention-output branches can use a
  standard llama.cpp LoRA; the MoE-block placement needs this virtual tensor)
- embedded adapters: `adapter.embedded=true` lets a correction sidecar travel
  inside the model file, so a corrected model loads with no `--lora` step
- LoRA hooks for the `olmoe` and `qwen35moe` graphs
- import shim for the legacy `Q1_0_g128` container (type id 43), byte-identical
  to `PQ2_0`, remapped at load time so old files need no repack

The container, the quantization rule and the kernels are Prism ML's work, not
mine. See `ggml/include/ggml.h` for the type ids and `ggml/src/gguf.cpp` for the
legacy import.

## TITAN X (Maxwell, sm_52) optimizations

This branch carries the round-6 CUDA optimization set, measured on a GTX TITAN X
(sm_52, 12 GiB) with the Scion model, tg128 39.5 -> 44.5 t/s (+12.7%) versus the
pre-optimization build. Full methodology, per-run numbers and the MTP-drafter
measurement: `TITANX-PROFILE-NOTES.md`. Per-patch verdicts for both P100 patch
repositories: `TITANX-PATCH-SURVEY.md`.

Kept changes, each measured faster by interleaved A/B on the TITAN X (all on by
default):

| source | change | measured effect |
| --- | --- | --- |
| fork round 1 | `mmvq.cu` decode MoE GEMV small-k rows fix | tg128 37.6 -> 39.3 t/s |
| shinbunbun 02 | mmvq rows-per-block for pre-Turing (ported to `calc_rows_per_block`) | +10.7% decode |
| shinbunbun 11 + 13 | sampler penalties-direct + prefilter (host) | +7.3% server decode; `LLAMA_SAMPLER_PREFILTER=0` disables 13 |
| shinbunbun 16/14/04 | cpy fastdiv, getrows narrow rows, concat non-cont flat | +0.9% decode, +0.6% pp |
| shinbunbun 17 + 18 | norm register cache, fuse sibling nodes | +1.2% decode |

Rejected with measurements (reverted, do not re-apply): shinbunbun 21, 07, 26, 48
and Mrinohk 0010 staged-load Q8_0 matvec (-2.7%). Not applicable to this model or
architecture: shinbunbun 28/29/30/40 (no IQ-family tensors, CCCL top-k already
present), the fp16 activation families and the sm_60 mega kernels (GM200 HFMA2
runs at 1/32 of the FP32 rate; mega kernels are hard-gated cc == 600).

KV cache: use q8_0 KV to double the context headroom; decode holds above
41 t/s through 65k tokens with q8_0 KV on the TITAN X (f16 KV caps out around
48k on 12 GiB). `--fit` (on by default) auto-sizes the context to what fits.

Recommended server start with recommended sampling, auto-fit context and
generation length allowed up to 16384 tokens:

```bash
./build/bin/llama-server -m Scion-35B-A3B-PQ2_0-corr.gguf -ngl 99 -fa 1 \
  -ctk q8_0 -ctv q8_0 -fitt 512 -fitc 4096 \
  --temp 0.6 --top-p 0.95 --top-k 20 --max-tokens 16384
```

Do not use the MTP drafter on the TITAN X: measured -19.8% (draft acceptance
0.18 at temp 0.7, 0.23 greedy).

## Container types in the Scion-35B-A3B release

The file is a mix, which is normal:

| type id | name | what |
| --- | --- | --- |
| 142 | `PQ2_0` | ternary expert banks (2-bit codes + one fp16 scale per 128) |
| 43 | legacy `Q1_0_g128` | the embedded rank-512 corrections, imported as `PQ2_0` |
| 8 | `Q8_0` | attention, embeddings, LM head |
| 0 | `F32` | norms, routers, output |

## Credits and license

MIT, following upstream: [llama.cpp](https://github.com/ggml-org/llama.cpp) by
the ggml-org and [Prism ML](https://github.com/PrismML-Eng/llama.cpp), with the
`Q1_0_g128` container conventions from
[CodeMasterCody3D](https://github.com/CodeMasterCody3D/prism-ml-llama.cpp).
Base weights for the release are
[empero-ai/Qwen3.8-35B-A3B-Distill](https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill)
and [Qwen](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). This fork is not
affiliated with or endorsed by any of them.
