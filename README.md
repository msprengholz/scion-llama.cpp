# scion-llama.cpp

Runtime for **Scion-35B-A3B** and any other GGUF using the Prism `PQ2_0`
container or embedded LoRA adapters. Fork of
[sky-is-green/prism-ml-llama.cpp](https://github.com/sky-is-green/prism-ml-llama.cpp);
the only difference is the TITAN X optimization set below. Everything else
(build, container types, troubleshooting) is documented in that repo.

Model: <https://huggingface.co/SkyIsNotGreen/Scion-35B-A3B>

## Build

```bash
git clone -b moe-corr-runtime https://github.com/msprensholz/scion-llama.cpp
cd scion-llama.cpp
./verify-container-support.sh          # must print OK before you build
rm -rf build
cmake -B build -DGGML_CUDA=ON && cmake --build build -j --target llama-cli llama-server
```

- NVIDIA: `-DGGML_CUDA=ON`
- AMD: `-DGGML_HIP=ON`
- CPU only: no flag

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

Recommended server start with recommended sampling, auto-fit context, 2
parallel slots, unified KV, reasoning preserved, generation length up to
16384 tokens:

```bash
./build/bin/llama-server \
  -m ~/.cache/huggingface/hub/models--SkyIsNotGreen--Scion-35B-A3B/snapshots/bb76b0075fe409a72959fce087d7455f88e5b79c/Scion-35B-A3B-PQ2_0-corr.gguf \
  -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -fitt 256 \
  --temp 0.6 --top-p 0.95 --top-k 20 \
  -n 16384 -np 2 -kvu --reasoning-preserve \
  --host 0.0.0.0 --port 9000
```

Do not use the MTP drafter on the TITAN X: measured -19.8% (draft acceptance
0.18 at temp 0.7, 0.23 greedy).

## License

MIT, following upstream llama.cpp and Prism ML. Base weights for the release
are empero-ai/Qwen3.8-35B-A3B-Distill and Qwen/Qwen3.6-35B-A3B.
