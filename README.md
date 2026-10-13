# H100 vLLM profiling results — Qwen3.8-27B (BF16)

Profiling results for decode performance of **Qwen/Qwen3.8-27B** (BF16) served with **vLLM 0.31.0** on one **NVIDIA H100 80GB HBM3** (GCP `a3-highgpu-1g`, Spot), captured 2026-10-09.

This repo holds results only. The notebooks and scripts that produce and analyze them will be added later.

## Nsight Systems traces

Open `.nsys-rep` files in the [Nsight Systems](https://developer.nvidia.com/nsight-systems) GUI (2025.1 or newer). The `.sqlite` files are the same traces exported for SQL / pandas analysis.

Each capture covers one prefill (512 input tokens) plus 64 decode steps, after warm-up (`--capture-range=cudaProfilerApi`, `--cuda-graph-trace=node`, `--trace=cuda,nvtx,osrt`).

| Capture | Batch | Mode | Report (download) | SQLite export | Summaries |
|---|---|---|---|---|---|
| b1_graphs  | 1  | CUDA graphs (production) | [b1_graphs.nsys-rep](https://github.com/tgarg01/h100-vllm-profiling-results/raw/main/runs/2026-10-09_qwen3.8-27b_bf16_initial/nsys/b1_graphs.nsys-rep) (9.1 MB) | [b1_graphs.sqlite](https://github.com/tgarg01/h100-vllm-profiling-results/raw/main/runs/2026-10-09_qwen3.8-27b_bf16_initial/nsys/b1_graphs.sqlite) (23 MB) | [kernels](runs/2026-10-09_qwen3.8-27b_bf16_initial/nsys/b1_graphs_cuda_gpu_kern_sum.csv) · [NVTX](runs/2026-10-09_qwen3.8-27b_bf16_initial/nsys/b1_graphs_nvtx_sum.csv) |
| b1_eager   | 1  | eager (`enforce_eager=True`) | [b1_eager.nsys-rep](https://github.com/tgarg01/h100-vllm-profiling-results/raw/main/runs/2026-10-09_qwen3.8-27b_bf16_initial/nsys/b1_eager.nsys-rep) (10.7 MB) | [b1_eager.sqlite](https://github.com/tgarg01/h100-vllm-profiling-results/raw/main/runs/2026-10-09_qwen3.8-27b_bf16_initial/nsys/b1_eager.sqlite) (29 MB) | [kernels](runs/2026-10-09_qwen3.8-27b_bf16_initial/nsys/b1_eager_cuda_gpu_kern_sum.csv) · [NVTX](runs/2026-10-09_qwen3.8-27b_bf16_initial/nsys/b1_eager_nvtx_sum.csv) |
| b32_graphs | 32 | CUDA graphs (production) | [b32_graphs.nsys-rep](https://github.com/tgarg01/h100-vllm-profiling-results/raw/main/runs/2026-10-09_qwen3.8-27b_bf16_initial/nsys/b32_graphs.nsys-rep) (9.5 MB) | [b32_graphs.sqlite](https://github.com/tgarg01/h100-vllm-profiling-results/raw/main/runs/2026-10-09_qwen3.8-27b_bf16_initial/nsys/b32_graphs.sqlite) (24 MB) | [kernels](runs/2026-10-09_qwen3.8-27b_bf16_initial/nsys/b32_graphs_cuda_gpu_kern_sum.csv) · [NVTX](runs/2026-10-09_qwen3.8-27b_bf16_initial/nsys/b32_graphs_nvtx_sum.csv) |
| b32_eager  | 32 | eager (`enforce_eager=True`) | [b32_eager.nsys-rep](https://github.com/tgarg01/h100-vllm-profiling-results/raw/main/runs/2026-10-09_qwen3.8-27b_bf16_initial/nsys/b32_eager.nsys-rep) (11.5 MB) | [b32_eager.sqlite](https://github.com/tgarg01/h100-vllm-profiling-results/raw/main/runs/2026-10-09_qwen3.8-27b_bf16_initial/nsys/b32_eager.sqlite) (30 MB) | [kernels](runs/2026-10-09_qwen3.8-27b_bf16_initial/nsys/b32_eager_cuda_gpu_kern_sum.csv) · [NVTX](runs/2026-10-09_qwen3.8-27b_bf16_initial/nsys/b32_eager_nvtx_sum.csv) |

NVTX ranges: `prefill_b{B}` = prefill only (max_tokens=1); `prefill_decode_b{B}` = prefill + 64 decode steps.

## Key findings

From [`analysis/findings.md`](runs/2026-10-09_qwen3.8-27b_bf16_initial/analysis/findings.md):

- B=1 decode step: 20.68 ms measured vs 15.40 ms predicted floor (1.34x); step-to-step CV 1.4%.
- GEMMs are 94% of decode kernel time and stream weights at 2.74 TB/s on average (82% of the 3.35 TB/s peak).
- GPU is busy 97.4% of decode; idle 539 µs per step, so host work is almost fully overlapped.
- Prefill (512 tokens): 45.0 ms of GPU time, GPU busy 82% (piecewise graphs leave GDN/attention ops eagerly launched).
- First prefill-only call: 230 ms wall vs 45 ms GPU, so 185 ms of host-side time (open question).
- Lowest-bandwidth GEMM: `gdn.in_proj_ba` at 0.21 TB/s (4.7 µs for 1.0 MB), latency-bound rather than bandwidth-bound.
- Eager vs graphs at B=1: 60.5 vs 20.7 ms per step (2.9x); GPU busy 38% vs 97%.
- B=32 step is 1.23x the B=1 step for 32x the tokens, so 26.0x throughput.

### Decode latency vs batch size (`vllm bench latency`, 512 input tokens)

Decode time per token = (latency(out=128) − latency(out=1)) / 127.

| Batch | Decode ms/step | Tokens/s | Predicted floor ms | Measured / predicted | Effective HBM TB/s |
|---:|---:|---:|---:|---:|---:|
| 1   | 20.55 | 49    | 15.40 | 1.33 | 2.51 |
| 8   | 21.94 | 365   | 16.15 | 1.36 | 2.47 |
| 32  | 25.19 | 1,270 | 18.72 | 1.35 | 2.49 |
| 64  | 29.67 | 2,157 | 22.15 | 1.34 | 2.50 |
| 128 | 59.46 | 2,153 | 29.00 | 2.05 | 1.63 |
| 256 | 97.82 | 2,617 | 42.71 | 2.29 | 1.46 |

Source: [`analysis/table1_baseline.csv`](runs/2026-10-09_qwen3.8-27b_bf16_initial/analysis/table1_baseline.csv). Raw bench JSONs are in [`bench/`](runs/2026-10-09_qwen3.8-27b_bf16_initial/bench).

## Setup

| | |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 (SXM), driver 580.178.04 |
| Software | vLLM 0.31.0, PyTorch 2.13.0+cu130, FlashInfer 0.7.0.post1, CUDA toolkit 12.9 |
| Profilers | Nsight Systems 2025.1.3, Nsight Compute 2025.2.0 |
| vLLM args | `language_model_only=True`, `max_model_len=8192`, `max_num_seqs=256`, `gpu_memory_utilization=0.92`, `enable_prefix_caching=False` |
| Weights loaded | 50.22 GiB (vLLM "Model loading took") |

Full settings: [`config.json`](runs/2026-10-09_qwen3.8-27b_bf16_initial/config.json).

## Layout

```
runs/2026-10-09_qwen3.8-27b_bf16_initial/
  config.json   settings the run used
  bench/        vllm bench latency JSONs (batch 1..256, out 1 / 128)
  nsys/         Nsight Systems reports, SQLite exports, kernel/NVTX summaries
  analysis/     figures, per-kernel / per-GEMM CSVs, findings.md
  ncu_plan/     ranked kernels + Nsight Compute commands for the follow-up session
  logs/         vLLM / nsys logs for every item
```

No Nsight Compute data yet: the first ncu attempt was preempted.
