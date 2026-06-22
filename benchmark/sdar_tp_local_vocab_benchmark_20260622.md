# SDAR TP-Local Vocabulary State Benchmark

This note records a smoke benchmark for the dLLM TP-local vocabulary state path.
The optimized path avoids materializing tensor-parallel full-vocabulary logits for
`LowConfidence`; each TP rank computes local `(max, argmax, logsumexp)` state and
then merges only that compact state.

## Setup

- Date: 2026-06-22
- Pod: `p-ai-efficiency-tech/ms-qwen3-torchspec-2node`
- GPUs: 8x H100, split as baseline `CUDA_VISIBLE_DEVICES=0,1,2,3` and optimized
  `CUDA_VISIBLE_DEVICES=4,5,6,7`
- SGLang source: `/tmp/sglang-lab/src/sglang`
- Python: `/tmp/sglang-lab/venvs/dllm/bin/python`
- TP size: 4 per endpoint
- dLLM algorithm: `LowConfidence`
- Attention backend: `flashinfer`
- Sampling backend: `flashinfer`
- Max running requests: 32
- Benchmark shape: 128 measured requests, 32 warmup requests, 32 generated tokens,
  temperature 0, 3 repeats per concurrency
- Concurrency: 1, 4, 16, 32

The benchmark compared the default full-logits path against
`SGLANG_DLLM_TP_LOCAL_VOCAB=true`. A deterministic equivalence smoke matched
output token ids for both measured models.

## JetLM/SDAR-8B-Chat

Artifacts:
`/kelp/vocab/sglang-lab/bench/paper-sdar-tp-local-vocab-20260622182638/paper_outputs/8b`

| concurrency | baseline tok/s | TP-local tok/s | speedup | baseline p95 latency | TP-local p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 185.2 | 184.9 | 0.998x | 0.1761 | 0.1755 |
| 4 | 563.7 | 578.2 | 1.026x | 0.2291 | 0.2230 |
| 16 | 1372 | 1474 | 1.074x | 0.3746 | 0.3647 |
| 32 | 1340 | 1469 | 1.096x | 3.739 | 2.789 |

The concurrency-32 mean includes a slow repeat on both variants. The
concurrency-16 row is the cleaner high-concurrency signal for this run.

## JetLM/SDAR-30B-A3B-Chat-b32

Artifacts:
`/kelp/vocab/sglang-lab/bench/paper-sdar30b-tp-local-vocab-20260622184732/paper_outputs/30b`

| concurrency | baseline tok/s | TP-local tok/s | speedup | baseline p95 latency | TP-local p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 141.3 | 140.8 | 0.996x | 0.2275 | 0.2288 |
| 4 | 428.6 | 437.5 | 1.021x | 0.3013 | 0.3014 |
| 16 | 1142 | 1239 | 1.084x | 0.4499 | 0.4156 |
| 32 | 1541 | 1658 | 1.076x | 0.6659 | 0.7223 |

The 30B run shows the expected high-concurrency benefit while preserving the
same deterministic smoke output. Concurrency 32 has higher TP-local variance due
to one slower repeat; concurrency 16 is again the lower-noise summary point.

## Summary

TP-local vocabulary state is near-neutral at low concurrency and improves
throughput when batching is large enough for full-vocabulary logit materialization
and TP full-logits traffic to matter:

- SDAR-8B: about 1.07x at concurrency 16, about 1.10x at concurrency 32.
- SDAR-30B-A3B: about 1.08x at concurrency 16, about 1.08x at concurrency 32.

This benchmark is a serving smoke benchmark, not a full quality evaluation.
