# Diffusion Kernel Profile Runbook 2026-06-22

This runbook captures profiler evidence for the diffusion TP-local vocabulary
and native FlashDenoise paths. Keep the raw profiler outputs with the benchmark
JSON so reviewers can distinguish a serving-level speedup from a kernel-level
mechanism.

## Required Artifacts

- Nsight Systems trace (`.nsys-rep` plus exported text or SQLite summary).
- Nsight Compute report (`.ncu-rep`) for the target kernel or pytest.
- Text summary with command, git commit, GPU type, CUDA driver, model or test
  shape, and the observed launch count, collective count, target kernel time,
  bandwidth, and SM utilization.

## vLLM Native Kernel Nsight Compute

Run this on an H100 node from the vLLM checkout after installing the branch-built
wheel or editable extension build:

```bash
cd /mnt/lvm/minsub/vocab/vllm
mkdir -p /mnt/lvm/minsub/vocab/artifacts/diffusion_profile/vllm_native
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  ncu \
    --set full \
    --target-processes all \
    --force-overwrite \
    --export /mnt/lvm/minsub/vocab/artifacts/diffusion_profile/vllm_native/flashdenoise_native_tp \
    python -m pytest -s tests/models/test_diffusion_gemma_flashdenoise_native_tp.py
```

Also save a text summary:

```bash
ncu --import /mnt/lvm/minsub/vocab/artifacts/diffusion_profile/vllm_native/flashdenoise_native_tp.ncu-rep \
  --page details \
  > /mnt/lvm/minsub/vocab/artifacts/diffusion_profile/vllm_native/flashdenoise_native_tp_ncu.txt
```

## SGLang Packed Gather Nsight Systems

Run this from the SGLang checkout. The pytest exercises the packed
rank-uniform TP-state merge and the legacy path in the same process, so the
trace can show gather launch differences without server noise:

```bash
cd /mnt/lvm/minsub/vocab/sglang
mkdir -p /mnt/lvm/minsub/vocab/artifacts/diffusion_profile/sglang_packed_gather
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  nsys profile \
    --trace cuda,nvtx,osrt \
    --force-overwrite true \
    --output /mnt/lvm/minsub/vocab/artifacts/diffusion_profile/sglang_packed_gather/tp_state_packed_gather \
    python -m pytest -s \
      test/srt/dllm/test_tp_local_vocab_state.py::test_dllm_vocab_state_tp_merge_uses_rank_uniform_packed_or_legacy_path
```

Export a readable summary next to the trace:

```bash
nsys stats \
  --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,nvtx_sum \
  /mnt/lvm/minsub/vocab/artifacts/diffusion_profile/sglang_packed_gather/tp_state_packed_gather.nsys-rep \
  > /mnt/lvm/minsub/vocab/artifacts/diffusion_profile/sglang_packed_gather/tp_state_packed_gather_nsys.txt
```

## Acceptance Criteria

Only claim a kernel-level mechanism when profiler artifacts show at least one
of the following against the appropriate baseline:

- Fewer CUDA kernel launches or fewer TP collectives in the profiled region.
- Lower time in the target kernel, target collectives, or total profiled region.
- Improved effective bandwidth or SM utilization for the target operation.

If the profiler does not show fewer launches, fewer collectives, lower target
time, or improved bandwidth/SM utilization, report the result as serving-level
evidence only and do not make a kernel-level claim.
