# Diffusion Correctness Runbook 2026-06-22

This runbook records the H100 correctness checks needed before using the
diffusion TP-local vocabulary and native FlashDenoise artifacts as paper or PR
evidence.

The H100 pod `p-ai-efficiency-tech/ms-qwen3-torchspec-2node` does not mount the
host path `/mnt/lvm/minsub/vocab`. Use `/tmp/codex-correctness` for copied source
inside the pod and `/kelp/vocab/paper-evidence` for persistent artifacts.

## Host-To-Pod Source Copy

Run these commands from the host workspace:

```bash
set -euo pipefail

for repo in /mnt/lvm/minsub/vocab/vllm /mnt/lvm/minsub/vocab/sglang; do
  if [ -n "$(git -C "${repo}" status --short)" ]; then
    echo "Commit, stash, or remove untracked changes before archiving ${repo}" >&2
    git -C "${repo}" status --short >&2
    exit 1
  fi
done

kubectl exec -n p-ai-efficiency-tech ms-qwen3-torchspec-2node -- \
  bash -lc 'rm -rf /tmp/codex-correctness && mkdir -p /tmp/codex-correctness/vllm /tmp/codex-correctness/sglang /kelp/vocab/paper-evidence'

cd /mnt/lvm/minsub/vocab/vllm
git archive --format=tar HEAD | \
  kubectl exec -i -n p-ai-efficiency-tech ms-qwen3-torchspec-2node -- \
  tar -xf - -C /tmp/codex-correctness/vllm

cd /mnt/lvm/minsub/vocab/sglang
git archive --format=tar HEAD | \
  kubectl exec -i -n p-ai-efficiency-tech ms-qwen3-torchspec-2node -- \
  tar -xf - -C /tmp/codex-correctness/sglang

write_source_provenance() {
  repo="$1"
  name="$2"
  tmp="/tmp/${name}-source-provenance.json"
  python3 - "$repo" "$name" > "$tmp" <<'PY'
import json
import subprocess
import sys

repo = sys.argv[1]
name = sys.argv[2]

def git(*args):
    return subprocess.check_output(["git", "-C", repo, *args], text=True).strip()

payload = {
    "repo": name,
    "path": repo,
    "commit": git("rev-parse", "HEAD"),
    "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    "status": subprocess.check_output(
        ["git", "-C", repo, "status", "--short"], text=True
    ),
    "remote_v": subprocess.check_output(
        ["git", "-C", repo, "remote", "-v"], text=True
    ),
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
  kubectl cp "$tmp" \
    "p-ai-efficiency-tech/ms-qwen3-torchspec-2node:/tmp/codex-correctness/${name}/.codex-source-provenance.json"
}

write_source_provenance /mnt/lvm/minsub/vocab/vllm vllm
write_source_provenance /mnt/lvm/minsub/vocab/sglang sglang
```

If a branch-built vLLM wheel exists on the host, copy it to the pod-accessible
artifact directory:

```bash
kubectl exec -n p-ai-efficiency-tech ms-qwen3-torchspec-2node -- \
  bash -lc 'mkdir -p /kelp/vocab/paper-evidence/wheels'

kubectl cp \
  /mnt/lvm/minsub/vocab/artifacts/wheels/vllm-branch-cu129.whl \
  p-ai-efficiency-tech/ms-qwen3-torchspec-2node:/kelp/vocab/paper-evidence/wheels/vllm-branch-cu129.whl
```

## Environment Manifest

Record environment and source provenance before running tests:

```bash
kubectl exec -n p-ai-efficiency-tech ms-qwen3-torchspec-2node -- bash -lc '
set -euo pipefail
cd /tmp/codex-correctness/sglang &&
PYTHONPATH=/tmp/codex-correctness/sglang/python \
/tmp/sglang-lab/venvs/dllm/bin/python benchmark/diffusion_evidence_manifest.py \
  --output-json /kelp/vocab/paper-evidence/manifest.json \
  --sglang-repo /tmp/codex-correctness/sglang \
  --vllm-repo /tmp/codex-correctness/vllm \
  --vllm-wheel /kelp/vocab/paper-evidence/wheels/vllm-branch-cu129.whl \
  --env-flag SGLANG_DLLM_TP_LOCAL_VOCAB=true \
  --env-flag SGLANG_DLLM_TP_LOCAL_VOCAB_PACKED_GATHER=true'
```

Expected: `/kelp/vocab/paper-evidence/manifest.json` contains both repo commits,
GPU information, package versions, env flags, and wheel SHA when the wheel is
present.

## SGLang Algebra Correctness

Run the local algebra tests for TP-local vocabulary state and packed gather:

```bash
kubectl exec -n p-ai-efficiency-tech ms-qwen3-torchspec-2node -- bash -lc '
set -euo pipefail
cd /tmp/codex-correctness/sglang &&
PYTHONPATH=/tmp/codex-correctness/sglang/python \
/tmp/sglang-lab/venvs/dllm/bin/python -m pytest -q \
test/srt/dllm/test_tp_local_vocab_state.py \
2>&1 | tee /kelp/vocab/paper-evidence/sglang-correctness.log'
```

Expected: all tests in `test_tp_local_vocab_state.py` pass except endpoint tests
that skip when endpoint URLs are not configured. Failures in local max,
logsumexp, clean argmax, sampled argmax, packed gather, or endpoint equivalence
block PR correctness claims until fixed or explained.

## vLLM Python Correctness With Release Extension Shims

If no branch-built wheel is installed, extract release extension shims only for
import and Python-level reference tests:

```bash
kubectl exec -n p-ai-efficiency-tech ms-qwen3-torchspec-2node -- bash -lc '
set -euo pipefail
rm -rf /tmp/codex-correctness/vllm-release-shim &&
cp -a /tmp/codex-correctness/vllm /tmp/codex-correctness/vllm-release-shim &&
cd /tmp/codex-correctness/vllm-release-shim &&
/tmp/sglang-lab/venvs/dllm/bin/python - <<PY
import zipfile
wheel = "/kelp/vllm-lab/upstream/vllm-0.23.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
needed = [
    "vllm/_C.abi3.so",
    "vllm/_C_stable_libtorch.abi3.so",
    "vllm/_version.py",
    "vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so",
    "vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so",
]
with zipfile.ZipFile(wheel) as z:
    for name in needed:
        z.extract(name, ".")
        print("extracted", name)
PY'
```

Then run:

```bash
kubectl exec -n p-ai-efficiency-tech ms-qwen3-torchspec-2node -- bash -lc '
set -euo pipefail
cd /tmp/codex-correctness/vllm-release-shim &&
PYTHONPATH=/tmp/codex-correctness/vllm-release-shim \
/tmp/sglang-lab/venvs/dllm/bin/python -m pytest -q \
tests/models/test_diffusion_gemma_flashdenoise.py \
tests/models/test_diffusion_gemma_flashdenoise_state.py \
tests/models/test_diffusion_gemma_local_vocab.py \
2>&1 | tee /kelp/vocab/paper-evidence/vllm-python-correctness.log'
```

Expected: Python reference tests pass. A skip for a native-only symbol is
acceptable in this phase only if the branch-built wheel phase below registers
and tests the native CUDA ops.

## vLLM Native CUDA Correctness After Branch-Built Wheel

Install the branch-built wheel copied into `/kelp/vocab/paper-evidence/wheels`,
then run the native op gate and dense-reference CUDA tests:

```bash
kubectl exec -n p-ai-efficiency-tech ms-qwen3-torchspec-2node -- bash -lc '
set -euo pipefail
cd /tmp/codex-correctness/vllm &&
/tmp/sglang-lab/venvs/dllm/bin/python -m pip install --force-reinstall \
  /kelp/vocab/paper-evidence/wheels/vllm-branch-cu129.whl &&
rm -rf /tmp/codex-correctness/vllm-native-wheel-check &&
mkdir -p /tmp/codex-correctness/vllm-native-wheel-check &&
cp benchmarks/diffusion_gemma_native_op_gate.py \
  /tmp/codex-correctness/vllm-native-wheel-check/ &&
cp tests/models/test_diffusion_gemma_flashdenoise_native_tp.py \
  /tmp/codex-correctness/vllm-native-wheel-check/ &&
cd /tmp &&
env -u PYTHONPATH /tmp/sglang-lab/venvs/dllm/bin/python \
  /tmp/codex-correctness/vllm-native-wheel-check/diffusion_gemma_native_op_gate.py \
  --output-json /kelp/vocab/paper-evidence/vllm-native-op-gate.json \
  --native-test-path /tmp/codex-correctness/vllm-native-wheel-check/test_diffusion_gemma_flashdenoise_native_tp.py \
  --run-native-tests \
2>&1 | tee /kelp/vocab/paper-evidence/vllm-native-op-gate.log'
```

Expected: `all_native_registered` is true in
`/kelp/vocab/paper-evidence/vllm-native-op-gate.json`, and
`tests/models/test_diffusion_gemma_flashdenoise_native_tp.py` passes. Any missing
native registration, CUDA assertion, dense-reference mismatch, or skipped native
test blocks native-kernel correctness claims. The gate exits 2 for missing
native registration, 3 when CUDA is unavailable, and 4 when pytest reports a
skip in the native CUDA tests.

## Endpoint A/B Correctness

After starting baseline and optimized servers, compare deterministic endpoint
outputs:

```bash
kubectl exec -n p-ai-efficiency-tech ms-qwen3-torchspec-2node -- bash -lc '
set -euo pipefail
cd /tmp/codex-correctness/sglang &&
PYTHONPATH=/tmp/codex-correctness/sglang/python \
/tmp/sglang-lab/venvs/dllm/bin/python benchmark/diffusion_endpoint_correctness_probe.py \
  --baseline-url http://127.0.0.1:18200 \
  --optimized-url http://127.0.0.1:18201 \
  --mode sglang \
  --output-json /kelp/vocab/paper-evidence/endpoint-ab-correctness.json'
```

Expected: the probe exits 0, `failures` is empty, and every row has
`match: true`. A mismatch means benchmark throughput artifacts are not valid
correctness evidence for the optimized path until the mismatch is resolved.
