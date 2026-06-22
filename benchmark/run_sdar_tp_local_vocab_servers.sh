#!/usr/bin/env bash
# Reproducible SDAR TP-local-vocab server runbook.
#
# This script intentionally manages only PID files it created under RUN_ROOT.
# It never uses broad process matching or pkill.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_SOURCE_ROOT="/tmp/sglang-lab/src/sglang"
DEFAULT_PYTHON="/tmp/sglang-lab/venvs/dllm/bin/python"
DEFAULT_SDAR_8B_MODEL="/kelp/vocab/sglang-lab/models/JetLM_SDAR-8B-Chat"
DEFAULT_SDAR_30B_MODEL="/kelp/vocab/sglang-lab/models/JetLM_SDAR-30B-A3B-Chat-b32"

if [[ -d "/mnt/lvm/minsub/vocab/artifacts/sglang-bench" ]]; then
  DEFAULT_ARTIFACT_ROOT="/mnt/lvm/minsub/vocab/artifacts/sglang-bench"
elif [[ -d "/kelp/vocab/sglang-lab/bench" ]]; then
  DEFAULT_ARTIFACT_ROOT="/kelp/vocab/sglang-lab/bench"
else
  DEFAULT_ARTIFACT_ROOT="/mnt/lvm/minsub/vocab/artifacts/sglang-bench"
fi

SOURCE_ROOT="${SOURCE_ROOT:-${DEFAULT_SOURCE_ROOT}}"
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
SDAR_8B_MODEL="${SDAR_8B_MODEL:-${DEFAULT_SDAR_8B_MODEL}}"
SDAR_30B_MODEL="${SDAR_30B_MODEL:-${DEFAULT_SDAR_30B_MODEL}}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${DEFAULT_ARTIFACT_ROOT}}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${ARTIFACT_ROOT}/sdar_tp_local_vocab_${RUN_ID}}"
CACHE_RUN_ID="${CACHE_RUN_ID:-$(basename "${RUN_ROOT}")}"
CACHE_ROOT="${CACHE_ROOT:-/tmp/sglang-lab/runtime-cache}"
HOST="${HOST:-127.0.0.1}"
TP_SIZE="${TP_SIZE:-4}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
BASELINE_CUDA_VISIBLE_DEVICES="${BASELINE_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TP_LOCAL_CUDA_VISIBLE_DEVICES="${TP_LOCAL_CUDA_VISIBLE_DEVICES:-4,5,6,7}"
BENCH_REPEATS="${BENCH_REPEATS:-3}"
BENCH_REQUESTS="${BENCH_REQUESTS:-128}"
BENCH_WARMUP_REQUESTS="${BENCH_WARMUP_REQUESTS:-32}"
BENCH_CONCURRENCY="${BENCH_CONCURRENCY:-1 4 16 32}"
SMOKE_TIMEOUT_S="${SMOKE_TIMEOUT_S:-900}"
SMOKE_INTERVAL_S="${SMOKE_INTERVAL_S:-10}"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") start  <8b|30b> <baseline|tp_local|both>
  $(basename "$0") stop   <8b|30b|all> <baseline|tp_local|both|all>
  $(basename "$0") status <8b|30b|all> <baseline|tp_local|both|all>
  $(basename "$0") smoke  <8b|30b|all> <baseline|tp_local|both|all>
  $(basename "$0") bench  <8b|30b>
  $(basename "$0") env

Defaults, override with env vars:
  SOURCE_ROOT=${SOURCE_ROOT}
  PYTHON_BIN=${PYTHON_BIN}
  SDAR_8B_MODEL=${SDAR_8B_MODEL}
  SDAR_30B_MODEL=${SDAR_30B_MODEL}
  ARTIFACT_ROOT=${ARTIFACT_ROOT}
  RUN_ROOT=${RUN_ROOT}
  CACHE_ROOT=${CACHE_ROOT}
  CACHE_RUN_ID=${CACHE_RUN_ID}
  HOST=${HOST}
  TP_SIZE=${TP_SIZE}
  MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS}
  BASELINE_CUDA_VISIBLE_DEVICES=${BASELINE_CUDA_VISIBLE_DEVICES}
  TP_LOCAL_CUDA_VISIBLE_DEVICES=${TP_LOCAL_CUDA_VISIBLE_DEVICES}
  BENCH_REPEATS=${BENCH_REPEATS}
  BENCH_REQUESTS=${BENCH_REQUESTS}
  BENCH_WARMUP_REQUESTS=${BENCH_WARMUP_REQUESTS}
  BENCH_CONCURRENCY="${BENCH_CONCURRENCY}"

Ports:
  8b  baseline=18200 tp_local=18201
  30b baseline=18300 tp_local=18301

Server logs:
  \${RUN_ROOT}/server_logs/<8b|30b>/<baseline|tp_local>.log

PID files:
  \${RUN_ROOT}/pids/<8b|30b>/<baseline|tp_local>.pid

Examples:
  # Start both variants for SDAR-8B, then run the local benchmark.
  RUN_ROOT=/kelp/vocab/sglang-lab/bench/sdar8b_trial1 \\
    $(basename "$0") start 8b both
  RUN_ROOT=/kelp/vocab/sglang-lab/bench/sdar8b_trial1 \\
    $(basename "$0") bench 8b

  # Stop only pids recorded by this run root.
  RUN_ROOT=/kelp/vocab/sglang-lab/bench/sdar8b_trial1 \\
    $(basename "$0") stop all all
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

require_file_layout() {
  [[ -d "${SOURCE_ROOT}/python" ]] || die "SOURCE_ROOT does not contain python/: ${SOURCE_ROOT}"
  [[ -x "${PYTHON_BIN}" ]] || die "PYTHON_BIN is not executable: ${PYTHON_BIN}"
}

model_path() {
  case "$1" in
    8b) printf '%s\n' "${SDAR_8B_MODEL}" ;;
    30b) printf '%s\n' "${SDAR_30B_MODEL}" ;;
    *) die "unknown model '$1'; expected 8b or 30b" ;;
  esac
}

port_for() {
  local model="$1"
  local variant="$2"
  case "${model}:${variant}" in
    8b:baseline) printf '%s\n' "${SDAR_8B_BASELINE_PORT:-18200}" ;;
    8b:tp_local) printf '%s\n' "${SDAR_8B_TP_LOCAL_PORT:-18201}" ;;
    30b:baseline) printf '%s\n' "${SDAR_30B_BASELINE_PORT:-18300}" ;;
    30b:tp_local) printf '%s\n' "${SDAR_30B_TP_LOCAL_PORT:-18301}" ;;
    *) die "unknown model/variant '${model}:${variant}'" ;;
  esac
}

env_value_for() {
  case "$1" in
    baseline) printf 'false\n' ;;
    tp_local) printf 'true\n' ;;
    *) die "unknown variant '$1'; expected baseline or tp_local" ;;
  esac
}

cuda_visible_devices_for() {
  case "$1" in
    baseline) printf '%s\n' "${BASELINE_CUDA_VISIBLE_DEVICES}" ;;
    tp_local) printf '%s\n' "${TP_LOCAL_CUDA_VISIBLE_DEVICES}" ;;
    *) die "unknown variant '$1'; expected baseline or tp_local" ;;
  esac
}

variants_for() {
  case "$1" in
    baseline) printf '%s\n' baseline ;;
    tp_local) printf '%s\n' tp_local ;;
    both) printf '%s\n%s\n' baseline tp_local ;;
    all) printf '%s\n%s\n' baseline tp_local ;;
    *) die "unknown variant selector '$1'" ;;
  esac
}

models_for() {
  case "$1" in
    8b) printf '%s\n' 8b ;;
    30b) printf '%s\n' 30b ;;
    all) printf '%s\n%s\n' 8b 30b ;;
    *) die "unknown model selector '$1'" ;;
  esac
}

pid_file() {
  printf '%s/pids/%s/%s.pid\n' "${RUN_ROOT}" "$1" "$2"
}

log_file() {
  printf '%s/server_logs/%s/%s.log\n' "${RUN_ROOT}" "$1" "$2"
}

is_running_pid() {
  local pid="$1"
  [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null
}

pid_from_file() {
  local file="$1"
  [[ -f "${file}" ]] || return 1
  local pid
  pid="$(tr -d '[:space:]' <"${file}")"
  [[ -n "${pid}" ]] || return 1
  printf '%s\n' "${pid}"
}

url_for() {
  printf 'http://%s:%s\n' "${HOST}" "$(port_for "$1" "$2")"
}

launch_one() {
  local model="$1"
  local variant="$2"
  local model_dir
  local port
  local env_value
  local pid_path
  local log_path
  local cache_path

  require_file_layout
  model_dir="$(model_path "${model}")"
  [[ -d "${model_dir}" ]] || die "model path does not exist: ${model_dir}"
  port="$(port_for "${model}" "${variant}")"
  env_value="$(env_value_for "${variant}")"
  pid_path="$(pid_file "${model}" "${variant}")"
  log_path="$(log_file "${model}" "${variant}")"
  cache_path="${CACHE_ROOT}/${CACHE_RUN_ID}/${model}/${variant}"

  mkdir -p \
    "$(dirname "${pid_path}")" \
    "$(dirname "${log_path}")" \
    "${cache_path}/triton" \
    "${cache_path}/torchinductor"

  if [[ -f "${pid_path}" ]]; then
    local existing_pid
    existing_pid="$(pid_from_file "${pid_path}" || true)"
    if [[ -n "${existing_pid}" ]] && is_running_pid "${existing_pid}"; then
      die "${model}/${variant} already has running pid ${existing_pid} in ${pid_path}"
    fi
    rm -f "${pid_path}"
  fi

  echo "Starting ${model}/${variant} on ${HOST}:${port}"
  echo "  cuda visible devices: $(cuda_visible_devices_for "${variant}")"
  echo "  cache: ${cache_path}"
  echo "  log: ${log_path}"
  (
    cd "${REPO_ROOT}"
    export PYTHONPATH="${SOURCE_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
    export SGLANG_DLLM_TP_LOCAL_VOCAB="${env_value}"
    export CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES="$(cuda_visible_devices_for "${variant}")"
    export TRITON_CACHE_DIR="${cache_path}/triton"
    export TORCHINDUCTOR_CACHE_DIR="${cache_path}/torchinductor"
    exec "${PYTHON_BIN}" -m sglang.launch_server \
      --model-path "${model_dir}" \
      --trust-remote-code \
      --host "${HOST}" \
      --port "${port}" \
      --tp-size "${TP_SIZE}" \
      --dllm-algorithm LowConfidence \
      --attention-backend flashinfer \
      --sampling-backend flashinfer \
      --max-running-requests "${MAX_RUNNING_REQUESTS}"
  ) >"${log_path}" 2>&1 &

  local pid=$!
  printf '%s\n' "${pid}" >"${pid_path}"
  echo "  pid: ${pid_path} -> ${pid}"
}

smoke_one() {
  local model="$1"
  local variant="$2"
  local url
  local deadline
  local body
  url="$(url_for "${model}" "${variant}")"
  deadline=$((SECONDS + SMOKE_TIMEOUT_S))
  body='{"text":"Reply with OK.","sampling_params":{"max_new_tokens":4,"temperature":0}}'

  echo "Smoking ${model}/${variant} at ${url}/generate"
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 60 \
      -H 'Content-Type: application/json' \
      -d "${body}" \
      "${url}/generate" >/dev/null; then
      echo "  ok"
      return 0
    fi
    sleep "${SMOKE_INTERVAL_S}"
  done
  echo "  failed; tailing log:"
  tail -80 "$(log_file "${model}" "${variant}")" 2>/dev/null || true
  return 1
}

start_cmd() {
  local model_selector="${1:-}"
  local variant_selector="${2:-}"
  [[ -n "${model_selector}" && -n "${variant_selector}" ]] || die "start needs <8b|30b> <baseline|tp_local|both>"
  local model variant
  for model in $(models_for "${model_selector}"); do
    for variant in $(variants_for "${variant_selector}"); do
      launch_one "${model}" "${variant}"
    done
  done
  for model in $(models_for "${model_selector}"); do
    for variant in $(variants_for "${variant_selector}"); do
      smoke_one "${model}" "${variant}"
    done
  done
}

stop_one() {
  local model="$1"
  local variant="$2"
  local pid_path
  pid_path="$(pid_file "${model}" "${variant}")"
  if [[ ! -f "${pid_path}" ]]; then
    echo "${model}/${variant}: no pid file at ${pid_path}"
    return 0
  fi

  local pid
  pid="$(pid_from_file "${pid_path}" || true)"
  if [[ -z "${pid}" ]]; then
    echo "${model}/${variant}: empty or invalid pid file; removing ${pid_path}"
    rm -f "${pid_path}"
    return 0
  fi

  if ! is_running_pid "${pid}"; then
    echo "${model}/${variant}: pid ${pid} is not running; removing ${pid_path}"
    rm -f "${pid_path}"
    return 0
  fi

  echo "${model}/${variant}: stopping pid ${pid}"
  kill "${pid}" 2>/dev/null || true
  for _ in $(seq 1 30); do
    if ! is_running_pid "${pid}"; then
      rm -f "${pid_path}"
      echo "${model}/${variant}: stopped"
      return 0
    fi
    sleep 1
  done

  echo "${model}/${variant}: pid ${pid} still running after SIGTERM; sending SIGKILL to the recorded pid"
  kill -KILL "${pid}" 2>/dev/null || true
  rm -f "${pid_path}"
}

stop_cmd() {
  local model_selector="${1:-all}"
  local variant_selector="${2:-all}"
  local model variant
  for model in $(models_for "${model_selector}"); do
    for variant in $(variants_for "${variant_selector}"); do
      stop_one "${model}" "${variant}"
    done
  done
}

status_one() {
  local model="$1"
  local variant="$2"
  local pid_path
  local url
  pid_path="$(pid_file "${model}" "${variant}")"
  url="$(url_for "${model}" "${variant}")"
  if [[ ! -f "${pid_path}" ]]; then
    echo "${model}/${variant}: stopped (no pid file), url=${url}, log=$(log_file "${model}" "${variant}")"
    return 0
  fi
  local pid
  pid="$(pid_from_file "${pid_path}" || true)"
  if [[ -n "${pid}" ]] && is_running_pid "${pid}"; then
    echo "${model}/${variant}: running pid=${pid}, url=${url}, log=$(log_file "${model}" "${variant}")"
  else
    echo "${model}/${variant}: stale pid file ${pid_path}, url=${url}, log=$(log_file "${model}" "${variant}")"
  fi
}

status_cmd() {
  local model_selector="${1:-all}"
  local variant_selector="${2:-all}"
  local model variant
  for model in $(models_for "${model_selector}"); do
    for variant in $(variants_for "${variant_selector}"); do
      status_one "${model}" "${variant}"
    done
  done
}

smoke_cmd() {
  local model_selector="${1:-all}"
  local variant_selector="${2:-all}"
  local model variant
  for model in $(models_for "${model_selector}"); do
    for variant in $(variants_for "${variant_selector}"); do
      smoke_one "${model}" "${variant}"
    done
  done
}

bench_cmd() {
  local model="${1:-}"
  [[ "${model}" == "8b" || "${model}" == "30b" ]] || die "bench needs <8b|30b>"
  smoke_one "${model}" baseline
  smoke_one "${model}" tp_local

  local out_dir="${RUN_ROOT}/bench_results/${model}"
  mkdir -p "${out_dir}"
  local out_file="${out_dir}/dllm_tp_local_vocab_$(date +%Y%m%d_%H%M%S).json"
  echo "Benchmarking ${model}; output=${out_file}"
  local model_name
  case "${model}" in
    8b) model_name="JetLM/SDAR-8B-Chat" ;;
    30b) model_name="JetLM/SDAR-30B-A3B-Chat-b32" ;;
  esac
  (
    cd "${REPO_ROOT}"
    export PYTHONPATH="${SOURCE_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
    # shellcheck disable=SC2086
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/dllm_tp_local_vocab_bench.py" \
      --endpoint "baseline=$(url_for "${model}" baseline)" \
      --endpoint "tp_local=$(url_for "${model}" tp_local)" \
      --concurrency ${BENCH_CONCURRENCY} \
      --requests "${BENCH_REQUESTS}" \
      --warmup-requests "${BENCH_WARMUP_REQUESTS}" \
      --repeats "${BENCH_REPEATS}" \
      --model-name "${model_name}" \
      --tp-size "${TP_SIZE}" \
      --backend flashinfer \
      --max-running-requests "${MAX_RUNNING_REQUESTS}" \
      --variant-metadata "baseline_env=false" \
      --variant-metadata "tp_local_env=true" \
      --check-equivalence \
      --run-root "${RUN_ROOT}/paper_outputs/${model}" \
      --output "${out_file}"
  )
}

env_cmd() {
  cat <<EOF
SOURCE_ROOT=${SOURCE_ROOT}
PYTHON_BIN=${PYTHON_BIN}
SDAR_8B_MODEL=${SDAR_8B_MODEL}
SDAR_30B_MODEL=${SDAR_30B_MODEL}
ARTIFACT_ROOT=${ARTIFACT_ROOT}
RUN_ROOT=${RUN_ROOT}
CACHE_ROOT=${CACHE_ROOT}
CACHE_RUN_ID=${CACHE_RUN_ID}
HOST=${HOST}
TP_SIZE=${TP_SIZE}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS}
BASELINE_CUDA_VISIBLE_DEVICES=${BASELINE_CUDA_VISIBLE_DEVICES}
TP_LOCAL_CUDA_VISIBLE_DEVICES=${TP_LOCAL_CUDA_VISIBLE_DEVICES}
BENCH_REPEATS=${BENCH_REPEATS}
BENCH_REQUESTS=${BENCH_REQUESTS}
BENCH_WARMUP_REQUESTS=${BENCH_WARMUP_REQUESTS}
BENCH_CONCURRENCY="${BENCH_CONCURRENCY}"
SMOKE_TIMEOUT_S=${SMOKE_TIMEOUT_S}
SMOKE_INTERVAL_S=${SMOKE_INTERVAL_S}
EOF
}

main() {
  local cmd="${1:-}"
  shift || true
  case "${cmd}" in
    start) start_cmd "$@" ;;
    stop) stop_cmd "$@" ;;
    status) status_cmd "$@" ;;
    smoke) smoke_cmd "$@" ;;
    bench) bench_cmd "$@" ;;
    env) env_cmd ;;
    -h|--help|help|"") usage ;;
    *) usage >&2; die "unknown command '${cmd}'" ;;
  esac
}

main "$@"
