#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external_baselines"
TARGET="${1:-all}"
ACTION="${ACTION:-check}"
ENV_ROOT="${ENV_ROOT:-}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${ROOT_DIR}/.cache/pip}"

usage() {
  cat <<'USAGE'
Usage:
  ACTION=check  bash scripts/prepare_official_baseline_envs.sh [all|o_lora|progressive_prompts|adaptercl_dialogue|continual_t0]
  ACTION=create bash scripts/prepare_official_baseline_envs.sh [all|o_lora|progressive_prompts|adaptercl_dialogue|continual_t0]

ACTION=check only prints exact environment commands and verifies source files.
ACTION=create creates conda envs and installs official requirements. It does not
download data/checkpoints and does not start training.

Optional:
  ENV_ROOT=/root/autodl-tmp/conda_envs stores new envs outside the root partition.
  PIP_CACHE_DIR=/root/autodl-tmp/pip_cache stores pip wheels outside /root.
USAGE
}

log() {
  printf '[official-env] %s\n' "$*"
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    log "MISSING ${path}"
    return 1
  fi
  log "FOUND ${path}"
}

create_pip_env() {
  local env_name="$1"
  local python_version="$2"
  local requirements="$3"
  local conda_target=(-n "${env_name}")
  local run_target=(-n "${env_name}")
  if [[ -n "${ENV_ROOT}" ]]; then
    mkdir -p "${ENV_ROOT}"
    conda_target=(-p "${ENV_ROOT}/${env_name}")
    run_target=(-p "${ENV_ROOT}/${env_name}")
  fi
  mkdir -p "${PIP_CACHE_DIR}"
  log "CREATE conda env ${env_name} python=${python_version} env_root=${ENV_ROOT:-<conda-default>}"
  conda create -y "${conda_target[@]}" "python=${python_version}"
  log "INSTALL requirements ${requirements}"
  PIP_CACHE_DIR="${PIP_CACHE_DIR}" conda run "${run_target[@]}" python -m pip install -r "${requirements}"
}

check_o_lora() {
  local req="${EXTERNAL_DIR}/o_lora/requirements.txt"
  require_file "${req}"
  require_file "${EXTERNAL_DIR}/o_lora/src/run_uie_lora.py"
  log "O-LoRA create command: ACTION=create bash scripts/prepare_official_baseline_envs.sh o_lora"
}

create_o_lora() {
  create_pip_env "lora_v10_o_lora" "3.9" "${EXTERNAL_DIR}/o_lora/requirements.txt"
  local run_target=(-n lora_v10_o_lora)
  if [[ -n "${ENV_ROOT}" ]]; then
    run_target=(-p "${ENV_ROOT}/lora_v10_o_lora")
  fi
  log "PIN O-LoRA numpy/pyarrow/pydantic for legacy datasets/deepspeed compatibility"
  PIP_CACHE_DIR="${PIP_CACHE_DIR}" conda run "${run_target[@]}" python -m pip install 'numpy<2' 'pyarrow<13' 'pydantic<2'
  conda run "${run_target[@]}" python "${EXTERNAL_DIR}/o_lora/src/run_uie_lora.py" --help >/tmp/lora_v10_o_lora_help.txt
  log "O-LoRA help smoke wrote /tmp/lora_v10_o_lora_help.txt"
}

check_progressive_prompts() {
  local env="${EXTERNAL_DIR}/progressive_prompts/environment.yaml"
  require_file "${env}"
  require_file "${EXTERNAL_DIR}/progressive_prompts/T5_codebase/train_t5_cl.py"
  require_file "${EXTERNAL_DIR}/progressive_prompts/BERT_codebase/train_cl2.py"
  log "Progressive Prompts create command: ACTION=create bash scripts/prepare_official_baseline_envs.sh progressive_prompts"
}

create_progressive_prompts() {
  local conda_target=(-n lora_v10_progressive_prompts)
  local run_target=(-n lora_v10_progressive_prompts)
  if [[ -n "${ENV_ROOT}" ]]; then
    mkdir -p "${ENV_ROOT}"
    conda_target=(-p "${ENV_ROOT}/lora_v10_progressive_prompts")
    run_target=(-p "${ENV_ROOT}/lora_v10_progressive_prompts")
  fi
  PIP_CACHE_DIR="${PIP_CACHE_DIR}" conda env create "${conda_target[@]}" -f "${EXTERNAL_DIR}/progressive_prompts/environment.yaml"
  conda run "${run_target[@]}" python "${EXTERNAL_DIR}/progressive_prompts/T5_codebase/train_t5_cl.py" --help >/tmp/lora_v10_progressive_prompts_t5_help.txt
  conda run "${run_target[@]}" python "${EXTERNAL_DIR}/progressive_prompts/BERT_codebase/train_cl2.py" --help >/tmp/lora_v10_progressive_prompts_bert_help.txt
  log "Progressive Prompts help smokes wrote /tmp/lora_v10_progressive_prompts_*_help.txt"
}

check_adaptercl_dialogue() {
  local req="${EXTERNAL_DIR}/adaptercl_dialogue/requirements.txt"
  require_file "${req}"
  require_file "${EXTERNAL_DIR}/adaptercl_dialogue/train.py"
  require_file "${ROOT_DIR}/scripts/convert_tod37_to_stream.py"
  log "AdapterCL create command: ACTION=create bash scripts/prepare_official_baseline_envs.sh adaptercl_dialogue"
}

create_adaptercl_dialogue() {
  create_pip_env "lora_v10_adaptercl_dialogue" "3.8" "${EXTERNAL_DIR}/adaptercl_dialogue/requirements.txt"
  local run_target=(-n lora_v10_adaptercl_dialogue)
  if [[ -n "${ENV_ROOT}" ]]; then
    run_target=(-p "${ENV_ROOT}/lora_v10_adaptercl_dialogue")
  fi
  conda run "${run_target[@]}" python "${EXTERNAL_DIR}/adaptercl_dialogue/train.py" --help >/tmp/lora_v10_adaptercl_help.txt
  log "AdapterCL help smoke wrote /tmp/lora_v10_adaptercl_help.txt"
}

check_continual_t0() {
  local req="${EXTERNAL_DIR}/continual_t0/requirements.txt"
  require_file "${req}"
  require_file "${EXTERNAL_DIR}/continual_t0/setup.py"
  log "Continual-T0 create command: ACTION=create bash scripts/prepare_official_baseline_envs.sh continual_t0"
  log "Continual-T0 still requires T0/T5 checkpoints, official processed mixture, and rehearsal data before full run."
}

create_continual_t0() {
  create_pip_env "lora_v10_continual_t0" "3.9" "${EXTERNAL_DIR}/continual_t0/requirements.txt"
  local run_target=(-n lora_v10_continual_t0)
  if [[ -n "${ENV_ROOT}" ]]; then
    run_target=(-p "${ENV_ROOT}/lora_v10_continual_t0")
  fi
  conda run "${run_target[@]}" python -m py_compile "${EXTERNAL_DIR}/continual_t0/setup.py"
  log "Continual-T0 env created; data/checkpoint blockers remain."
}

run_one() {
  local target="$1"
  case "${target}:${ACTION}" in
    o_lora:check) check_o_lora ;;
    o_lora:create) create_o_lora ;;
    progressive_prompts:check) check_progressive_prompts ;;
    progressive_prompts:create) create_progressive_prompts ;;
    adaptercl_dialogue:check) check_adaptercl_dialogue ;;
    adaptercl_dialogue:create) create_adaptercl_dialogue ;;
    continual_t0:check) check_continual_t0 ;;
    continual_t0:create) create_continual_t0 ;;
    *) usage; return 2 ;;
  esac
}

case "${TARGET}" in
  all)
    run_one o_lora
    run_one progressive_prompts
    run_one adaptercl_dialogue
    run_one continual_t0
    ;;
  o_lora|progressive_prompts|adaptercl_dialogue|continual_t0)
    run_one "${TARGET}"
    ;;
  *)
    usage
    exit 2
    ;;
esac
