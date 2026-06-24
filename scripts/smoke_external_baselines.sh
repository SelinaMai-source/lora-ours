#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external_baselines"
TARGET="${1:-all}"
TIMEOUT_SECONDS="${SMOKE_TIMEOUT_SECONDS:-20}"

status=0

log() {
  printf '[external-smoke] %s\n' "$*"
}

run_cmd() {
  local label="$1"
  shift
  log "RUN ${label}: $*"
  if timeout "${TIMEOUT_SECONDS}" "$@"; then
    log "PASS ${label}"
  else
    local code=$?
    log "NEEDS_ENV ${label} (exit=${code})"
    status=1
  fi
}

compile_if_exists() {
  local label="$1"
  local file="$2"
  if [[ -f "${file}" ]]; then
    run_cmd "${label}" python -m py_compile "${file}"
  else
    log "SKIP ${label}: missing ${file}"
    status=1
  fi
}

shell_check_if_exists() {
  local label="$1"
  local file="$2"
  if [[ -f "${file}" ]]; then
    run_cmd "${label}" bash -n "${file}"
  else
    log "SKIP ${label}: missing ${file}"
    status=1
  fi
}

help_if_exists() {
  local label="$1"
  local file="$2"
  if [[ -f "${file}" ]]; then
    run_cmd "${label}" python "${file}" --help
  else
    log "SKIP ${label}: missing ${file}"
    status=1
  fi
}

smoke_o_lora() {
  compile_if_exists "o_lora.engine.py_compile" "${EXTERNAL_DIR}/o_lora/engine.py"
  compile_if_exists "o_lora.run_uie_lora.py_compile" "${EXTERNAL_DIR}/o_lora/src/run_uie_lora.py"
  help_if_exists "o_lora.run_uie_lora.help" "${EXTERNAL_DIR}/o_lora/src/run_uie_lora.py"
}

smoke_progressive_prompts() {
  compile_if_exists "progressive_prompts.train_t5_cl.py_compile" "${EXTERNAL_DIR}/progressive_prompts/T5_codebase/train_t5_cl.py"
  compile_if_exists "progressive_prompts.train_cl2.py_compile" "${EXTERNAL_DIR}/progressive_prompts/BERT_codebase/train_cl2.py"
  help_if_exists "progressive_prompts.train_t5_cl.help" "${EXTERNAL_DIR}/progressive_prompts/T5_codebase/train_t5_cl.py"
  help_if_exists "progressive_prompts.train_cl2.help" "${EXTERNAL_DIR}/progressive_prompts/BERT_codebase/train_cl2.py"
}

smoke_continual_t0() {
  compile_if_exists "continual_t0.setup.py_compile" "${EXTERNAL_DIR}/continual_t0/setup.py"
}

smoke_lfpt5() {
  compile_if_exists "lfpt5.convertmodel.py_compile" "${EXTERNAL_DIR}/lfpt5/convertmodel.py"
  shell_check_if_exists "lfpt5.classification_shell.syntax" "${EXTERNAL_DIR}/lfpt5/Classification/Classification.sh"
}

smoke_adaptercl_dialogue() {
  compile_if_exists "adaptercl_dialogue.train.py_compile" "${EXTERNAL_DIR}/adaptercl_dialogue/train.py"
  help_if_exists "adaptercl_dialogue.train.help" "${EXTERNAL_DIR}/adaptercl_dialogue/train.py"
}

smoke_lamol() {
  compile_if_exists "lamol.train.py_compile" "${EXTERNAL_DIR}/lamol/train.py"
  shell_check_if_exists "lamol.train_shell.syntax" "${EXTERNAL_DIR}/lamol/train.sh"
  shell_check_if_exists "lamol.test_shell.syntax" "${EXTERNAL_DIR}/lamol/test.sh"
}

smoke_trace_rcl() {
  compile_if_exists "trace_rcl.train.py_compile" "${EXTERNAL_DIR}/trace_rcl/train.py"
  compile_if_exists "trace_rcl.training_main.py_compile" "${EXTERNAL_DIR}/trace_rcl/training/main.py"
  shell_check_if_exists "trace_rcl.train_seq_cl.syntax" "${EXTERNAL_DIR}/trace_rcl/scripts/train_seq_cl.sh"
  shell_check_if_exists "trace_rcl.infer_seq.syntax" "${EXTERNAL_DIR}/trace_rcl/scripts/infer_seq.sh"
}

smoke_inf_lora() {
  compile_if_exists "inf_lora.main.py_compile" "${EXTERNAL_DIR}/inf_lora/main.py"
}

smoke_arper_dialog_nlg() {
  shell_check_if_exists "arper_dialog_nlg.run.syntax" "${EXTERNAL_DIR}/arper_dialog_nlg/run.sh"
  shell_check_if_exists "arper_dialog_nlg.preprocess.syntax" "${EXTERNAL_DIR}/arper_dialog_nlg/preprocess.sh"
}

smoke_bnm_reference() {
  compile_if_exists "bnm_reference.bnm_train_image.py_compile" "${EXTERNAL_DIR}/bnm_reference/DA/BNM/train_image.py"
  compile_if_exists "bnm_reference.cdan_bnm_train_image.py_compile" "${EXTERNAL_DIR}/bnm_reference/DA/CDAN-BNM/train_image.py"
}

run_target() {
  case "$1" in
    o_lora) smoke_o_lora ;;
    progressive_prompts) smoke_progressive_prompts ;;
    continual_t0) smoke_continual_t0 ;;
    lfpt5) smoke_lfpt5 ;;
    adaptercl_dialogue) smoke_adaptercl_dialogue ;;
    lamol) smoke_lamol ;;
    trace_rcl) smoke_trace_rcl ;;
    inf_lora) smoke_inf_lora ;;
    arper_dialog_nlg) smoke_arper_dialog_nlg ;;
    bnm_reference) smoke_bnm_reference ;;
    all)
      smoke_o_lora
      smoke_progressive_prompts
      smoke_continual_t0
      smoke_lfpt5
      smoke_adaptercl_dialogue
      smoke_lamol
      smoke_trace_rcl
      smoke_inf_lora
      smoke_arper_dialog_nlg
      smoke_bnm_reference
      ;;
    *)
      log "Unknown target: $1"
      log "Expected one of: all o_lora progressive_prompts continual_t0 lfpt5 adaptercl_dialogue lamol trace_rcl inf_lora arper_dialog_nlg bnm_reference"
      return 2
      ;;
  esac
}

run_target "${TARGET}"
exit "${status}"
