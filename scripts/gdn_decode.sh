WANDB_ENTITY="${WANDB_ENTITY:-han2024}"
KSEARCH_ROOT="${KSEARCH_ROOT:-$PWD}"
DATASET_ROOT="$HOME/dataset/flashinfer-trace"

MODEL_NAME="${MODEL_NAME:-claude-opus-4-6}"
API_KEY="${API_KEY:-}"
BASE_URL="${BASE_URL:-}"
USE_CLAUDE_CLI="${USE_CLAUDE_CLI:-}"

DEFINITION="gdn_decode_qk4_v8_d128_k_last"
LANGUAGE="${LANGUAGE:-cuda}"
TARGET_GPU="${TARGET_GPU:-H100}"

BASELINE_SOLUTION="flashinfer_wrapper_9b7f1e"
CONTINUE_FROM_SOLUTION="${CONTINUE_FROM_SOLUTION:-}"

MAX_OPT_ROUNDS="${MAX_OPT_ROUNDS:-120}"
WM_STAGNATION_WINDOW="${WM_STAGNATION_WINDOW:-5}"

ARTIFACTS_DIR="${ARTIFACTS_DIR:-.ksearch-output}"
GPU_LOCK_PATH="${GPU_LOCK_PATH:-/tmp/ksearch_gpu.lock}"

WANDB_PROJECT="${WANDB_PROJECT:-kernel_agent}"
RUN_NAME="${RUN_NAME:-${MODEL_NAME}-${LANGUAGE}-wm-${DEFINITION}-seed-opt${MAX_OPT_ROUNDS}}"

python -u "${KSEARCH_ROOT}/generate_kernels_and_eval.py" \
  --local "${DATASET_ROOT}" \
  --task-source flashinfer \
  --task-path "${DATASET_ROOT}" \
  --definition "${DEFINITION}" \
  --model-name "${MODEL_NAME}" \
  ${API_KEY:+--api-key "${API_KEY}"} \
  ${BASE_URL:+--base-url "${BASE_URL}"} \
  ${USE_CLAUDE_CLI:+--use-claude-cli} \
  --language "${LANGUAGE}" \
  --target-gpu "${TARGET_GPU}" \
  --world-model \
  --auto-resume \
  --wm-stagnation-window "${WM_STAGNATION_WINDOW}" \
  --max-opt-rounds "${MAX_OPT_ROUNDS}" \
  --parallel-workloads \
  ${CONTINUE_FROM_SOLUTION:+--continue-from-solution "${CONTINUE_FROM_SOLUTION}"} \
  --save-solutions \
  --use-isolated-runner \
  --baseline-solution "${BASELINE_SOLUTION}" \
  --wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --run-name "${RUN_NAME}" \
  --artifacts-dir "${ARTIFACTS_DIR}" \
  --gpu-lock-path "${GPU_LOCK_PATH}" \
  --feedback-workloads \
    901e5104-dccb-4c3f-ae13-ef4d31a4d456 \
    ec9d2340-6d13-40e4-a6fe-4483a1cacd0d \
    2640f1e3-4b02-4041-bdc1-59a28e0b9954 \
    a8c8beff-e414-4580-b2f7-e5b8f13bc269 \
    53385c7f-393d-41db-aec8-5b9eb5bf35d1 \
    eaf0a285-447c-4432-8e68-d287acc3cb08
