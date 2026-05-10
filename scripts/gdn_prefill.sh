WANDB_ENTITY="${WANDB_ENTITY:-han2024}"
KSEARCH_ROOT="${KSEARCH_ROOT:-$PWD}"
DATASET_ROOT="$HOME/dataset/flashinfer-trace"

MODEL_NAME="${MODEL_NAME:-claude-opus-4-6}"
API_KEY="${API_KEY:-}"
BASE_URL="${BASE_URL:-}"

DEFINITION="gdn_prefill_qk4_v8_d128_k_last"
LANGUAGE="${LANGUAGE:-cuda}"
TARGET_GPU="${TARGET_GPU:-H100}"

BASELINE_SOLUTION="flashinfer_wrapper_123ca6"
CONTINUE_FROM_SOLUTION="${CONTINUE_FROM_SOLUTION:-}"

MAX_OPT_ROUNDS="${MAX_OPT_ROUNDS:-1000}"
WM_STAGNATION_WINDOW="${WM_STAGNATION_WINDOW:-5}"

ARTIFACTS_DIR="${ARTIFACTS_DIR:-.ksearch-output}"

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
  --feedback-workloads \
    77daf91d-0660-4c4b-8c32-336a69281cd9 \
    1efaf2a9-05db-4737-8f41-6880ba1bb487 \
    fd072ba6-2190-4ce6-b96c-5212d4caf6a0 \
    4b94d568-35ce-45a5-91eb-f8dc4b7077a7 \
    109addb1-15e0-4ff2-9b39-df3e79746af0 \
    9a5d694b-7d4c-4ee6-8315-a13053ab6f92
