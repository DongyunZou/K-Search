export WANDB_ENTITY="han2024"
KSEARCH_ROOT="${KSEARCH_ROOT:-$PWD}"
DATASET_ROOT="$HOME/dataset/flashinfer-trace"

MODEL_NAME="${MODEL_NAME:-claude-opus-4-6}"
API_KEY="${API_KEY:-}"
BASE_URL="${BASE_URL:-}"

DEFINITION="moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
LANGUAGE="${LANGUAGE:-cuda}"
TARGET_GPU="${TARGET_GPU:-H100}"

BASELINE_SOLUTION="flashinfer_wrapper_9sdjf3"
CONTINUE_FROM_SOLUTION="${CONTINUE_FROM_SOLUTION:-}"

MAX_OPT_ROUNDS="${MAX_OPT_ROUNDS:-20}"
WM_STAGNATION_WINDOW="${WM_STAGNATION_WINDOW:-7}"

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
  --wm-stagnation-window "${WM_STAGNATION_WINDOW}" \
  --max-opt-rounds "${MAX_OPT_ROUNDS}" \
  --parallel-workloads \
  ${CONTINUE_FROM_SOLUTION:+--continue-from-solution "${CONTINUE_FROM_SOLUTION}"} \
  --save-solutions \
  --use-isolated-runner \
  --baseline-solution "${BASELINE_SOLUTION}" \
  --wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --run-name "${RUN_NAME}" \
  --artifacts-dir "${ARTIFACTS_DIR}" \
  --feedback-workloads \
    e05c6c03-5603-4a1c-b34c-dcce0ecaeea4 \
    a7c2bcfd-a2f4-479e-8d32-200115df89cf \
    fc378037-e8fa-4305-b00f-4af47933fd53 \
    74d7ff04-0365-4cf8-a824-ce61b7131dea \
    5eadab1e-a0e3-4966-b0fd-1115cd77497c \
    5e8dc11c-f2a9-42d5-8dce-9419cbf34d5d
