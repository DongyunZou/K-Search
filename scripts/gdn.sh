export WANDB_ENTITY="han2024"
KSEARCH_ROOT="${KSEARCH_ROOT:-$PWD}"
DATASET_ROOT="$HOME/dataset/flashinfer-trace"

MODEL_NAME="${MODEL_NAME:-gpt-5.4}"
API_KEY="${API_KEY:-}"
BASE_URL="${BASE_URL:-}"

DEFINITION="gdn_decode_qk4_v8_d128_k_last"
LANGUAGE="${LANGUAGE:-cuda}"
TARGET_GPU="${TARGET_GPU:-H100}"

BASELINE_SOLUTION="flashinfer_wrapper_9b7f1e"
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
  --api-key "${API_KEY}" \
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
    901e5104-dccb-4c3f-ae13-ef4d31a4d456 \
    a5714b69-525c-4b95-bb7a-a0f9770c2f48 \
    3daa0974-293c-4414-b3c2-1f04368c1189 \
    22d5cef5-4f30-4f43-9d5f-0e9e95dc2201 \
    aed4bdd4-3139-4a1b-ae2f-aab8d4ba4090 \
    49a125e5-edf0-492e-a8b3-3676d14adaa3 \
    798635cf-d424-4343-a959-c96b0c0e81fb