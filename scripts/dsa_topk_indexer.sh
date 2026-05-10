WANDB_ENTITY="${WANDB_ENTITY:-han2024}"
KSEARCH_ROOT="${KSEARCH_ROOT:-$PWD}"
DATASET_ROOT="$HOME/dataset/flashinfer-trace"

MODEL_NAME="${MODEL_NAME:-claude-opus-4-6}"
API_KEY="${API_KEY:-}"
BASE_URL="${BASE_URL:-}"

DEFINITION="dsa_topk_indexer_fp8_h64_d128_topk2048_ps64"
LANGUAGE="${LANGUAGE:-cuda}"
TARGET_GPU="${TARGET_GPU:-H100}"

BASELINE_SOLUTION="flashinfer_deepgemm_wrapper_2ba145"
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
    30cecff1-7ea4-474b-90fc-7f4a87206d8e \
    4279d75e-b93c-4198-9016-4d1d21e17bf2 \
    a4cdaee6-2c3e-46fa-a60a-fef4ccf4c30b \
    1571c14a-181c-4f15-97a5-178e4b316ca5 \
    de54c4e6-7c89-43c7-aefb-db20265f4cdf \
    5db1b172-eda8-4714-9981-a069dc33d7e9
