WANDB_ENTITY="${WANDB_ENTITY:-}"
KSEARCH_ROOT="${KSEARCH_ROOT:-$PWD}"
DATASET_ROOT="$HOME/dataset/flashinfer-trace"

MODEL_NAME="${MODEL_NAME:-claude-opus-4-6}"
API_KEY="${API_KEY:-}"
BASE_URL="${BASE_URL:-}"
USE_CLAUDE_CLI="${USE_CLAUDE_CLI:-}"

DEFINITION="dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64"
LANGUAGE="${LANGUAGE:-cuda}"
TARGET_GPU="${TARGET_GPU:-H100}"

BASELINE_SOLUTION="flashinfer_wrapper_5af199"
CONTINUE_FROM_SOLUTION="${CONTINUE_FROM_SOLUTION:-}"

MAX_OPT_ROUNDS="${MAX_OPT_ROUNDS:-120}"
WM_STAGNATION_WINDOW="${WM_STAGNATION_WINDOW:-5}"

ARTIFACTS_DIR="${ARTIFACTS_DIR:-.ksearch-output}"

WANDB_PROJECT="${WANDB_PROJECT:-}"
RUN_NAME="${RUN_NAME:-${MODEL_NAME}-${LANGUAGE}-wm-${DEFINITION}-seed-opt${MAX_OPT_ROUNDS}}"

uv run python -u "${KSEARCH_ROOT}/generate_kernels_and_eval.py" \
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
  ${WANDB_PROJECT:+--wandb-project "${WANDB_PROJECT}"} \
  ${WANDB_ENTITY:+--wandb-entity "${WANDB_ENTITY}"} \
  --run-name "${RUN_NAME}" \
  --artifacts-dir "${ARTIFACTS_DIR}" \
  --feedback-workloads \
    0c23b10c7b7645719517828c12eaa1d2 \
    9d4a5f21268e484ea05a2f2af91d9fa7 \
    ddfa9e340b264f76abe7418692faa876 \
    3838996164a94d728710f913477feba8 \
    385742b2717e4f02b918c7349dde23d8
