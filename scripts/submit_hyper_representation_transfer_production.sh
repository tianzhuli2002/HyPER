#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-production}
case "$MODE" in
    production|smoke) ;;
    *) echo "Usage: $0 [production|smoke]" >&2; exit 1 ;;
esac

REPO=${HYPER_REPRESENTATION_REPO:-/net/scratch/w00238tl/HyPER_representation_transfer_prod}
STAGE_SCRIPT="$REPO/scripts/run_hyper_representation_transfer_production_stage.sh"

if [[ "$MODE" == "smoke" ]]; then
    OUTPUT=${HYPER_REPRESENTATION_OUTPUT:-$REPO/results/representation_transfer/ttbar1L/production_smoke_2000}
    ALIGNMENT_EVENTS=${HYPER_ALIGNMENT_EVENTS:-2000}
    CKA_EVENTS=${HYPER_CKA_EVENTS:-2000}
    TEST_EVENTS=${HYPER_TEST_EVENTS:-2000}
    SHUFFLES=${HYPER_SHUFFLES:-3}
    RANDOM_CONTROLS=${HYPER_RANDOM_CONTROLS:-3}
    BOOTSTRAPS=${HYPER_BOOTSTRAPS:-50}
    BATCH_SIZE=${HYPER_BATCH_SIZE:-256}
    GPU_TIME=08:00:00
    CPU_TIME=08:00:00
    GPU_MEMORY=64G
    ALIGNMENT_MEMORY=128G
    STATISTICS_MEMORY=128G
else
    OUTPUT=${HYPER_REPRESENTATION_OUTPUT:-$REPO/results/representation_transfer/ttbar1L/physics_selection_20260728T111408Z_retry1_production}
    ALIGNMENT_EVENTS=${HYPER_ALIGNMENT_EVENTS:-100000}
    CKA_EVENTS=${HYPER_CKA_EVENTS:-100000}
    TEST_EVENTS=${HYPER_TEST_EVENTS:-912666}
    SHUFFLES=${HYPER_SHUFFLES:-50}
    RANDOM_CONTROLS=${HYPER_RANDOM_CONTROLS:-20}
    BOOTSTRAPS=${HYPER_BOOTSTRAPS:-2000}
    BATCH_SIZE=${HYPER_BATCH_SIZE:-512}
    GPU_TIME=4-00:00:00
    CPU_TIME=2-00:00:00
    GPU_MEMORY=128G
    ALIGNMENT_MEMORY=256G
    STATISTICS_MEMORY=256G
fi

LOG_DIR="$OUTPUT/logs"
[[ -x "$STAGE_SCRIPT" ]] || { echo "ERROR: missing stage script $STAGE_SCRIPT" >&2; exit 1; }
mkdir -p "$LOG_DIR"
COMMIT=$(git -C "$REPO" rev-parse HEAD)
EXPORTS="ALL,HYPER_EXPECTED_COMMIT=$COMMIT,HYPER_REPRESENTATION_REPO=$REPO,HYPER_REPRESENTATION_OUTPUT=$OUTPUT,HYPER_ALIGNMENT_EVENTS=$ALIGNMENT_EVENTS,HYPER_CKA_EVENTS=$CKA_EVENTS,HYPER_TEST_EVENTS=$TEST_EVENTS,HYPER_SHUFFLES=$SHUFFLES,HYPER_RANDOM_CONTROLS=$RANDOM_CONTROLS,HYPER_BOOTSTRAPS=$BOOTSTRAPS,HYPER_BATCH_SIZE=$BATCH_SIZE"

export_job=$(sbatch --parsable \
    --job-name=HyPER_rep_export \
    --partition=gpuL --gres=gpu:1 --cpus-per-task=12 --mem=96G --time="$GPU_TIME" \
    --output="$LOG_DIR/export-%j.out" --error="$LOG_DIR/export-%j.err" \
    --export="$EXPORTS" "$STAGE_SCRIPT" export)

align_job=$(sbatch --parsable \
    --job-name=HyPER_rep_align \
    --partition=multicore --cpus-per-task=32 --mem="$ALIGNMENT_MEMORY" --time="$CPU_TIME" \
    --array=0-2 --dependency=afterok:$export_job \
    --output="$LOG_DIR/align-%A_%a.out" --error="$LOG_DIR/align-%A_%a.err" \
    --export="$EXPORTS" "$STAGE_SCRIPT" align)

cka_job=$(sbatch --parsable \
    --job-name=HyPER_rep_cka \
    --partition=multicore --cpus-per-task=32 --mem=192G --time="$CPU_TIME" \
    --array=0-2 --dependency=afterok:$export_job \
    --output="$LOG_DIR/cka-%A_%a.out" --error="$LOG_DIR/cka-%A_%a.err" \
    --export="$EXPORTS" "$STAGE_SCRIPT" cka)

full_cka_job=$(sbatch --parsable \
    --job-name=HyPER_rep_fullcka \
    --partition=gpuL --gres=gpu:1 --cpus-per-task=12 --mem=96G --time="$GPU_TIME" \
    --dependency=afterok:$export_job \
    --output="$LOG_DIR/full-cka-%j.out" --error="$LOG_DIR/full-cka-%j.err" \
    --export="$EXPORTS" "$STAGE_SCRIPT" full-cka)

evaluate_job=$(sbatch --parsable \
    --job-name=HyPER_rep_eval \
    --partition=gpuL --gres=gpu:1 --cpus-per-task=12 --mem="$GPU_MEMORY" --time="$GPU_TIME" \
    --dependency=afterok:$align_job \
    --output="$LOG_DIR/evaluate-%j.out" --error="$LOG_DIR/evaluate-%j.err" \
    --export="$EXPORTS" "$STAGE_SCRIPT" evaluate)

statistics_job=$(sbatch --parsable \
    --job-name=HyPER_rep_stats \
    --partition=multicore --cpus-per-task=32 --mem="$STATISTICS_MEMORY" --time="$CPU_TIME" \
    --dependency=afterok:$evaluate_job \
    --output="$LOG_DIR/statistics-%j.out" --error="$LOG_DIR/statistics-%j.err" \
    --export="$EXPORTS" "$STAGE_SCRIPT" statistics)

plots_job=$(sbatch --parsable \
    --job-name=HyPER_rep_plots \
    --partition=multicore --cpus-per-task=8 --mem=64G --time=08:00:00 \
    --dependency=afterok:$statistics_job:$cka_job:$full_cka_job \
    --output="$LOG_DIR/plots-%j.out" --error="$LOG_DIR/plots-%j.err" \
    --export="$EXPORTS" "$STAGE_SCRIPT" plots)

validate_job=$(sbatch --parsable \
    --job-name=HyPER_rep_validate \
    --partition=multicore --cpus-per-task=4 --mem=32G --time=02:00:00 \
    --dependency=afterok:$plots_job \
    --output="$LOG_DIR/validate-%j.out" --error="$LOG_DIR/validate-%j.err" \
    --export="$EXPORTS" "$STAGE_SCRIPT" validate)

cat <<EOF
HyPER representation-transfer $MODE workflow submitted at commit $COMMIT
  export:     $export_job
  align:      $align_job
  CKA:        $cka_job
  full CKA:   $full_cka_job
  evaluation: $evaluate_job
  statistics: $statistics_job
  plots:      $plots_job
  validation: $validate_job
  output:     $OUTPUT
  settings:   alignment=$ALIGNMENT_EVENTS CKA=$CKA_EVENTS test=$TEST_EVENTS shuffles=$SHUFFLES random=$RANDOM_CONTROLS bootstraps=$BOOTSTRAPS
EOF
