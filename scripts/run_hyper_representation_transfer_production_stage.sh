#!/usr/bin/env bash
set -euo pipefail

STAGE=${1:?Usage: $0 {export|align|cka|full-cka|evaluate|statistics|plots|validate}}

REPO=${HYPER_REPRESENTATION_REPO:-/net/scratch/w00238tl/HyPER_representation_transfer_prod}
PROD=${HYPER_ORIGINAL_RUNTIME:-/net/scratch/w00238tl/HyPER_24_2_speedup_prod}
PY=${HYPER_PYTHON:-/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/envs/HyPER_ONNX/bin/python}
OUTPUT=${HYPER_REPRESENTATION_OUTPUT:-$REPO/results/representation_transfer/ttbar1L/physics_selection_20260728T111408Z_retry1_production}

CAMPAIGN="$PROD/results/tuning_campaigns/stage1_v2_20260721_production_fix2/physics_selection_20260728T111408Z_retry1/ttbar1L"
CLASS_CONFIG="$CAMPAIGN/final_configs/classification_only.yaml"
RECO_CONFIG="$CAMPAIGN/final_configs/reconstruction_only.yaml"
JOINT_CONFIG="$CAMPAIGN/final_configs/joint.yaml"

CLASS_RUN="$PROD/results/sb_transfer/hyper/ttbar1L/classification_only/promoted_physics_selection_20260728T111408Z_retry1_ttbar1L_classification_only"
RECO_RUN="$PROD/results/sb_transfer/hyper/ttbar1L/reconstruction_only/promoted_physics_selection_20260728T111408Z_retry1_ttbar1L_reconstruction_only"
JOINT_RUN="$PROD/results/sb_transfer/hyper/ttbar1L/joint/promoted_physics_selection_20260728T111408Z_retry1_ttbar1L_joint"

CLASS_CHECKPOINT="$CLASS_RUN/training/version_0/checkpoints/best-total-epoch=038-val_loss=0.350171.ckpt"
RECO_CHECKPOINT="$RECO_RUN/training/version_0/checkpoints/best-total-epoch=091-val_loss=0.076488.ckpt"
JOINT_CHECKPOINT="$JOINT_RUN/training/version_0/checkpoints/best-total-epoch=051-val_loss=0.085340.ckpt"

H5="$PROD/HyPER_ttbarSL_typed/raw/training_dataset_even.h5"
SPLIT_CACHE="$PROD/results/integration_validation/splits/ttbar1L_full_schema4.npz"
RECO_MANIFEST="$RECO_RUN/run_manifest.json"
JOINT_MANIFEST="$JOINT_RUN/run_manifest.json"

ALIGNMENT_EVENTS=${HYPER_ALIGNMENT_EVENTS:-100000}
CKA_EVENTS=${HYPER_CKA_EVENTS:-100000}
TEST_EVENTS=${HYPER_TEST_EVENTS:-912666}
SHUFFLES=${HYPER_SHUFFLES:-50}
RANDOM_CONTROLS=${HYPER_RANDOM_CONTROLS:-20}
BOOTSTRAPS=${HYPER_BOOTSTRAPS:-2000}
SEED=${HYPER_SEED:-42}
BATCH_SIZE=${HYPER_BATCH_SIZE:-512}
NUM_WORKERS=${HYPER_NUM_WORKERS:-11}
CONTROL_CHUNK_SIZE=${HYPER_CONTROL_CHUNK_SIZE:-10}
OVERWRITE=${HYPER_OVERWRITE:-0}

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TERM=${TERM:-xterm-256color}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd "$REPO"

if [[ -n "${HYPER_EXPECTED_COMMIT:-}" ]]; then
    CURRENT_COMMIT=$(git rev-parse HEAD)
    if [[ "$CURRENT_COMMIT" != "$HYPER_EXPECTED_COMMIT" ]]; then
        echo "ERROR: runtime commit $CURRENT_COMMIT differs from expected $HYPER_EXPECTED_COMMIT" >&2
        exit 1
    fi
fi

for path in \
    "$CLASS_CONFIG" "$RECO_CONFIG" "$JOINT_CONFIG" \
    "$CLASS_CHECKPOINT" "$RECO_CHECKPOINT" "$JOINT_CHECKPOINT" \
    "$H5" "$SPLIT_CACHE" "$RECO_MANIFEST" "$JOINT_MANIFEST"
do
    [[ -e "$path" ]] || { echo "ERROR: missing required input $path" >&2; exit 1; }
done

DATASET_ROOT=$(
    "$PY" - "$RECO_MANIFEST" <<'PYDATASET'
import json
import sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(Path(manifest["graph_db_path"]).resolve().parent)
PYDATASET
)
RECO_PREDICTIONS=$(
    "$PY" - "$RECO_MANIFEST" <<'PYPREDRECO'
import json
import sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(Path(manifest["prediction_output"]).resolve())
PYPREDRECO
)
JOINT_PREDICTIONS=$(
    "$PY" - "$JOINT_MANIFEST" <<'PYPREDJOINT'
import json
import sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(Path(manifest["prediction_output"]).resolve())
PYPREDJOINT
)

mkdir -p "$OUTPUT"/{representations,alignments,cka,full_test_cka,scores,metrics,plots,logs}

valid_export() {
    local path=$1
    local expected=$2
    "$PY" - "$path" "$expected" <<'PYVALIDEXPORT' >/dev/null 2>&1
import sys
from pathlib import Path
import numpy as np
path = Path(sys.argv[1])
expected = int(sys.argv[2])
if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit(1)
with np.load(path, allow_pickle=False) as loaded:
    required = (
        "source_event_index", "truth_class", "truth_fully_matched",
        "block_0", "block_1", "block_2", "final_event",
        "classification_head_input",
    )
    if any(name not in loaded for name in required):
        raise SystemExit(1)
    if len(loaded["source_event_index"]) != expected:
        raise SystemExit(1)
    if len(np.unique(loaded["source_event_index"])) != expected:
        raise SystemExit(1)
    for name in required[1:]:
        value = loaded[name]
        if value.shape[0] != expected or not np.isfinite(value).all():
            raise SystemExit(1)
    if loaded["final_event"].shape[1] != 1024:
        raise SystemExit(1)
    if not np.array_equal(loaded["final_event"], loaded["classification_head_input"]):
        raise SystemExit(1)
PYVALIDEXPORT
}

valid_cka() {
    local directory=$1
    local expected=$2
    "$PY" - "$directory" "$expected" <<'PYVALIDCKA' >/dev/null 2>&1
import json
import sys
from pathlib import Path
directory = Path(sys.argv[1])
expected = int(sys.argv[2])
required = (
    "cka_summary.json", "cka_matrix.csv", "cka_heatmap.pdf",
    "cka_heatmap.png", "cka_inclusive_heatmap.pdf",
    "cka_inclusive_heatmap.png", "cka_corresponding_layers.csv",
)
if any(not (directory / name).is_file() or (directory / name).stat().st_size == 0 for name in required):
    raise SystemExit(1)
summary = json.loads((directory / "cka_summary.json").read_text(encoding="utf-8"))
if summary.get("event_count") != expected:
    raise SystemExit(1)
if summary.get("classification_head_input_alias") != "final_event":
    raise SystemExit(1)
PYVALIDCKA
}

valid_full_cka() {
    local directory=$1
    local expected=$2
    "$PY" - "$directory" "$expected" <<'PYVALIDFULLCKA' >/dev/null 2>&1
import json
import sys
from pathlib import Path
directory = Path(sys.argv[1])
expected = int(sys.argv[2])
required = (
    "full_test_final_event_cka.json", "full_test_final_event_cka.csv",
    "full_test_final_event_cka.pdf", "full_test_final_event_cka.png",
)
if any(not (directory / name).is_file() or (directory / name).stat().st_size == 0 for name in required):
    raise SystemExit(1)
summary = json.loads((directory / "full_test_final_event_cka.json").read_text(encoding="utf-8"))
if summary.get("event_count") != expected or summary.get("subset_counts", {}).get("all") != expected:
    raise SystemExit(1)
if summary.get("representation") != "final_event":
    raise SystemExit(1)
PYVALIDFULLCKA
}

valid_evaluation() {
    local directory=$1
    local expected=$2
    local shuffles=$3
    local random_controls=$4
    "$PY" - "$directory" "$expected" "$shuffles" "$random_controls" <<'PYVALIDEVALUATION' >/dev/null 2>&1
import json
import sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
expected = int(sys.argv[2])
shuffles = int(sys.argv[3])
random_controls = int(sys.argv[4])
summary_path = root / "evaluation_summary.json"
if not summary_path.is_file():
    raise SystemExit(1)
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if summary.get("event_count") != expected or summary.get("num_random_controls") != random_controls:
    raise SystemExit(1)
scores = root / "scores"
required_scores = [
    "test_source_event_index", "test_truth_class", "test_truth_fully_matched",
    *summary.get("principal_score_fields", []),
]
if len(summary.get("principal_score_fields", [])) != 10:
    raise SystemExit(1)
for stem in required_scores:
    path = scores / f"{stem}.npy"
    if not path.is_file():
        raise SystemExit(1)
    array = np.load(path, mmap_mode="r")
    if array.shape != (expected,):
        raise SystemExit(1)
if len(np.unique(np.load(scores / "test_source_event_index.npy", mmap_mode="r"))) != expected:
    raise SystemExit(1)
controls = root / "controls"
metadata_path = controls / "control_metadata.json"
if not metadata_path.is_file():
    raise SystemExit(1)
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
for direction in (
    "reconstruction_to_classification",
    "reconstruction_to_joint",
    "joint_to_classification",
):
    for kind, count in (("shuffled", shuffles), ("random", random_controls)):
        path = controls / f"{direction}_{kind}_scores.npy"
        if not path.is_file() or np.load(path, mmap_mode="r").shape != (count, expected):
            raise SystemExit(1)
        if len(metadata.get(direction, {}).get(kind, [])) != count:
            raise SystemExit(1)
PYVALIDEVALUATION
}

valid_statistics() {
    local directory=$1
    local expected_events=$2
    local expected_bootstraps=$3
    local expected_shuffles=$4
    "$PY" - "$directory" "$expected_events" "$expected_bootstraps" "$expected_shuffles" <<'PYVALIDSTATS' >/dev/null 2>&1
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
events = int(sys.argv[2])
bootstraps = int(sys.argv[3])
shuffles = int(sys.argv[4])
required = (
    "full_test_metrics.csv", "full_test_metrics.json", "operating_points.csv",
    "bootstrap_distributions.npz", "bootstrap_auc_differences.npz",
    "bootstrap_summary.csv", "bootstrap_differences.csv", "bootstrap_summary.json",
    "alignment_control_auc_distributions.npz", "shuffled_null_summary.csv",
    "shuffled_null_summary.json",
)
if any(not (root / name).is_file() or (root / name).stat().st_size == 0 for name in required):
    raise SystemExit(1)
bootstrap = json.loads((root / "bootstrap_summary.json").read_text(encoding="utf-8"))
null = json.loads((root / "shuffled_null_summary.json").read_text(encoding="utf-8"))
if bootstrap.get("event_count") != events or bootstrap.get("replicates") != bootstraps:
    raise SystemExit(1)
if null.get("event_count") != events or null.get("expected_shuffled_alignments_per_direction") != shuffles:
    raise SystemExit(1)
PYVALIDSTATS
}

valid_plots() {
    local directory=$1
    "$PY" - "$directory" <<'PYVALIDPLOTS' >/dev/null 2>&1
import sys
from pathlib import Path
root = Path(sys.argv[1])
stems = (
    "zero_shot_score_distributions", "zero_shot_roc", "zero_shot_background_rejection",
    "main_transfer_roc", "bridge_transfer_roc", "auc_bootstrap_summary",
    "bootstrap_auc_differences", "shuffled_alignment_nulls", "alignment_diagnostics",
    "cka_corresponding_layers", "cka_100k_vs_full_test",
    "score_correlation_reconstruction_to_classification_paired",
    "score_correlation_joint_to_classification_paired",
    "score_correlation_reconstruction_to_joint_paired",
    "cka_inclusive_classification_vs_reconstruction",
    "cka_inclusive_classification_vs_joint",
    "cka_inclusive_reconstruction_vs_joint",
    "scientific_summary",
)
for stem in stems:
    for suffix in (".pdf", ".png"):
        path = root / f"{stem}{suffix}"
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(1)
for name in ("scientific_summary.csv", "scientific_summary.json"):
    path = root / name
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(1)
PYVALIDPLOTS
}

prepare_output() {
    local path=$1
    local label=$2
    if [[ -d "$path" ]] && [[ -z "$(find "$path" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        rmdir "$path"
        return
    fi
    if [[ -e "$path" ]]; then
        if [[ "$OVERWRITE" == "1" ]]; then
            echo "OVERWRITE removing incomplete $label: $path"
            rm -rf "$path"
        else
            echo "ERROR: existing $label is incomplete or incompatible: $path" >&2
            echo "Set HYPER_OVERWRITE=1 to replace it deliberately." >&2
            exit 1
        fi
    fi
}

echo "stage=$STAGE"
echo "commit=$(git rev-parse HEAD)"
echo "output=$OUTPUT"
echo "dataset_root=$DATASET_ROOT"
echo "alignment_events=$ALIGNMENT_EVENTS cka_events=$CKA_EVENTS test_events=$TEST_EVENTS"
echo "shuffles=$SHUFFLES random_controls=$RANDOM_CONTROLS bootstraps=$BOOTSTRAPS"

case "$STAGE" in
export)
    if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi
    for split_count in "val:$ALIGNMENT_EVENTS" "test:$CKA_EVENTS"; do
        split=${split_count%%:*}
        count=${split_count##*:}
        for mode in classification reconstruction joint; do
            case "$mode" in
                classification) config=$CLASS_CONFIG; checkpoint=$CLASS_CHECKPOINT ;;
                reconstruction) config=$RECO_CONFIG; checkpoint=$RECO_CHECKPOINT ;;
                joint) config=$JOINT_CONFIG; checkpoint=$JOINT_CHECKPOINT ;;
            esac
            destination="$OUTPUT/representations/${mode}_${split}.npz"
            if valid_export "$destination" "$count"; then
                echo "REUSE verified representation export $destination"
                continue
            fi
            prepare_output "$destination" "representation export"
            "$PY" -u tools/export_hyper_representations.py \
                --config "$config" \
                --checkpoint "$checkpoint" \
                --h5 "$H5" \
                --split-cache "$SPLIT_CACHE" \
                --dataset-root "$DATASET_ROOT" \
                --split "$split" \
                --max-events "$count" \
                --seed "$SEED" \
                --batch-size "$BATCH_SIZE" \
                --num-workers "$NUM_WORKERS" \
                --accelerator gpu \
                --model-name "$mode" \
                --representations block_0 block_1 block_2 final_event classification_head_input \
                --output "$destination"
            valid_export "$destination" "$count" || {
                echo "ERROR: generated export failed validation: $destination" >&2
                exit 1
            }
        done
    done
    ;;
align)
    task=${SLURM_ARRAY_TASK_ID:-0}
    case "$task" in
        0) direction=reconstruction_to_classification; source=reconstruction; target=classification ;;
        1) direction=reconstruction_to_joint; source=reconstruction; target=joint ;;
        2) direction=joint_to_classification; source=joint; target=classification ;;
        *) echo "ERROR: alignment array index must be 0, 1, or 2" >&2; exit 1 ;;
    esac
    command=(
        "$PY" -u tools/fit_hyper_alignment_controls.py
        --source "$OUTPUT/representations/${source}_val.npz"
        --target "$OUTPUT/representations/${target}_val.npz"
        --source-key final_event
        --target-key classification_head_input
        --direction "$direction"
        --num-shuffles "$SHUFFLES"
        --seed-start 0
        --expected-event-count "$ALIGNMENT_EVENTS"
        --output-dir "$OUTPUT/alignments/$direction"
    )
    [[ "$OVERWRITE" == "1" ]] && command+=(--overwrite)
    "${command[@]}"
    ;;
cka)
    task=${SLURM_ARRAY_TASK_ID:-0}
    case "$task" in
        0) left=classification; right=reconstruction ;;
        1) left=classification; right=joint ;;
        2) left=reconstruction; right=joint ;;
        *) echo "ERROR: CKA array index must be 0, 1, or 2" >&2; exit 1 ;;
    esac
    pair=${left}_vs_${right}
    destination="$OUTPUT/cka/$pair"
    if valid_cka "$destination" "$CKA_EVENTS"; then
        echo "REUSE verified CKA output $destination"
        exit 0
    fi
    prepare_output "$destination" "CKA output"
    "$PY" -u tools/compute_hyper_cka.py \
        --left "$OUTPUT/representations/${left}_test.npz" \
        --right "$OUTPUT/representations/${right}_test.npz" \
        --output-dir "$destination" \
        --title "ttbar single-lepton: ${left} vs ${right}"
    valid_cka "$destination" "$CKA_EVENTS" || {
        echo "ERROR: generated CKA output failed validation: $destination" >&2
        exit 1
    }
    ;;
full-cka)
    if valid_full_cka "$OUTPUT/full_test_cka" "$TEST_EVENTS"; then
        echo "REUSE verified full-test CKA $OUTPUT/full_test_cka"
        exit 0
    fi
    prepare_output "$OUTPUT/full_test_cka" "full-test CKA output"
    mkdir -p "$OUTPUT/full_test_cka"
    if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi
    "$PY" -u tools/compute_hyper_streaming_cka.py \
        --classification-config "$CLASS_CONFIG" \
        --classification-checkpoint "$CLASS_CHECKPOINT" \
        --reconstruction-config "$RECO_CONFIG" \
        --reconstruction-checkpoint "$RECO_CHECKPOINT" \
        --joint-config "$JOINT_CONFIG" \
        --joint-checkpoint "$JOINT_CHECKPOINT" \
        --h5 "$H5" \
        --split-cache "$SPLIT_CACHE" \
        --dataset-root "$DATASET_ROOT" \
        --split test \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --accelerator gpu \
        --max-events "$TEST_EVENTS" \
        --expected-event-count "$TEST_EVENTS" \
        --output-dir "$OUTPUT/full_test_cka" \
        --title "ttbar single-lepton: full-test final-event CKA"
    valid_full_cka "$OUTPUT/full_test_cka" "$TEST_EVENTS" || {
        echo "ERROR: full-test CKA output failed validation" >&2
        exit 1
    }
    ;;
evaluate)
    if valid_evaluation "$OUTPUT/scores" "$TEST_EVENTS" "$SHUFFLES" "$RANDOM_CONTROLS"; then
        echo "REUSE verified full-test score evaluation $OUTPUT/scores"
        exit 0
    fi
    prepare_output "$OUTPUT/scores" "full-test score output"
    if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi
    "$PY" -u tools/evaluate_hyper_transfer_production.py \
        --classification-config "$CLASS_CONFIG" \
        --classification-checkpoint "$CLASS_CHECKPOINT" \
        --reconstruction-config "$RECO_CONFIG" \
        --reconstruction-checkpoint "$RECO_CHECKPOINT" \
        --joint-config "$JOINT_CONFIG" \
        --joint-checkpoint "$JOINT_CHECKPOINT" \
        --reconstruction-predictions "$RECO_PREDICTIONS" \
        --joint-predictions "$JOINT_PREDICTIONS" \
        --reconstruction-to-classification-alignments "$OUTPUT/alignments/reconstruction_to_classification" \
        --reconstruction-to-joint-alignments "$OUTPUT/alignments/reconstruction_to_joint" \
        --joint-to-classification-alignments "$OUTPUT/alignments/joint_to_classification" \
        --h5 "$H5" \
        --split-cache "$SPLIT_CACHE" \
        --dataset-root "$DATASET_ROOT" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --max-events "$TEST_EVENTS" \
        --expected-event-count "$TEST_EVENTS" \
        --num-random-controls "$RANDOM_CONTROLS" \
        --expected-shuffles "$SHUFFLES" \
        --expected-alignment-event-count "$ALIGNMENT_EVENTS" \
        --control-chunk-size "$CONTROL_CHUNK_SIZE" \
        --accelerator gpu \
        --seed "$SEED" \
        --output-dir "$OUTPUT/scores"
    valid_evaluation "$OUTPUT/scores" "$TEST_EVENTS" "$SHUFFLES" "$RANDOM_CONTROLS" || {
        echo "ERROR: full-test evaluation output failed validation" >&2
        exit 1
    }
    ;;
statistics)
    if valid_statistics "$OUTPUT/metrics" "$TEST_EVENTS" "$BOOTSTRAPS" "$SHUFFLES"; then
        echo "REUSE verified statistical outputs $OUTPUT/metrics"
        exit 0
    fi
    prepare_output "$OUTPUT/metrics" "statistical output"
    mkdir -p "$OUTPUT/metrics"
    "$PY" -u tools/bootstrap_hyper_transfer_metrics.py \
        --scores-dir "$OUTPUT/scores/scores" \
        --output-dir "$OUTPUT/metrics" \
        --replicates "$BOOTSTRAPS" \
        --chunk-size 8 \
        --seed "$SEED" \
        --expected-event-count "$TEST_EVENTS"
    "$PY" -u tools/summarize_hyper_alignment_controls.py \
        --scores-dir "$OUTPUT/scores/scores" \
        --controls-dir "$OUTPUT/scores/controls" \
        --output-dir "$OUTPUT/metrics" \
        --expected-shuffles "$SHUFFLES" \
        --workers "${SLURM_CPUS_PER_TASK:-1}"
    valid_statistics "$OUTPUT/metrics" "$TEST_EVENTS" "$BOOTSTRAPS" "$SHUFFLES" || {
        echo "ERROR: statistical output failed validation" >&2
        exit 1
    }
    ;;
plots)
    if valid_plots "$OUTPUT/plots"; then
        echo "REUSE verified final plots $OUTPUT/plots"
        exit 0
    fi
    prepare_output "$OUTPUT/plots" "plot output"
    mkdir -p "$OUTPUT/plots"
    "$PY" -u tools/plot_hyper_representation_transfer.py \
        --scores-dir "$OUTPUT/scores/scores" \
        --metrics-dir "$OUTPUT/metrics" \
        --controls-summary-dir "$OUTPUT/metrics" \
        --cka-root "$OUTPUT/cka" \
        --full-test-cka-dir "$OUTPUT/full_test_cka" \
        --alignments-root "$OUTPUT/alignments" \
        --output-dir "$OUTPUT/plots" \
        --title "ttbar single-lepton representation transfer"
    valid_plots "$OUTPUT/plots" || {
        echo "ERROR: final plot output failed validation" >&2
        exit 1
    }
    ;;
validate)
    "$PY" -u tools/validate_hyper_representation_production.py \
        --output-root "$OUTPUT" \
        --alignment-event-count "$ALIGNMENT_EVENTS" \
        --cka-event-count "$CKA_EVENTS" \
        --test-event-count "$TEST_EVENTS" \
        --shuffled-alignments "$SHUFFLES" \
        --bootstrap-replicates "$BOOTSTRAPS" \
        --random-controls "$RANDOM_CONTROLS"
    ;;
*)
    echo "ERROR: unknown stage $STAGE" >&2
    exit 1
    ;;
esac
