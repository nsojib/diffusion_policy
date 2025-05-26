#!/usr/bin/env bash
set -euo pipefail

BASE_PATH="$1"  # path to either a .ckpt file or a directory
# DATASET_PATH="$2"  # path to the dataset file (optional)
DATASET_PATH="${2:-}"  # set to empty string if not provided



echo "BASE_PATH: $BASE_PATH"

# Initialize
CHECKPOINT_PATH=""

if [ -f "$BASE_PATH" ] && [[ "$BASE_PATH" == *.ckpt ]]; then
    echo "Provided path is a checkpoint file."
    CHECKPOINT_PATH="$BASE_PATH"

elif [ -d "$BASE_PATH" ]; then
    echo "Provided path is a directory. Searching for .ckpt files in: $BASE_PATH"
    for file in "$BASE_PATH"/*.ckpt; do
        [ -e "$file" ] || continue
        echo "Found checkpoint: $file"
        CHECKPOINT_PATH="$file"
        break
    done

else
    echo "Error: '$BASE_PATH' is neither a .ckpt file nor a directory." >&2
    exit 1
fi

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "No .ckpt files found in directory: $BASE_PATH" >&2
    exit 1
fi

echo "CHECKPOINT FOUND: $CHECKPOINT_PATH"

# === New: create a sibling directory named after the checkpoint (minus .ckpt) ===
CKPT_DIR="$(dirname "$CHECKPOINT_PATH")"
CKPT_NAME="$(basename "$CHECKPOINT_PATH" .ckpt)"
TARGET_DIR="$CKPT_DIR/$CKPT_NAME"

if [ -e "$TARGET_DIR" ]; then
    echo "Directory already exists: $TARGET_DIR"
    exit 1
else
    mkdir -p "$TARGET_DIR"
    echo "Created directory: $TARGET_DIR"
fi

# (Continue with whatever you need to do inside $TARGET_DIR…)

echo "running with seeds ... "

SEEDS=(42 43 44)
# SEEDS=(17 42 43 100 911 1000 4999 7919 40000 100000)
# SEEDS=(17 42 100 911 1000 4999 100000)

# TODO: best performing seeds and worst performing seeds.
# worst: 100, 1000, 100000

SCORES_FILE="${TARGET_DIR}/eval_scores.txt"
mkdir -p "$TARGET_DIR"  # Ensure directory exists
echo "checkpoint: ${CHECKPOINT_PATH}" > "$SCORES_FILE"

# Array to store scores
SCORE_LIST=()

for SEED in "${SEEDS[@]}"; do
    OUTPUT_DIR="${TARGET_DIR}/seed${SEED}"
    echo "Running evaluation with seed ${SEED}, output will be saved to ${OUTPUT_DIR}"


    if [ -z "$DATASET_PATH" ]; then
        echo "DATASET_PATH is empty"
        python eval.py \
            --checkpoint "$CHECKPOINT_PATH" \
            -o "$OUTPUT_DIR" \
            --seed "$SEED" \
            --save_rollout
    else
        echo "using DATASET_PATH: $DATASET_PATH"
        python eval.py \
            --checkpoint "$CHECKPOINT_PATH" \
            -o "$OUTPUT_DIR" \
            --seed "$SEED" \
            --dataset_path "$DATASET_PATH" \
            --save_rollout
    fi




    SCORE=$(jq -r '."test/mean_score"' "${OUTPUT_DIR}/eval_log.json")
    
    echo "seed ${SEED}, mean_score ${SCORE}" >> "$SCORES_FILE"
    SCORE_LIST+=("$SCORE")
done

# Compute average and std
stats=$(printf "%s\n" "${SCORE_LIST[@]}" | awk '
    {
        x[NR]=$1; sum+=$1
    }
    END {
        mean = sum/NR
        for (i=1; i<=NR; i++) {
            sq_diff += (x[i] - mean)^2
        }
        std = sqrt(sq_diff/NR)
        printf "average, %.15f\nstd, %.15f\n", mean, std
    }
')

echo "$stats" >> "$SCORES_FILE"

echo "All evaluations completed. Summary saved to ${SCORES_FILE}"


# ./eval_checkpoint.sh /home/ns1254/diffusion_policy/data/dp_logs/can_bc_mh_img_better/checkpoints/epoch=0020-test_mean_score=0.840.ckpt
