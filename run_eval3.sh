#!/bin/bash


TASK="can_mh"
CHECKPOINT_PATH="data/outputs/2025.04.27/01.10.52_train_diffusion_unet_hybrid_can_image/checkpoints/epoch=0200-test_mean_score=1.000.ckpt"


BASE_OUTPUT_DIR="data/{$TASK}_eval_output"
SEEDS=(42 100 1000)

# Loop through each seed and run the command
for SEED in "${SEEDS[@]}"; do
    OUTPUT_DIR="${BASE_OUTPUT_DIR}_seed${SEED}"  # Append seed to output dir
    echo "Running evaluation with seed ${SEED}, output will be saved to ${OUTPUT_DIR}"
    
    python eval.py \
        --checkpoint "$CHECKPOINT_PATH" \
        -o "$OUTPUT_DIR" \
        --seed "$SEED"
done

echo "All evaluations completed."
