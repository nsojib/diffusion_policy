#!/bin/bash
set -e  # stop if any command fails

# GPU to use
export CUDA_VISIBLE_DEVICES=0

# List of seeds to run
SEEDS=(42 43 44)
now=$(date +%Y.%m.%d-%H.%M.%S)

TASK="lift_lowdim"

# Loop through seeds
for SEED in "${SEEDS[@]}"; do
    echo "=============================="
    echo "Running training with seed $SEED"
    echo "=============================="
 

    python train.py \
        --config-dir="diffusion_policy/config" \
        --config-name="train_robomimic_lowdim_workspace.yaml" \
        training.seed=$SEED \
        training.device="cuda:0" \
        hydra.run.dir="data/outputs/seed3_${now}/${TASK}_${SEED}" \
        task=$TASK \
        horizon=1 \
        policy.algo_name="bc" \
        task.dataset_type="mh" \
        +task.dataset.hdf5_filter_key="better" \
        training.num_epochs=1200

    echo "✅ Finished seed $SEED"
done
