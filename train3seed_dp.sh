#!/bin/bash
set -e  # stop if any command fails

# GPU to use
export CUDA_VISIBLE_DEVICES=0

# List of seeds to run
SEEDS=(42 43 44)
now=$(date +%Y.%m.%d-%H.%M.%S)

# HDF5_FILTER_KEY="train"

# # Loop through seeds
# for SEED in "${SEEDS[@]}"; do
#     echo "=============================="
#     echo "Running training with seed $SEED"
#     echo "=============================="
 

#     python train.py --config-dir="configs/low_dim/lift_mh/diffusion_policy_cnn" \
#         --config-name=config.yaml \
#         training.seed=$SEED \
#         training.device=cuda:0 \
#         hydra.run.dir="data/outputs/${TASK}_${HDF5_FILTER_KEY}_${now}/${SEED}"  \
#         +task.dataset.hdf5_filter_key=$HDF5_FILTER_KEY \
#         training.num_epochs=1201 &

#     sleep 40s 

#     echo "✅ Started seed $SEED"
# done

DATASET_PATH="/workspaces/dp/devcontainer/diffusion_policy/data/robomimic/datasets/wob30/lift_low_dim_abs_wob30.hdf5"

# Loop through seeds
for SEED in "${SEEDS[@]}"; do
    echo "=============================="
    echo "Running training with seed $SEED"
    echo "=============================="
 

    python train.py --config-dir="configs/low_dim/square_mh/diffusion_policy_cnn" \
        --config-name=config.yaml \
        training.seed=$SEED \
        training.device=cuda:0 \
        hydra.run.dir="data/outputs/${TASK}_${now}/${SEED}"  \
        task.dataset.dataset_path=$DATASET_PATH \
        task.dataset_path=$DATASET_PATH \
        task.env_runner.dataset_path=$DATASET_PATH \
        +task.dataset.mask_files=['masked_files/small30/oracle_rm_worse10.txt'] \
        training.num_epochs=1201 

    sleep 40s 

    echo "✅ Started seed $SEED"
done

# # Loop through seeds
# for SEED in "${SEEDS[@]}"; do
#     echo "=============================="
#     echo "Running training with seed $SEED"
#     echo "=============================="
 

#     python train.py --config-dir="configs/low_dim/lift_mh/diffusion_policy_cnn" \
#         --config-name=config.yaml \
#         training.seed=$SEED \
#         training.device=cuda:0 \
#         hydra.run.dir="data/outputs/${TASK}_${now}/${SEED}"  \
#         task.dataset.dataset_path=$DATASET_PATH \
#         task.dataset_path=$DATASET_PATH \
#         task.env_runner.dataset_path=$DATASET_PATH \
#         training.num_epochs=1201 &

#     sleep 40s 

#     echo "✅ Started seed $SEED"
# done

