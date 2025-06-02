
ROLLOUT_DIR="/home/ns1254/diffusion_policy/data/dp_logs/can_bc_mh_img_better/rollouts"

for file in "$ROLLOUT_DIR"/*.hdf5; do
    [ -e "$file" ] || continue
    # echo "Found checkpoint: $file"
    RAW_PATH="$file"
    echo "RAW_PATH FOUND: $RAW_PATH"

    LOWDIM_PATH="${RAW_PATH%.hdf5}_low_dim.hdf5"
    echo "LOWDIM_PATH FOUND: $LOWDIM_PATH"

    python diffusion_policy/scripts/dataset_states_to_obs.py --dataset $RAW_PATH --output_name $LOWDIM_PATH --done_mode 2

    # remove the original file
    rm "$RAW_PATH"
    echo ""
done
