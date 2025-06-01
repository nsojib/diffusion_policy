

# TARGET_DIR="/home/ns1254/diffusion_policy/data/outputs/can_mh_img1/checkpoints/epoch=0380-test_mean_score=0.920.ckpt"
# TARGET_DIR="/home/ns1254/diffusion_policy/data/outputs/can_mh_img1/checkpoints/epoch=0280-test_mean_score=0.920.ckpt"
# TARGET_DIR="/home/ns1254/diffusion_policy/data/outputs/can_mh_img1/checkpoints/epoch=0100-test_mean_score=0.920.ckpt"
# TARGET_DIR="/home/ns1254/diffusion_policy/data/outputs/can_mh_img1/checkpoints/epoch=0180-test_mean_score=0.900.ckpt"
# TARGET_DIR="/home/ns1254/diffusion_policy/data/outputs/can_mh_img1/checkpoints/epoch=0200-test_mean_score=0.900.ckpt"
# TARGET_DIR="/home/ns1254/diffusion_policy/data/outputs/can_mh_img1/checkpoints/epoch=0240-test_mean_score=0.900.ckpt"
# TARGET_DIR="/home/ns1254/diffusion_policy/data/outputs/can_mh_img1/checkpoints/epoch=0020-test_mean_score=0.660.ckpt"
# TARGET_DIR="/home/ns1254/diffusion_policy/data/outputs/can_mh_img1/checkpoints/epoch=0040-test_mean_score=0.640.ckpt"
# TARGET_DIR="/home/ns1254/diffusion_policy/data/outputs/can_mh_img1/checkpoints/epoch=0140-test_mean_score=0.720.ckpt"
# TARGET_DIR="/home/ns1254/diffusion_policy/data/outputs/can_mh_img1/checkpoints/epoch=0260-test_mean_score=0.820.ckpt"

# ./eval_checkpoint.sh "$TARGET_DIR"

# echo "All files processed."

# sudo apt-get install jq

BASE_PATH="/home/ns1254/diffusion_policy/data/outputs/can_mh_img1/checkpoints/"
for file in "$BASE_PATH"/*.ckpt; do
    [ -e "$file" ] || continue
    # echo "Found checkpoint: $file"
    CHECKPOINT_PATH="$file"
    echo "CHECKPOINT FOUND: $CHECKPOINT_PATH"

    ./eval_checkpoint.sh "$CHECKPOINT_PATH"
done

# ./eval_checkpoint.sh /home/ns1254/diffusion_policy/data/dp_logs/can_bc_mh_img_better/checkpoints/epoch=0020-test_mean_score=0.840.ckpt /home/ns1254/diffusion_policy/data/robomimic/datasets/can/mh/image.hdf5

