wandb offline
wandb disabled


# training_path="data/outputs/pusht_img1"

# python train.py --config-dir="configs" \
#     --config-name=image_pusht_diffusion_policy_cnn.yaml \
#     training.seed=42 \
#     training.device=cuda:0 \
#     hydra.run.dir="$training_path"\
#     checkpoint.topk.k=21\
#     training.num_epochs=401 \
#     training.checkpoint_every=20 \
#     training.rollout_every=20 \
#     +save_rollout=true

# training_path="data/outputs/pusht_lowdim1"

# python train.py --config-dir="configs" \
#     --config-name=config_lowdim_pusht_cnn.yaml \
#     training.seed=42 \
#     training.device=cuda:0 \
#     hydra.run.dir="$training_path"\
#     checkpoint.topk.k=20\
#     training.num_epochs=41 \
#     training.checkpoint_every=20 \
#     training.rollout_every=20 \
#     +save_rollout=true



training_path="data/outputs/lift_mh_img1"

python train.py --config-dir="configs" \
    --config-name=config_image_lift_mh.yaml \
    training.seed=42 \
    training.device=cuda:0 \
    hydra.run.dir="$training_path"\
    checkpoint.topk.k=21\
    training.num_epochs=401 \
    training.checkpoint_every=20 \
    training.rollout_every=20 \
    +save_rollout=true \
    +task.dataset.hdf5_filter_key="worse"


python save_training_rollouts.py  --training_path "$training_path"

