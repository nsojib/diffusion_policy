"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    save_rollout = False
    if hasattr(cfg, 'save_rollout'):  
        save_rollout = cfg.save_rollout 
    print("save_rollout", save_rollout)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    # workspace.run()
    workspace.run(save_rollout=save_rollout)

if __name__ == "__main__":
    main()


# python train.py --config-dir="configs" \
#     --config-name=image_pusht_diffusion_policy_cnn.yaml \
#     training.seed=42 \
#     training.device=cuda:0 \
#     hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'\
#     checkpoint.topk.k=20\
#     training.num_epochs=400 \
#     training.checkpoint_every=20 \
#     training.rollout_every=20 \
#     +save_rollout=true



# python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'


# python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' +save_rollout=true




# python train.py --config-dir=. --config-name=config_lowdim_lift_mh.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
# python train.py --config-dir=. --config-name=config_image_lift_mh.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# python train.py --config-dir=. --config-name=config_image_lift_mh.yaml +task.dataset.hdf5_filter_key="worse" training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

