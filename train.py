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

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
    
    
 # python train.py --config-dir=. --config-name=image_square_mh_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
  
 # python train.py --config-dir=. --config-name=image_can_20ptrain_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
  
 # python train.py --config-dir=. --config-name=image_lift_20p_train_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
  

# python train.py --config-dir=. --config-name=image_square_mh_diffusion_policy_cnn_worse.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
#   /home/carl_lab/diffusion_policy/data/outputs/2024.12.13/03.05.17_train_diffusion_unet_hybrid_square_image/checkpoints/epoch=0200-test_mean_score=0.640.ckpt

# resume training
# python train.py --config-dir=. --config-name=image_square_mh_diffusion_policy_cnn_worse.yaml training.seed=42 training.device=cuda:0 hydra.run.dir="/home/carl_lab/diffusion_policy/data/outputs/2024.12.13/03.05.17_train_diffusion_unet_hybrid_square_image"
 
  # python train.py --config-dir=. --config-name=config_image_lift_mh.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
  
  # python train.py --config-dir="configs" --config-name=config_lowdim_can_30.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
  