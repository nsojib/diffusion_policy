"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output

python eval.py --checkpoint /home/ns1254/diffusion_policy/data/outputs/official/epoch=2000-test_mean_score=1.000.ckpt -o data/can_off_eval_output


python eval.py \
    --checkpoint /home/ns1254/diffusion_policy/data/outputs/can_mh_img1/checkpoints/epoch=0220-test_mean_score=0.860.ckpt\
    -o data/can_mh_img1_eval_is_output_220 \
    -istates /home/ns1254/diffusion_policy/init_states/init_states_can_mh_image_abs_300.npz \
    -n 100 \
    --save_rollout


python eval.py \
    --checkpoint /home/ns1254/diffusion_policy/data/dp_logs/can_bc_mh_img_better/checkpoints/epoch=0020-test_mean_score=0.840.ckpt\
    -o data/can_bc_mh_img_better \
    --dataset_path /home/ns1254/diffusion_policy/data/robomimic/datasets/can/mh/image.hdf5 \
    --save_rollout



"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import numpy as np

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('-s', '--seed', default=None, type=int)
@click.option('-save_rollout', '--save_rollout', default=False, is_flag=True)
@click.option('-istates', '--istates', default=None, type=str)
@click.option('-n', '--n_envs', default=None, type=int)
@click.option('--dataset_path', default=None, type=str, help='Path to dataset if needed')
def main(checkpoint, output_dir, device, seed, save_rollout, istates, n_envs, dataset_path):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    
    if seed is not None:
        cfg.task.env_runner.test_start_seed = seed 
    
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if hasattr(cfg.training, "use_ema"):
        if cfg.training.use_ema:
            policy = workspace.ema_model
    else:
        # print('not exist')
        pass
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # dataset_path = cfg.task.env_runner.dataset_path
    cfg_te={key:value for key,value in cfg.task.env_runner.items()}
    if dataset_path is not None:
        cfg_te['dataset_path'] = dataset_path

    if istates is not None: 
        init_states = np.load(istates)['init_states'] 
        cfg_te['init_states'] = init_states

    if n_envs is not None:
        cfg_te['n_envs']=n_envs 

    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg_te,
        output_dir=output_dir)
    

    kwargs={'epoch':'inference', 'save_rollout':save_rollout}
    runner_log = env_runner.run(policy, kwargs=kwargs)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
