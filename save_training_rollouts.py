# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time 
import tqdm
import h5py
import json 
import robomimic.envs.env_base as EB
import yaml
import shutil
import argparse


# training_path = "/root/diffusion_policy/data/outputs/2025.05.15/20.56.04_train_diffusion_unet_lowdim_lift_lowdim"
# training_path = "/root/diffusion_policy/data/outputs/2025.05.18/01.15.16_train_diffusion_unet_hybrid_pusht_image"


def load_rollout(rollout_path):
    rollout = np.load(rollout_path, allow_pickle=True).item()
    states= np.array(rollout['states'])
    actions= np.array(rollout['actions']) 
    return states, actions


def load_dataset_attributes(dataset_path): 
    file=h5py.File(dataset_path, 'r') 
    key = 'env_args'
    env_args=file['data'].attrs[key]
    model_file=file['data/demo_1'].attrs["model_file"]
    file.close()
    return env_args, model_file

def rollout_to_hdf5(rollout_dir, env_args, model_file, dataset_save_dir=None):
    """ 
    rollout_dir: contains rollout files (.npy) (e.g. 56 files)
    dataset_save_dir: create a hdf5 file from all the rollouts and save inside this dir.
    """

    if dataset_save_dir==None:
        dataset_save_dir = os.path.dirname(rollout_dir)
    
    rollout_files = [rollout_dir+"/"+filename for filename in os.listdir(rollout_dir)] 

    base_name= os.path.basename(rollout_dir)  #e.g. "epoch_0"
    dataset_path_sub = dataset_save_dir+f"/{base_name}.hdf5"
    f_sub = h5py.File(dataset_path_sub, "w") 
    grp = f_sub.create_group("data")
    f_sub.create_group("mask")

    #copy attributes 
    if env_args is not None:
        key = 'env_args'
        f_sub['data'].attrs[key]=env_args

    for i, rollout_path in enumerate(rollout_files):
        states, actions = load_rollout(rollout_path)
        ep_data_grp = grp.create_group(f"demo_{i}")
        
        if model_file is not None:
            ep_data_grp.attrs["model_file"] = model_file 

        if type(states[0])== dict:
            # convert to list
            images= [state['image'] for state in states]
            agent_poss= [state['agent_pos'] for state in states] 
            obs=ep_data_grp.create_group("obs")
            obs.create_dataset("images", data=images)
            obs.create_dataset("agent_poss", data=agent_poss)
            ep_data_grp.create_dataset("states", data=agent_poss)
        else:
            ep_data_grp.create_dataset("states", data=states)
        ep_data_grp.create_dataset("actions", data=actions)

    f_sub.close()


def main(training_path, dataset_path=None):
    checkpoint_path = os.path.join(training_path, "checkpoints")
    rollout_path = os.path.join(training_path, "rollouts")

    config_path = os.path.join(training_path, ".hydra/config.yaml")
    # load yaml config
    with open(config_path, 'r') as f:
        config = f.read()
    config = yaml.safe_load(config)

    dataset_type=None 
    
    if dataset_path is None:
        if 'dataset_path' in config['task']['dataset']:
            dataset_path = config['task']['dataset']['dataset_path']
        if 'zarr_path' in config['task']['dataset']:
            dataset_path = config['task']['dataset']['zarr_path'] 

    if '.hdf5' in dataset_path:
        dataset_type = 'hdf5'
    elif '.zarr' in dataset_path:
        dataset_type = 'zarr'


    checkpoints = os.listdir(checkpoint_path)
    for checkpoint in checkpoints:
        if "epoch=" in checkpoint:
            epoch = checkpoint.split("=")[1].split("-")[0]
            epoch = int(epoch)
            print('epoch', epoch)

    env_args, model_file =None, None 
    if dataset_type=='hdf5':
        env_args, model_file = load_dataset_attributes(dataset_path)
    


    rollouts = os.listdir(rollout_path)
    for rollout_name in tqdm.tqdm(rollouts): 
        rollout_dir = os.path.join(rollout_path, rollout_name)
        if os.path.isdir(rollout_dir):
            rollout_to_hdf5(rollout_dir, env_args, model_file) 
            shutil.rmtree(rollout_dir, ignore_errors=True)

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--training_path", type=str, required=True, help="Path to the training directory")
    argparse.add_argument("--dataset_path", type=str, default=None, help="Path to the dataset (if applicable)")
    args = argparse.parse_args() 
    main(args.training_path, args.dataset_path)

# python save_training_rollouts.py --training_path /root/diffusion_policy/data/outputs/2025.05.18/01.22.32_train_diffusion_unet_lowdim_pusht_lowdim

# python save_training_rollouts.py \
#     --training_path /home/ns1254/diffusion_policy/data/dp_logs/can_bc_mh_img_better\
#     --dataset_path /home/ns1254/diffusion_policy/data/robomimic/datasets/can/mh/image.hdf5
