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
import os 
import json 

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

    #---------------remove demos: BED mask-------------------- 
    mask_fn= None
    if hasattr(cfg, 'mask_fn'):
        mask_fn = cfg.mask_fn
        print("mask_fn", mask_fn)
        if os.path.exists(mask_fn):
            print(f"Mask file {mask_fn} exists, will use it to remove demos.")
        else:
            print(f"Mask file {mask_fn} does not exist, will not use it to remove demos.")
            mask_fn = None
    
    remove_demos = []
    if mask_fn is not None:
        with open(mask_fn, 'r') as f:
            remove_demos = [line.strip() for line in f.readlines()]
    print("remove_demos=", remove_demos)

    #---------------OR remove segments: GiB mask--------------------
    #note: the data needs to have add_uids.py
    segments_toremove_file = ""
    if hasattr(cfg, 'segments_toremove_file'):
        segments_toremove_file = cfg.segments_toremove_file
        if segments_toremove_file is None:
            segments_toremove_file=""
        
    if os.path.exists(segments_toremove_file):
        print('using segs file: ', segments_toremove_file)
        with open(segments_toremove_file, 'r') as f:
            data = json.load(f) 
        segs_toremove = data['data'] 
    else:
        segs_toremove = {}  
    
    print(f"segs_toremove: {segs_toremove}") 

    # if len(segs_toremove)>1:
    #     print('Fresh loading without cache...')
    #     cfg.task.dataset.use_cache = False

    # workspace: BaseWorkspace = cls(cfg, segs_toremove=segs_toremove)
    # workspace.run() 

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    # workspace.run()
    workspace.run(save_rollout=save_rollout, remove_demos=remove_demos, segs_toremove=segs_toremove)

if __name__ == "__main__":
    main()


# python train.py --config-dir="configs" \
#     --config-name=image_pusht_diffusion_policy_cnn.yaml \
#     training.seed=42 \
#     training.device=cuda:0 \
#     hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'\
#     checkpoint.topk.k=20\
#     training.num_epochs=401 \
#     training.checkpoint_every=20 \
#     training.rollout_every=20 \
#     +save_rollout=true



# python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'


# python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' +save_rollout=true




# python train.py --config-dir=. --config-name=config_lowdim_lift_mh.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
# python train.py --config-dir=. --config-name=config_image_lift_mh.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# python train.py --config-dir=. --config-name=config_image_lift_mh.yaml +task.dataset.hdf5_filter_key="worse" training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# python train.py --config-dir="configs" --config-name=config_lowdim_lift10e_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'


# python train.py --config-dir="configs" --config-name=config_lowdim_can30e_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'



#all 300
# /home/carl_lab/ns/diffusion_policy/data/outputs/2025.06.10/20.14.20_train_robomimic_lowdim_can_lowdim

#deminf remove 0.8
# python train.py --config-dir="configs" --config-name=config_lowdim_can_mh_bc.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' +mask_fn=/home/carl_lab/ns/diffusion_policy/data/can_mh_deminf_masked_0_0.78_60.txt
# /home/carl_lab/ns/diffusion_policy/data/outputs/2025.06.10/17.21.38_train_robomimic_lowdim_can_lowdim


# bed remove 0.8
# python train.py --config-dir="configs" --config-name=config_lowdim_can_mh_bc.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' +mask_fn=/home/carl_lab/ns/diffusion_policy/data/can_mh_bed_masked_0_60.txt


#gib
# python train.py --config-dir="configs" --config-name=config_lowdim_can_mh_bc.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' +mask_fn=/home/carl_lab/ns/diffusion_policy/data/can_mh_bed_masked_0_60.txt +segments_toremove_file="/home/ns1254/gib/gib_results/gib_can_mh_image/subtask_rm120/segs_index_rm120_gib_can_mh_image.json"



#image
# python train.py --config-dir="configs" --config-name=config_image_can_mh_bc.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# python train.py --config-dir="configs" --config-name=config_image_can_mh_bc.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' +mask_fn=/home/carl_lab/ns/diffusion_policy/data/can_mh_deminf_masked_0_0.78_60.txt

