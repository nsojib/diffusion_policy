# %%
import sys
# use line-buffering for both stdout and stderr
# sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
# sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import os
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace
 
import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
# from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.robomimic_lowdim_policy import RobomimicLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to

from diffusion_policy.policy.robomimic_lowdim_policy import RobomimicLowdimPolicy
from diffusion_policy.policy.robomimic_image_policy import RobomimicImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

# from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace
import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Sampler 

import h5py

# import mimicgen
# import mimicgen.utils.file_utils as MG_FileUtils
# import mimicgen.utils.robomimic_utils as RobomimicUtils
# from mimicgen.utils.misc_utils import add_red_border_to_frame
# from mimicgen.configs import MG_TaskSpec

OmegaConf.register_new_resolver("eval", eval, replace=True)

# %%
config_path='.'
# config_name = 'image_square_mh_diffusion_policy_cnn_worse_uid.yaml'
config_name = "image_square_mh_diffusion_policy_cnn_g40b30.yaml"

# %%
with initialize(version_base=None, config_path=config_path):
    cfg_org = compose(
        config_name=config_name,
        overrides=[
            "hydra.run.dir=data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}",
            "training.seed=42",
            "training.device=cuda:0"
        ],
    )
    print(cfg_org)
    
OmegaConf.resolve(cfg_org)

print('resume: ', cfg_org.training.resume)

# %%
# last_checkpoint_dir = "/home/carl_lab/diffusion_policy/data/outputs/2024.12.13/03.05.17_train_diffusion_unet_hybrid_square_image/"
last_checkpoint_dir = None 

# %%
class TrainDiffusionUnetHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

# %%


# %% [markdown]
# ### recreating workspace

# %%
output_dir = "/home/carl_lab/diffusion_policy/data/outputs/custom"
workspace = TrainDiffusionUnetHybridWorkspace(cfg_org, output_dir=output_dir)

self = workspace

# %%


# %%


# %%
cfg = copy.deepcopy(self.cfg)

# resume training
# if cfg.training.resume:
#     lastest_ckpt_path = self.get_checkpoint_path()
#     if lastest_ckpt_path.is_file():
#         print(f"Resuming from checkpoint {lastest_ckpt_path}")
#         self.load_checkpoint(path=lastest_ckpt_path)

# %%


# %%
# new_config = {key: value for key, value in cfg.task.dataset.items() if key != '_target_'}
# dataset = RobomimicReplayImageDataset(**new_config)
# len(dataset)

# %%
new_config = OmegaConf.to_container(cfg.task.dataset, resolve=True )
del new_config['_target_']
new_config['shape_meta']['obs']['demo_no']={'shape':[] }
new_config['shape_meta']['obs']['index_in_demo']={'shape':[] }
# new_config['use_cache'] = False
 

# %%
new_config

# %%
# dataset = RobomimicReplayImageDataset(**new_config, hdf5_filter_key = "g40b30")
dataset = RobomimicReplayImageDataset(**new_config)
len(dataset)

# %%
dt = dataset.__getitem__(100)
dt['action'].shape, dt['obs']['agentview_image'].shape

# %%
dt['obs']['demo_no'], dt['obs']['index_in_demo']

# %%
# hdf5_filter_key

# %%
# segs_toremove={
#  'demo_0': [ (5,20), (30, 40) ],
#  'demo_1': [ (12, 20) ],
#  'demo_12': [(0, 10), (15, 20), (30,33) ],
# }

# %%
# segs_toremove={
#  'demo_3': [(199, 730)],
#  'demo_4': [(176, 544)],
#  'demo_11': [(200, 488)],
#  'demo_12': [(0, 404), (405, 780)],
#  'demo_13': [(0, 163), (164, 602)],
#  'demo_14': [(0, 107), (108, 993)],
#  'demo_20': [(177, 979)],
#  'demo_21': [(162, 673)],
#  'demo_26': [(68, 617)],
#  'demo_27': [(211, 662)],
#  'demo_28': [(0, 160), (161, 1014)],
#  'demo_29': [(218, 1120)],
#  'demo_32': [(166, 753)],
#  'demo_34': [(191, 968)],
#  'demo_35': [(171, 1134)],
#  'demo_36': [(153, 735)],
#  'demo_37': [(162, 594)],
#  'demo_100': [(191, 354)],
#  'demo_101': [(0, 321)],
#  'demo_102': [(0, 581)],
#  'demo_103': [(0, 505)],
#  'demo_104': [(0, 412), (413, 644)],
#  'demo_105': [(0, 362), (363, 590)],
#  'demo_106': [(0, 296)],
#  'demo_107': [(58, 715)],
#  'demo_108': [(0, 462)],
#  'demo_109': [(0, 349)],
#  'demo_110': [(0, 476)],
#  'demo_111': [(0, 503)],
#  'demo_112': [(0, 853), (854, 1083)],
#  'demo_113': [(0, 535), (536, 727)],
#  'demo_42': [(213, 435)]}

#new indexing in the g40b30 file
segs_toremove= {
 'demo_38': [(199, 730)],
 'demo_47': [(176, 544)],
 'demo_2': [(200, 488)],
 'demo_24': [(0, 404), (405, 780)],
 'demo_25': [(0, 163), (164, 602)],
 'demo_26': [(0, 107), (108, 993)],
 'demo_31': [(177, 979)],
 'demo_32': [(162, 673)],
 'demo_34': [(68, 617)],
 'demo_35': [(211, 662)],
 'demo_36': [(0, 160), (161, 1014)],
 'demo_37': [(218, 1120)],
 'demo_40': [(166, 753)],
 'demo_41': [(191, 968)],
 'demo_42': [(171, 1134)],
 'demo_43': [(153, 735)],
 'demo_44': [(162, 594)],
 'demo_10': [(191, 354)],
 'demo_11': [(0, 321)],
 'demo_12': [(0, 581)],
 'demo_13': [(0, 505)],
 'demo_14': [(0, 412), (413, 644)],
 'demo_15': [(0, 362), (363, 590)],
 'demo_16': [(0, 296)],
 'demo_17': [(58, 715)],
 'demo_18': [(0, 462)],
 'demo_19': [(0, 349)],
 'demo_20': [(0, 476)],
 'demo_21': [(0, 503)],
 'demo_22': [(0, 853), (854, 1083)],
 'demo_23': [(0, 535), (536, 727)],
 'demo_49': [(213, 435)]}


# %%
remove_ids={}
for key in segs_toremove.keys():
    segs = segs_toremove[key]
    ids = [] 
    for start, end in segs:
        ids.extend(range(start, end + 1))  # Include the end value

    demo_no = int(key.split("_")[1])
    remove_ids[demo_no]=ids

# %%
# remove_ids[112]

# %%


# %%
valid_indices =[]  #in the dataset.

for index in tqdm.tqdm( range(len(dataset)) ):
    data = dataset.__getitem__(index)
    demo_no =int( data['obs']['demo_no'][0].item() )
    indices_in_demo = data['obs']['index_in_demo'].numpy().astype(int)
 
    should_remove = False
    if demo_no in remove_ids:
        should_remove = bool(set(remove_ids[demo_no]) & set(indices_in_demo))
    if should_remove: continue 
    valid_indices.append(index)

len(valid_indices)

# %%


# %%
class CustomIndicesSampler(Sampler):
    def __init__(self, custom_indices):
        self.custom_indices = np.random.permutation(custom_indices)

    def __iter__(self): 
        return iter(self.custom_indices)

    def __len__(self):
        return len( self.custom_indices )

# %%
sampler = CustomIndicesSampler(valid_indices)

# %%
del dataset

# %%
new_config = OmegaConf.to_container(cfg.task.dataset, resolve=True )
del new_config['_target_']
# new_config['shape_meta']['obs']['demo_no']={'shape':[] }
# new_config['shape_meta']['obs']['index_in_demo']={'shape':[] }


dataset = RobomimicReplayImageDataset(**new_config)
len(dataset)

# %%
cfg_dataloader = {key:value for key,value in cfg.dataloader.items()}
cfg_dataloader['shuffle']=False

# %%
train_dataloader = DataLoader(dataset, **cfg_dataloader, sampler=sampler)
normalizer = dataset.get_normalizer()

# configure validation dataset
val_dataset = dataset.get_validation_dataset()
val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

# %%


# %%
batch =  next(iter(train_dataloader))
batch.keys()

# %%
batch['action'].shape

# %%
batch['obs']['agentview_image'].shape

# %%
self.model.set_normalizer(normalizer)
if cfg.training.use_ema:
    self.ema_model.set_normalizer(normalizer)

# configure lr scheduler
lr_scheduler = get_scheduler(
    cfg.training.lr_scheduler,
    optimizer=self.optimizer,
    num_warmup_steps=cfg.training.lr_warmup_steps,
    num_training_steps=(
        len(train_dataloader) * cfg.training.num_epochs) \
            // cfg.training.gradient_accumulate_every,
    # pytorch assumes stepping LRScheduler every epoch
    # however huggingface diffusers steps it every batch
    last_epoch=self.global_step-1
)

# configure ema
ema: EMAModel = None
if cfg.training.use_ema:
    ema = hydra.utils.instantiate(
        cfg.ema,
        model=self.ema_model)

# %%
env_runner: BaseImageRunner
env_runner = hydra.utils.instantiate(
    cfg.task.env_runner,
    output_dir=self.output_dir)
assert isinstance(env_runner, BaseImageRunner)

# %%
 

# %%
topk_manager = TopKCheckpointManager(
    save_dir=os.path.join(self.output_dir, 'checkpoints'),
    **cfg.checkpoint.topk
)

# device transfer
device = torch.device(cfg.training.device)
self.model.to(device)
if self.ema_model is not None:
    self.ema_model.to(device)
optimizer_to(self.optimizer, device)

# %%


# %%
train_sampling_batch = None
log_path = os.path.join(self.output_dir, 'logs.json.txt')
with JsonLogger(log_path) as json_logger:
    for local_epoch_idx in range(cfg.training.num_epochs):
        step_log = dict()
        # ========= train for this epoch ==========
        train_losses = list()
        with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                # device transfer
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                if train_sampling_batch is None:
                    train_sampling_batch = batch

                # compute loss
                raw_loss = self.model.compute_loss(batch)
                loss = raw_loss / cfg.training.gradient_accumulate_every
                loss.backward()

                # step optimizer
                if self.global_step % cfg.training.gradient_accumulate_every == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    lr_scheduler.step()
                
                # update ema
                if cfg.training.use_ema:
                    ema.step(self.model)

                # logging
                raw_loss_cpu = raw_loss.item()
                tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                train_losses.append(raw_loss_cpu)
                step_log = {
                    'train_loss': raw_loss_cpu,
                    'global_step': self.global_step,
                    'epoch': self.epoch,
                    'lr': lr_scheduler.get_last_lr()[0]
                }

                is_last_batch = (batch_idx == (len(train_dataloader)-1))
                if not is_last_batch:
                    # log of last step is combined with validation and rollout
                     
                    json_logger.log(step_log)
                    self.global_step += 1

                if (cfg.training.max_train_steps is not None) \
                    and batch_idx >= (cfg.training.max_train_steps-1):
                    break

        # at the end of each epoch
        # replace train_loss with epoch average
        train_loss = np.mean(train_losses)
        step_log['train_loss'] = train_loss

        # ========= eval for this epoch ==========
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()

        # run rollout
        if (self.epoch % cfg.training.rollout_every) == 0:
            runner_log = env_runner.run(policy)
            # log all
            step_log.update(runner_log)

        # run validation
        if (self.epoch % cfg.training.val_every) == 0:
            with torch.no_grad():
                val_losses = list()
                with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        loss = self.model.compute_loss(batch)
                        val_losses.append(loss)
                        if (cfg.training.max_val_steps is not None) \
                            and batch_idx >= (cfg.training.max_val_steps-1):
                            break
                if len(val_losses) > 0:
                    val_loss = torch.mean(torch.tensor(val_losses)).item()
                    # log epoch average validation loss
                    step_log['val_loss'] = val_loss

        # run diffusion sampling on a training batch
        if (self.epoch % cfg.training.sample_every) == 0:
            with torch.no_grad():
                # sample trajectory from training set, and evaluate difference
                batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                obs_dict = batch['obs']
                gt_action = batch['action']
                
                result = policy.predict_action(obs_dict)
                pred_action = result['action_pred']
                mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                step_log['train_action_mse_error'] = mse.item()
                del batch
                del obs_dict
                del gt_action
                del result
                del pred_action
                del mse
        
        # checkpoint
        if (self.epoch % cfg.training.checkpoint_every) == 0:
            # checkpointing
            if cfg.checkpoint.save_last_ckpt:
                self.save_checkpoint()
            if cfg.checkpoint.save_last_snapshot:
                self.save_snapshot()

            # sanitize metric names
            metric_dict = dict()
            for key, value in step_log.items():
                new_key = key.replace('/', '_')
                metric_dict[new_key] = value
            
            # We can't copy the last checkpoint here
            # since save_checkpoint uses threads.
            # therefore at this point the file might have been empty!
            topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

            if topk_ckpt_path is not None:
                self.save_checkpoint(path=topk_ckpt_path)
        # ========= eval end for this epoch ==========
        policy.train()

        # end of epoch
        # log of last step is combined with validation and rollout
         
        json_logger.log(step_log)
        self.global_step += 1
        self.epoch += 1



# /home/carl_lab/diffusion_policy/data/outputs/custom/checkpoints/epoch=0500-test_mean_score=0.160.ckpt