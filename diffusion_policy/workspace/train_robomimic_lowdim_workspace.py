if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

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
from diffusion_policy.policy.robomimic_lowdim_policy import RobomimicLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from torch.utils.data import DataLoader, Sampler 

OmegaConf.register_new_resolver("eval", eval, replace=True)


class CustomIndicesSampler(Sampler):
    def __init__(self, custom_indices):
        self.custom_indices = np.random.permutation(custom_indices)

    def __iter__(self): 
        return iter(self.custom_indices)

    def __len__(self):
        return len( self.custom_indices )

def parse_1_data(data):
    """ 
    data: at time t from dataset.
    #each timestamp can contain multiple uid because of obs_horizon
    """
    # if 'demo_no' not in data['obs']:
    #     raise Exception("Please add demo_no and index_in_demo to the obs first.")
    
    print('info: ', data['obs'].keys())

    demo_nos = data['obs']['demo_no']
    indices_in_demo = data['obs']['index_in_demo']
    return demo_nos, indices_in_demo


class TrainRobomimicLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: RobomimicLowdimPolicy = hydra.utils.instantiate(cfg.policy)

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self, save_rollout=False, remove_demos=[], segs_toremove={}):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseLowdimDataset
        # dataset = hydra.utils.instantiate(cfg.task.dataset)

        cfg_dataset={key: value for key, value in cfg.task.dataset.items() }
        cfg_dataset['remove_demos'] = remove_demos
        # dataset= hydra.utils.instantiate(cfg_dataset)


        self.remove_ids={}                    # all the ids to remove for the key
        self.segs_toremove = segs_toremove    #[start,end]

        for key in segs_toremove.keys():
            segs = segs_toremove[key]
            ids = [] 
            for start, end in segs:
                ids.extend(range(start, end + 1))  # Include the end value
            self.remove_ids[key]=ids



        # assert isinstance(dataset, BaseLowdimDataset)
        # train_dataloader = DataLoader(dataset, **cfg.dataloader)


        if len(self.remove_ids)==0:
            print('---------******--------full traj dataset---------******--------')
            dataset = hydra.utils.instantiate(cfg.task.dataset)
            assert isinstance(dataset, BaseLowdimDataset)
            train_dataloader = DataLoader(dataset, **cfg.dataloader) 
        else:
            print('---------++++++++--------partial traj dataset---------++++++++--------') 
            dataset = hydra.utils.instantiate(cfg.task.dataset)
            assert isinstance(dataset, BaseLowdimDataset) 

            valid_indices =[]  #in the dataset.
            print('generating valid indices ...')
            for index in tqdm.tqdm( range(len(dataset)) ):
                data = dataset.__getitem__(index)
                # demo_no, indices_in_demo = parse_1_data(data) 
                demo_no=data['demo_no']
                indices_in_demo=data['index_in_demo']

                if len(demo_no)>1:
                    assert torch.all( demo_no[0]==demo_no[1] )                 #obs history from same demo
                demo_name=f'demo_{int(demo_no[0])}'
                ids = indices_in_demo[0].numpy().astype(int)           #TODO: double check
                
                should_remove = False
                if demo_name in self.remove_ids:
                    should_remove = bool(set(self.remove_ids[demo_name]) & set(ids.tolist()))
                if should_remove: continue 
                valid_indices.append(index)

            print(f'Valid indices: {len(valid_indices)} / {len(dataset)} ')


            sampler = CustomIndicesSampler(valid_indices)
            new_config = {key:value for key,value in cfg.dataloader.items()}
            new_config['shuffle'] = False
            new_config['sampler'] = sampler 

            train_dataloader = DataLoader(dataset, **new_config) 
            # train_dataloader = DataLoader(dataset, **cfg.dataloader) 
            # -----------------end of partial traj dataloader -----------------------



        

        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)

        # configure env
        env_runner: BaseLowdimRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseLowdimRunner)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1

        # training loop
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
                        info = self.model.train_on_batch(batch, epoch=self.epoch)

                        # logging 
                        loss_cpu = info['losses']['action_loss'].item()
                        tepoch.set_postfix(loss=loss_cpu, refresh=False)
                        train_losses.append(loss_cpu)
                        step_log = {
                            'train_loss': loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
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
                self.model.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    kwargs={'epoch':self.epoch, 'save_rollout':save_rollout}
                    runner_log = env_runner.run(self.model, kwargs=kwargs)
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
                                info = self.model.train_on_batch(batch, epoch=self.epoch, validate=True)
                                loss = info['losses']['action_loss']
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

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
                self.model.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainRobomimicLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
