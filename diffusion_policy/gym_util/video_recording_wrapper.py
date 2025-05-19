import gym
import numpy as np
from diffusion_policy.real_world.video_recorder import VideoRecorder
import time 
import os 
from pathlib import Path
import diffusion_policy

class VideoRecordingWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            video_recoder: VideoRecorder,
            mode='rgb_array',
            file_path=None,
            steps_per_render=1,
            **kwargs
        ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)
        
        self.mode = mode
        self.render_kwargs = kwargs
        self.steps_per_render = steps_per_render
        self.file_path = file_path
        self.video_recoder = video_recoder

        self.step_count = 0
        self.kwargs = {}  
        self.traj_save_path=""
        self.is_save_rollout=False
        self.is_pusht_env=False

    def set_kwargs(self, **kwargs):
        self.kwargs= kwargs 

        dirpath = self.traj_save_path.parent
        # print('--------dirpath:', dirpath)
 
        if 'epoch' in self.kwargs and 'save_rollout' in self.kwargs:
            epoch=self.kwargs['epoch']
            self.is_save_rollout=self.kwargs['save_rollout']
            if not self.is_save_rollout:
                # print(f"not save rollout, epoch: {epoch}")
                return
            
            dirpath = dirpath / f"epoch_{epoch}"
            if not os.path.exists(dirpath):
                try:
                    os.makedirs(dirpath)
                    # print(f"create directory: {dirpath}")
                except OSError as e:
                    # print(f"Error creating directory {dirpath}: {e}")
                    pass 

    def set_traj_save_path(self, traj_save_path):
        self.traj_save_path=traj_save_path
        # print(f"set traj save path: {self.traj_save_path}=======") 



    def stop_now(self):
        # print('--------stop now signal received--------')
        if len(self.traj)>0: 
            if 'epoch' in self.kwargs and self.is_save_rollout:
                epoch=self.kwargs['epoch']
                is_save_rollout=self.kwargs['save_rollout']
                if not is_save_rollout:
                    # print(f"not save rollout, epoch: {epoch}")
                    return
                len_sa=len(self.traj['actions']) 

                dirpath = self.traj_save_path.parent / f"epoch_{epoch}"
                rname = self.traj_save_path.name
                sa_filename = dirpath / f"rollout_{rname}_{len_sa}.npy"
                # print(f"save sa to: {sa_filename}") 
                np.save(sa_filename, self.traj)


    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.frames = list()
        self.step_count = 1
        self.video_recoder.stop()
 
        if str(self.env.__class__).find('robomimic') != -1:
            self.is_pusht_env=False
        else:
            self.is_pusht_env=True

        if self.is_pusht_env:
            state_dict = {'states': obs}
        else:
            state_dict = self.env.env.get_state()
        self.traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)

        # print(f"video recording wrapper reset kwargs: {self.kwargs} {self.file_path}")
        if self.file_path is not None and self.kwargs is not None:
            if 'epoch' in self.kwargs:
                file_path_without_ext = self.file_path.split('.mp4')[0] 
                file_path_with_ext = f"{file_path_without_ext}_epoch_{self.kwargs['epoch']}.mp4"
                self.file_path = file_path_with_ext
                # print(f"video recording wrapper reset file_path: {self.file_path}")

        return obs
    
    # obs, reward, done, info = env.step(env_action)
    def step(self, action):

        if self.is_save_rollout:
            if self.is_pusht_env:
                obs = self.env._get_obs()
                state_dict = {'states': obs}
            else:
                state_dict = self.env.env.get_state()

            # state_dict = self.env.env.get_state() 

            self.traj['states'].append(state_dict['states'])
            self.traj['actions'].append(action)

        result = super().step(action)

        if self.is_save_rollout:
            self.traj['rewards'].append(result[1])
            self.traj['dones'].append(result[2])
            
        if result[2]:
            print(f'----------end of episode-------{self.kwargs} {self.file_path}---')

        self.step_count += 1
        if self.file_path is not None \
            and ((self.step_count % self.steps_per_render) == 0):
            if not self.video_recoder.is_ready():
                self.video_recoder.start(self.file_path)

            frame = self.env.render(
                mode=self.mode, **self.render_kwargs)
            assert frame.dtype == np.uint8
            self.video_recoder.write_frame(frame)
        return result
    
    def render(self, mode='rgb_array', **kwargs):
        if self.video_recoder.is_ready():
            self.video_recoder.stop()
        return self.file_path
