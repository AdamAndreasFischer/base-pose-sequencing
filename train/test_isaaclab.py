import argparse
import math
import warnings
from isaaclab.app import AppLauncher
from omegaconf import OmegaConf, DictConfig
import hydra
import time
import os
import traceback
parser = argparse.ArgumentParser(description="Testfile for running isaacLab")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs")
#parser.add_argument("--enable_cameras")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
import time
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from gymnasium.spaces import Box, Dict
import numpy as np

import random
from datetime import datetime
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize, VecTransposeImage
from gymnasium.spaces import Box

from isaaclab.envs import (
    DirectMARLEnv,
    ManagerBasedRLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
print("################# PRE IMPORT ######################")
from base_pose_sequencing.task.task_isaac_lab import BasePosePlanningEnvCfg
print("POST IMPORT")


#### Inverse kinematics controller: https://isaac-sim.github.io/IsaacLab/v2.0.0/source/tutorials/05_controllers/run_diff_ik.html 


def make_env(cfg):

    env = ManagerBasedRLEnv(cfg=cfg)
    return env
print("After make_env")

#@hydra.main(version_base=None, config_path="/home/adamfi/codes/base-pose-sequencing/conf", config_name="config")
print("Pre main")
def main():
    #config = OmegaConf.to_yaml(cfg_task)
    try:
        print("Preprint conf")
        root_path = "/home/adamfi/codes/"
        #print(BasePosePlanningEnvCfg(root_path = "/home/adamfi/codes/"))
        env_cfg = BasePosePlanningEnvCfg(root_path = "/home/adamfi/codes/")
        env_cfg.sim.device = args_cli.device
        env_cfg.root_path = root_path
        n_env = args_cli.num_envs
        env_cfg.scene.num_envs = n_env

        print("Env CFG initialized")
        env_RL = make_env(env_cfg)
        #env = gym.make("Base pose planning", cfg = env_cfg)
        #print(dir(env))
        scene = env_RL.scene

        env = Sb3VecEnvWrapper(env_RL)
        print("Dir of SB3 Environment")
        print(dir(env))
        print("Observation space ################################")
        print(env.observation_space)


        # Initialize observation space and set it to int
        if isinstance(env.observation_space, Dict) and len(env.observation_space.keys())==2:
            # ensure dtype is uint8 for image detection
            for key in env.observation_space.keys():

                if env.observation_space[key].dtype != np.uint8:
                    env.observation_space[key] = Box(
                        low=0,
                        high=255,
                        shape=env.observation_space[key].shape,
                        dtype=np.uint8
                    )
            #env = VecTransposeImage(env)

        print(env.observation_space)

        for i in range(300):

            x = np.zeros((n_env, 1))
            y = np.ones((n_env, 1))

            theta = np.ones((n_env, 1))*-np.pi/2 

            action = np.concatenate((x,y,theta),axis=1)

            action_torch = torch.from_numpy(action).to(args_cli.device) # This is valid action space shape


            #env.SceneEntityCfg("camera")["rgb"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                env.step(torch.tensor(action_torch))

            #for _ in range(15):
            #    env_RL.sim.render()
            #env_RL.sim.step()
            #print(scene["camera"].data.output)
            #env_RL.sim.render(1)
            #rgb = scene["camera"].data.output["rgb"][0].cpu().numpy()
            #depth = scene["camera"].data.output["distance_to_image_plane"][0].cpu().numpy()
            #print(rgb.shape)
        #
            #fig, (ax_rgb, ax_depth) = plt.subplots(1, 2, figsize=(10, 4))
            #ax_rgb.imshow(rgb)
            #ax_rgb.set_title("RGB")
            #ax_rgb.axis("off")
        #
            #depth_plot = ax_depth.imshow(depth, cmap="viridis")  # “viridis” colormap
            #ax_depth.set_title("Depth")
            #ax_depth.axis("off")
            #fig.colorbar(depth_plot, ax=ax_depth, fraction=0.046, pad=0.04)
        #
            #plt.tight_layout()
            #plt.show()



        simulation_app.close()

    except KeyboardInterrupt: 
        print("Registered Keyboard interuption....")
        print("Shutind down....")
        simulation_app.close()
    except Exception as e:
        print(f"Simulator crashed with error: {e}....")
        print(traceback.format_exc())
        print("Shuting down....")
        simulation_app.close()
if __name__ == "__main__":

    main()