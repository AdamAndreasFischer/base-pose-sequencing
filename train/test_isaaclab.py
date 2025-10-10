import argparse
import math

from isaaclab.app import AppLauncher
from omegaconf import OmegaConf, DictConfig
import hydra

parser = argparse.ArgumentParser(description="Testfile for running isaacLab")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel envs")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from gymnasium.spaces import Box, Dict
import numpy as np
import os
import random
from datetime import datetime
import torch
import torch.nn as nn


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

from base_pose_sequencing.task.task_isaac_lab import BasePosePlanningEnvCfg

def make_env(cfg):

    env = ManagerBasedRLEnv(cfg=cfg)
    return env

from gymnasium import ObservationWrapper

#@hydra.main(version_base=None, config_path="/home/adamfi/codes/base-pose-sequencing/conf", config_name="config")
def main():
    #config = OmegaConf.to_yaml(cfg_task)
    print("Preprint conf")
    root_path = "/home/adamfi/codes/"
    print(BasePosePlanningEnvCfg(root_path = root_path))
    env_cfg = BasePosePlanningEnvCfg(root_path = root_path)
    env_cfg.sim.device = args_cli.device
    env_cfg.root_path = root_path

    print("Env CFG initialized")
    env = make_env(env_cfg)
    print(dir(env))
    for i in range(50):

        x,y = torch.FloatTensor((1,)).uniform_(-2.5,2-5),torch.FloatTensor((1,)).uniform_(-2.5,2-5)
        theta = torch.FloatTensor((1,)).uniform_(-torch.pi, torch.pi)

        obs,_ =env.step(torch.tensor([x,y,theta]))

        