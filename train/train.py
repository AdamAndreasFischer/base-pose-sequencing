import argparse
import math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train PPO on ObjTrackingWithRecordedBaseTrajs in IsaacLab using SB3")
parser.add_argument("--num_envs", type=int, default=10, help="Number of parallel envs")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
"""Rest everything follows."""

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
from base_pose_sequencing.networks.CNN import SimpleCNNImageExtractor


# -----------------------------------------------------------------------------
def make_env(cfg):
    """Factory for RL Games envs."""
    env = ManagerBasedRLEnv(cfg=cfg)
    return env

from gymnasium import ObservationWrapper


def main():
    # configure env
    env_cfg = BasePosePlanningEnvCfg(root_path="/home/adamfi/codes/")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.root_path = "/home/adamfi/codes/"

    env = make_env(env_cfg)

    policy_kwargs = dict(
        activation_fn= nn.LeakyReLU, #Allows for some negative numbers as well
        net_arch=[dict(pi=[128, 32], vf=[128, 32])], #pi is net arch for policy network, vf is net arch for value network
        squash_output= False,
        features_extractor_class=SimpleCNNImageExtractor,
        features_extractor_kwargs=dict(features_dim=3,trans_lim=(-2.5,2.5),rot_lim=(-torch.pi, torch.pi)),
    )


    agent_cfg = {
        "seed":42,
        "n_timesteps":10000000.0,
        # "policy":"MlpPolicy",
        "policy":"CnnPolicy",
        "n_steps": 512, #8, # 16
        "batch_size": 128, #256, # 4096
        "gae_lambda":0.95,
        "gamma":0.999,
        "n_epochs": 10, #20,
        "ent_coef":0.01,
        "learning_rate":0.0003,
        "clip_range":0.2,
        "policy_kwargs": policy_kwargs,
        "vf_coef":0.5,
        "max_grad_norm":0.5,
        "device":"cuda:0",
        "normalize_input": False,
        "normalize_reward": False,
        }

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg,num_envs = env_cfg.scene.num_envs)

    # read configurations about the agent-training
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    # For custom policy networks, check: 
    # https://stable-baselines3.readthedocs.io/en/v1.0/guide/custom_policy.html 


    # wrap around environment for rl-games
    env = Sb3VecEnvWrapper(env)
    env.action_space = Box(low= np.array([-2.5, -2.5, -np.pi]), high=np.array([2.5,2.5,np.pi]))

    if isinstance(env.observation_space, Dict):
        # ensure dtype is uint8 for image detection
        for key in env.observation_space.keys():

            if env.observation_space[key].dtype != np.uint8:
                env.observation_space[key] = Box(
                    low=0,
                    high=255,
                    shape=env.observation_space[key].shape,
                    dtype=np.uint8
                )
        env = VecTransposeImage(env) # Transpose the observation to take chanels first, then W, H
    print("Env observation space: ", env.observation_space)
    normalize_input = agent_cfg.pop("normalize_input", False)
    normalize_reward = agent_cfg.pop("normalize_reward", False)

    clip_obs = agent_cfg.pop("clip_obs",np.inf)

    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join("logs", "sb3", "base_pose_planning"))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    print(f"Exact experiment name requested from command line: {run_info}")
    log_dir = os.path.join(log_root_path, run_info)
    print("In main: ", policy_arch)
    agent = PPO(policy_arch, env, verbose=1, **agent_cfg)

    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    agent.set_logger(new_logger)

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix="model", verbose=2)
    agent.learn(total_timesteps=n_timesteps, callback=checkpoint_callback)
    env.close()

if __name__=="__main__":
    main()
    simulation_app.close()