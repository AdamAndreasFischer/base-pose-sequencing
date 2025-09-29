# python libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import trange
import sys
import random
import argparse
import datetime
import pathlib
import time
# from pympler import asizeof


import hydra
from omegaconf import DictConfig, OmegaConf
import os
import tracemalloc


from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import *
from mushroom_rl.rl_utils.parameters import LinearParameter, Parameter
from mushroom_rl.rl_utils.replay_memory import PrioritizedReplayMemory, ReplayMemory
from mushroom_rl.algorithms.policy_search.policy_gradient import REINFORCE
from mushroom_rl.policy import BoltzmannTorchPolicy, Policy
# from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils import LinearParameter

from base_pose_sequencing.task.task import Task
from visual_mm_planning.algorithms.sac_discrete_etn import SAC
from visual_mm_planning.algorithms.dqn_etn import AbstractDQN
from visual_mm_planning.networks.optimize_base_pose_etn import ActorNetwork, CriticNetwork, DuelingQNetwork




def test(cfg):
    print(cfg.task.train.save_dir)
    mdp = Task(cfg)
    x = 1
    y = 1
    theta = np.pi
    mdp.move_robot(x, y, theta, yumi_joint_angles=None)
    frame_segment = mdp.camera.get_current_frame()["semantic_segmentation"]
    frame = mdp.camera.get_current_frame()
    print(frame)
    print(frame_segment)
    for i in range(50):
        x = np.random.randint(-2,2)
        y = np.random.randint(-2,2)
        mdp.move_robot(x, y, theta, yumi_joint_angles=None)
        mdp.move_both_arms(mdp.yumi_default_joint_angles)
        mdp.world.step(render=cfg.render)
        mdp._get_reward()
        frame = mdp.camera.get_current_frame()
        print(frame)
        mdp.reset()
        time.sleep(0.5)
    
    
    

    mdp.shutdown()



@hydra.main(version_base=None, config_path="/home/adamfi/codes/base-pose-sequencing/conf", config_name="config")
def main(cfg : DictConfig) -> None:
    config = OmegaConf.to_yaml(cfg)
    print(config)
    
    test(cfg)

if __name__ == '__main__':
    main()
