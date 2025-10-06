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
import matplotlib.pyplot as plt


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

#from base_pose_sequencing.task.task_torch_kinematics import Task

from base_pose_sequencing.task.task_torch_kinematics import Task as PK_Task
from visual_mm_planning.algorithms.sac_discrete_etn import SAC
from visual_mm_planning.algorithms.dqn_etn import AbstractDQN
from visual_mm_planning.networks.optimize_base_pose_etn import ActorNetwork, CriticNetwork, DuelingQNetwork

def plot_rgbd(rgb, depth):
    # Visualize RGB-D image
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot RGB image (remove alpha channel if present)
        if rgb.shape[-1] == 4:
            rgb_display = rgb[:, :, :3]  # Remove alpha channel
        else:
            rgb_display = rgb
        ax1.imshow(rgb_display)
        ax1.set_title('RGB Image')
        ax1.axis('off')
        
        # Plot Depth image
        im = ax2.imshow(depth, cmap='viridis')
        ax2.set_title('Depth Image')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, shrink=0.8)
        
        plt.tight_layout()
        plt.show()



def test(cfg):
    print(cfg.task.train.save_dir)
    mdp = PK_Task(cfg)
    x = 1
    y = 1
    theta = np.pi
    mdp.move_robot(x, y, theta, yumi_joint_angles=None)
    frame_segment = mdp.camera.get_current_frame()["semantic_segmentation"]
    frame = mdp.camera.get_current_frame()
   
    mdp.reset()
    for i in range(50):
        
      
        r,s = mdp._get_reward()
        mdp.world.step(render=cfg.render)
     
        
        
        if i %2 == 0 or s == True:
            mdp.reset() #Setting render steps to 1 in reset seems to work, camera gets updated image
           
        

    mdp.shutdown()



@hydra.main(version_base=None, config_path="/home/adamfi/codes/base-pose-sequencing/conf", config_name="config")
def main(cfg : DictConfig) -> None:
    config = OmegaConf.to_yaml(cfg)
    print(config)
    
    test(cfg)

if __name__ == '__main__':
    main()
