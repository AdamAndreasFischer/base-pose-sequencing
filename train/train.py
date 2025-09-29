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



def get_stats(dataset, logger):
    score = dataset.compute_metrics()
    logger.info(('min_reward: %f, max_reward: %f, mean_reward: %f,'
                ' median_reward: %f, games_completed: %d' % score))

    return score
    

def experiment(cfg, alg):
    np.random.seed()

    # tracemalloc.start()

    logger = Logger(alg.__name__, results_dir=cfg.task.train.save_dir)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    # MDP
    mdp = Task(cfg)
  
    print('observation space:', mdp.info.observation_space.shape)
    print('action space:', mdp.info.action_space.shape)

    n_epochs=1000 
    n_steps=5 
    
    
    # Settings
    n_steps_test=5
    initial_replay_size = 10
    max_replay_size = 500 #4000 This will require approx 25 GB of RAM. RTX 4070 has 12 GB VRAM. Max buffer size ~500
    batch_size = 16
    n_features = 128
    warmup_transitions = 500
    tau = 0.001
    lr_alpha = 3e-4
    n_steps_per_fit = 2
    dqn_target_update_frequency = 8
    

    

    # Test settings
    # n_steps_test = 2
    # initial_replay_size = 2
    # max_replay_size = 4
    # batch_size = 4
    # n_features = 128
    # warmup_transitions = 2
    # tau = 0.001
    # lr_alpha = 3e-4
    # n_steps_per_fit = 2

    # Approximator
    actor_input_shape = mdp.info.observation_space.shape
    
    
    actor_map_params = dict(network=ActorNetwork,
                                input_shape=actor_input_shape,
                                output_shape=mdp.info.action_space.shape,
                                n_features=n_features,
                                state_shape=mdp.info.observation_space.shape,
                                action_shape=mdp.info.action_space.shape)
    actor_kernel_params = dict(network=ActorNetwork,
                                input_shape=actor_input_shape,
                                output_shape=mdp.info.action_space.shape,
                                n_features=n_features,
                                state_shape=mdp.info.observation_space.shape,
                                action_shape=mdp.info.action_space.shape) #Only necesarry if we want to tweak actor networks separately
 
    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': 5e-4}}

    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_state_shape = (4,160,160)
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 5e-4}},
                         loss=F.mse_loss,
                         output_shape=(1,),
                         n_features=n_features,
                         input_shape=critic_input_shape,
                         state_shape= critic_state_shape,# mdp.info.observation_space.shape,
                         action_shape=mdp.info.action_space.shape)

    dueling_Q_params = dict(network = DuelingQNetwork,
                            input_shape = mdp.info.action_space.shape,
                            action_shape = mdp.info.action_space.shape,
                            output_shape = mdp.info.action_space.shape,
                            hidden_shape = mdp.info.action_space.shape 
                            )
    # Agent 
    #agent = alg(mdp.info, actor_map_params, actor_kernel_params,
    #            actor_optimizer, critic_params, batch_size, initial_replay_size,
    #            max_replay_size, warmup_transitions, tau, lr_alpha,
    #            critic_fit_params=None, target_entropy=None)
    dqn_replay_buffer = {'class': ReplayMemory,
                         'params': {'alpha':0.6, 'beta': 0.4}}
    
 
    dqn_agent = alg(mdp.info, approximator_params =  actor_map_params, policy_optimizer= actor_optimizer,batch_size= batch_size, target_update_frequency = dqn_target_update_frequency, 
                    initial_replay_size= initial_replay_size, max_replay_size= max_replay_size, replay_memory =dqn_replay_buffer, dueling_Q_params= dueling_Q_params )
    agent = dqn_agent
    if cfg.task.train.enable_q_priors:
        agent.initialize_q_prior(cfg.path_prefix)

    if cfg.task.train.enable_action_priors:
        agent.initialize_action_prior(cfg.path_prefix)

    # Algorithm
    core = Core(agent, mdp)

    # RUN
    dataset = core.evaluate(n_steps=n_steps_test, render=False)

    J = np.mean(dataset.discounted_return)
    R = np.mean(dataset.undiscounted_return)
    E = "N/A"# agent.policy.entropy(dataset.state).item()

    # del dataset

    logger.epoch_info(0, J=J, R=R, entropy=E)

    dataset = core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size, quiet=True)
    del dataset

    for n in trange(n_epochs, leave=False):
        dataset = core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit, quiet=True)
        del dataset
        with torch.no_grad():
            dataset = core.evaluate(n_steps=n_steps_test, render=False, quiet=True)

        J = np.mean(dataset.discounted_return)
        R = np.mean(dataset.undiscounted_return)
        E = "N/A"# agent.policy.entropy(dataset.state).item()

        logger.epoch_info(n+1, J=J, R=R, entropy=E)

        # Save the agent after each epoch
        agent.save(os.path.join(cfg.path_prefix + cfg.task.train.save_dir, f"dqn_agent_epoch.msh"))
        

        # NOTE: Dataset size is close to 300 MB for 50 samples, replay memory seems to be a pointer so cannot be directly measured but it east up the RAM
        # dataset_size = asizeof.asizeof(dataset)
        # print(f"Dataset size: {dataset_size} bytes ({dataset_size / (1024 ** 2):.2f} MB)")

        del dataset

        # print("Memory allocation:", tracemalloc.get_traced_memory())
    
    # tracemalloc.stop()


    mdp.shutdown()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg : DictConfig) -> None:
    config = OmegaConf.to_yaml(cfg)
    print(type(config))
    experiment(cfg, alg=SAC)

if __name__ == '__main__':
    main()
