import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import matplotlib.pyplot as plt

class SimpleCNNImageExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 3, trans_lim = (-2.5,2.5), rot_lim= (-th.pi, th.pi)):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        print("Observation shape in CNN init: ", observation_space)
        print("Shape of observation space in CNN init: ", observation_space.shape)
        print("Trans and rot lim: ", trans_lim, " ", rot_lim)
        self.range = th.tensor([trans_lim[-1], trans_lim[-1], rot_lim[-1]], device="cuda" if th.cuda.is_available else "cpu") # Only adding max limit as output in (-1,1)
        
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim)) # No activation function on output 

    def forward(self, observations: th.Tensor) -> th.Tensor:
        
        x= self.linear(self.cnn(observations))
        
        print(x)

        return x