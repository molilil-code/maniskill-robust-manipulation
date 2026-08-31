import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from src.models.encoders import (
    StateEncoder,
    DepthEncoder,
    DepthGoalEncoder,
    DepthGoalFrameStackEncoder,
)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, encoder_type="state"):
        super().__init__()

        # ---------------------------------------------
        # Observation Encoder
        # ---------------------------------------------
        if encoder_type == "state":
            self.encoder = StateEncoder(
                envs.single_observation_space
            )

        elif encoder_type == "depth":
            self.encoder = DepthEncoder()

        elif encoder_type == "depth_goal":
            self.encoder = DepthGoalEncoder()

        elif encoder_type == "depth_goal_stack4":
            self.encoder = DepthGoalFrameStackEncoder()  

        else:
            raise ValueError(
                f"Unsupported encoder_type: {encoder_type}"
            )

        feature_dim = self.encoder.output_dim

        action_dim = int(
            np.prod(envs.single_action_space.shape)
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(feature_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(feature_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(
                nn.Linear(256, action_dim),
                std=0.01,
            ),
        )

        self.actor_logstd = nn.Parameter(
            torch.ones(1, action_dim) * -0.5
        )

    def encode(self, obs):
        return self.encoder(obs)

    def get_value(self, obs):
        x = self.encode(obs)
        return self.critic(x)

    def get_action(self, obs, deterministic=False):
        x = self.encode(obs)

        action_mean = self.actor_mean(x)

        if deterministic:
            return action_mean

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        probs = Normal(action_mean, action_std)

        return probs.sample()

    def get_action_and_value(self, obs, action=None):
        x = self.encode(obs)

        action_mean = self.actor_mean(x)

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )