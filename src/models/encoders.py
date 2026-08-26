import numpy as np
import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    def __init__(self, observation_space):
        super().__init__()

        self.output_dim = int(
            np.prod(observation_space.shape)
        )

    def forward(self, obs):
        return obs.reshape(obs.shape[0], -1)


class DepthEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(32 * 6 * 6, 256),
            nn.ReLU(),
        )

        self.proprio_dim = 9 + 9 + 7
        self.visual_dim = 256

        self.output_dim = (
            self.visual_dim + self.proprio_dim
        )

    def forward(self, obs):
        qpos = obs["agent"]["qpos"]
        qvel = obs["agent"]["qvel"]

        tcp_pose = obs["extra"]["tcp_pose"]
        goal_pos = obs["extra"]["goal_pos"]

        aux = torch.cat(
            [
                qpos,
                qvel,
                tcp_pose,
                goal_pos,
            ],
            dim=-1,
        )

        feature = torch.cat(
            [
                visual_feature,
                aux,
            ],
            dim=-1,
        )

        return feature
        