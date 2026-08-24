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
        return obs.reshape(
            obs.shape[0],
            -1,
        )

class DepthEncoder(nn.Module):
    def __init__(self, observation_space):
        super().__init__()

        raise NotImplementedError(
            "Depth encoder will be implemented "
            "after state GPU baseline."
        )