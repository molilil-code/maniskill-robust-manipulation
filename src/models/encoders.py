import numpy as np
import torch
import torch.nn as nn


# ============================================================
# State Encoder
# ============================================================

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


# ============================================================
# Pure Depth Encoder
# depth + qpos + qvel + tcp_pose
# ============================================================

class DepthEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(
                1, 16,
                kernel_size=8,
                stride=4,
            ),
            nn.ReLU(),

            nn.Conv2d(
                16, 32,
                kernel_size=4,
                stride=2,
            ),
            nn.ReLU(),

            nn.Conv2d(
                32, 32,
                kernel_size=3,
                stride=2,
            ),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(
                32 * 6 * 6,
                256,
            ),
            nn.ReLU(),
        )

        self.visual_dim = 256

        # qpos 9 + qvel 9 + tcp 7
        self.aux_dim = 9 + 9 + 7

        # 256 + 25 = 281
        self.output_dim = (
            self.visual_dim
            + self.aux_dim
        )

    def forward(self, obs):

        # ------------------------
        # Depth
        # ------------------------
        depth = (
            obs["sensor_data"]
            ["base_camera"]
            ["depth"]
        )

        # mm -> m
        depth = depth.float() / 1000.0

        # clip + normalize
        depth = torch.clamp(
            depth,
            0.0,
            2.0,
        ) / 2.0

        # [B,H,W,1] -> [B,1,H,W]
        depth = depth.permute(
            0, 3, 1, 2
        )

        visual_feature = self.cnn(depth)

        # ------------------------
        # Robot proprioception
        # ------------------------
        qpos = obs["agent"]["qpos"]
        qvel = obs["agent"]["qvel"]
        tcp_pose = obs["extra"]["tcp_pose"]

        aux = torch.cat(
            [
                qpos,
                qvel,
                tcp_pose,
            ],
            dim=-1,
        )

        # ------------------------
        # Fusion
        # ------------------------
        feature = torch.cat(
            [
                visual_feature,
                aux,
            ],
            dim=-1,
        )

        return feature


# ============================================================
# Depth + Goal Encoder
# depth + qpos + qvel + tcp_pose + goal_pos
# ============================================================

class DepthGoalEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(
                1, 16,
                kernel_size=8,
                stride=4,
            ),
            nn.ReLU(),

            nn.Conv2d(
                16, 32,
                kernel_size=4,
                stride=2,
            ),
            nn.ReLU(),

            nn.Conv2d(
                32, 32,
                kernel_size=3,
                stride=2,
            ),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(
                32 * 6 * 6,
                256,
            ),
            nn.ReLU(),
        )

        self.visual_dim = 256

        # qpos 9
        # + qvel 9
        # + tcp_pose 7
        # + goal_pos 3
        self.aux_dim = 9 + 9 + 7 + 3

        # 256 + 28 = 284
        self.output_dim = (
            self.visual_dim
            + self.aux_dim
        )

    def forward(self, obs):

        # ------------------------
        # Depth
        # ------------------------
        depth = (
            obs["sensor_data"]
            ["base_camera"]
            ["depth"]
        )

        depth = depth.float() / 1000.0

        depth = torch.clamp(
            depth,
            0.0,
            2.0,
        ) / 2.0

        depth = depth.permute(
            0, 3, 1, 2
        )

        visual_feature = self.cnn(depth)

        # ------------------------
        # Robot + Goal
        # ------------------------
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

        # ------------------------
        # Fusion
        # ------------------------
        feature = torch.cat(
            [
                visual_feature,
                aux,
            ],
            dim=-1,
        )

        return feature

class DepthGoalFrameStackEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(
                4, 16,
                kernel_size=8,
                stride=4,
            ),
            nn.ReLU(),

            nn.Conv2d(
                16, 32,
                kernel_size=4,
                stride=2,
            ),
            nn.ReLU(),

            nn.Conv2d(
                32, 32,
                kernel_size=3,
                stride=2,
            ),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(
                32 * 6 * 6,
                256,
            ),
            nn.ReLU(),
        )

        self.visual_dim = 256

        # qpos 9
        # + qvel 9
        # + tcp_pose 7
        # + goal_pos 3
        self.aux_dim = 9 + 9 + 7 + 3

        # 256 + 28 = 284
        self.output_dim = (
            self.visual_dim
            + self.aux_dim
        )

    def forward(self, obs):

        # --------------------------------
        # Stacked Depth
        # [B, H, W, 4]
        # --------------------------------
        depth = obs["depth_stack"]

        # mm -> m
        depth = depth.float() / 1000.0

        # clip + normalize
        depth = torch.clamp(
            depth,
            0.0,
            2.0,
        ) / 2.0

        # [B,H,W,4] -> [B,4,H,W]
        depth = depth.permute(
            0, 3, 1, 2
        )

        visual_feature = self.cnn(
            depth
        )

        # --------------------------------
        # Robot proprioception + Goal
        # --------------------------------
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

        # --------------------------------
        # Fusion
        # --------------------------------
        feature = torch.cat(
            [
                visual_feature,
                aux,
            ],
            dim=-1,
        )

        return feature