import numpy as np
import torch
from typing import Any
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose


@register_env("PushCubeDR-v1", max_episode_steps=50)
class PushCubeDREnv(PushCubeEnv):
    """
    Episode-level domain randomized version of PushCube-v1.

    Randomizations:
    1. cube xy initial position range
    2. robot initial qpos noise
    3. cube-to-goal distance
    """

    def __init__(
        self,
        *args,
        cube_xy_range=0.13,
        goal_offset_range=(0.17, 0.23),
        robot_init_qpos_noise=0.04,
        **kwargs,
    ):
        self.cube_xy_range = cube_xy_range
        self.goal_offset_range = goal_offset_range

        super().__init__(
            *args, 
            robot_init_qpos_noise=robot_init_qpos_noise,
            **kwargs,
            )

    def _initialize_episode(
        self,
        env_idx: torch.Tensor,
        options: dict,
    ):

        with torch.device(self.device):
            b = len(env_idx)

            self.table_scene.initialize(env_idx)
            # 随机 cube xy
            xyz = torch.zeros((b, 3))

            xyz[..., :2] = (
                torch.rand((b, 2)) * 2 - 1
            ) * self.cube_xy_range

            xyz[..., 2] = self.cube_half_size

            self.obj.set_pose(
                Pose.create_from_pq(
                    p=xyz,
                    q=[1, 0, 0, 0],
                )
            )

            # ------------------------------------------------
            # Randomize goal distance
            # ------------------------------------------------
            goal_min, goal_max = self.goal_offset_range

            goal_offset = (
                torch.rand(b) * (goal_max - goal_min)
                + goal_min
            )

            target_xyz = xyz.clone()

            # 目标仍然在 cube 的 +x 方向,只是距离随机
            target_xyz[..., 0] += goal_offset

            target_xyz[..., 2] = 1e-3

            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=target_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

            # 后面保持官方