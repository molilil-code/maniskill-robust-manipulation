import numpy as np
import torch
from transforms3d.euler import euler2quat

from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose

from .pushcube_physics_dr import PushCubePhysicsDREnv


@register_env("PushCubeFullDR-v1", max_episode_steps=50)
class PushCubeFullDREnv(PushCubePhysicsDREnv):
    """
    Full domain randomization for PushCube.

    Episode-level DR:
        - cube xy position
        - goal offset
        - robot initial qpos

    Physics DR:
        - cube mass
        - cube friction
    """

    def __init__(
        self,
        *args,

        # Episode DR
        cube_xy_range=0.13,
        goal_offset_range=(0.17, 0.23),
        robot_init_qpos_noise=0.04,

        # Physics DR
        mass_range=(0.0448, 0.0832),
        friction_range=(0.21, 0.39),

        **kwargs,
    ):

        self.cube_xy_range = cube_xy_range
        self.goal_offset_range = goal_offset_range

        # Physics DR交给父类处理

    
        super().__init__(
            *args,
            mass_range=mass_range,
            friction_range=friction_range,
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

            xyz = torch.zeros(
                (b, 3),
                device=self.device,
            )

            xyz[..., :2] = (
                torch.rand(
                    (b, 2),
                    device=self.device,
                ) * 2 - 1
            ) * self.cube_xy_range

            xyz[..., 2] = self.cube_half_size

            self.obj.set_pose(
                Pose.create_from_pq(
                    p=xyz,
                    q=[1, 0, 0, 0],
                )
            )

            goal_min, goal_max = (
                self.goal_offset_range
            )

            goal_offset = (
                torch.rand(
                    b,
                    device=self.device,
                )
                * (goal_max - goal_min)
                + goal_min
            )

            target_xyz = xyz.clone()

            target_xyz[..., 0] += goal_offset

            target_xyz[..., 2] = 1e-3

            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=target_xyz,
                    q=euler2quat(
                        0,
                        np.pi / 2,
                        0,
                    ),
                )
            )