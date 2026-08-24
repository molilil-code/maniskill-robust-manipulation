import numpy as np
import torch
from transforms3d.euler import euler2quat

from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose

from .pushcube_physics_dr import PushCubePhysicsDREnv


@register_env("PushCubeFullEval-v1", max_episode_steps=50)
class PushCubeFullEvalEnv(PushCubePhysicsDREnv):
    """
    Full OOD evaluation environment.

    OOD factors:
    - cube position
    - goal offset
    - robot initial qpos
    - cube mass
    - cube friction
    """

    def __init__(
        self,
        *args,
        condition="full_combined",
        **kwargs,
    ):
        if condition != "full_combined":
            raise ValueError(
                f"Unknown condition: {condition}"
            )

        self.condition = condition

        # -----------------------------
        # Physics OOD
        # -----------------------------
        mass = 0.096
        friction = 0.15

        # -----------------------------
        # Robot initial-state OOD
        # -----------------------------
        robot_init_qpos_noise = 0.06

        super().__init__(
            *args,

            # 固定为OOD测试值，而不是随机训练范围
            mass_range=(mass, mass),
            friction_range=(friction, friction),

            robot_init_qpos_noise=robot_init_qpos_noise,

            **kwargs,
        )

    def _sample_cube_outer_band(self, b):
        """
        Cube必须至少有一个轴位于DR训练范围之外：

        abs(x) in [0.13, 0.16]
        或
        abs(y) in [0.13, 0.16]
        """

        # 首先在完整 [-0.16, 0.16]^2 中采样
        xy = (
            torch.rand(
                (b, 2),
                device=self.device,
            ) * 0.32
            - 0.16
        )

        # 每个环境随机选择 x 或 y，
        # 强制这个轴落在 outer band
        axis = torch.randint(
            0,
            2,
            (b,),
            device=self.device,
        )

        # ±方向
        sign = torch.where(
            torch.rand(
                b,
                device=self.device,
            ) < 0.5,
            -1.0,
            1.0,
        )

        # 绝对值范围 [0.13, 0.16]
        magnitude = (
            torch.rand(
                b,
                device=self.device,
            ) * 0.03
            + 0.13
        )

        rows = torch.arange(
            b,
            device=self.device,
        )

        xy[rows, axis] = sign * magnitude

        return xy

    def _initialize_episode(
        self,
        env_idx: torch.Tensor,
        options: dict,
    ):
        with torch.device(self.device):
            b = len(env_idx)


            self.table_scene.initialize(env_idx)

 
            xy = self._sample_cube_outer_band(b)

            xyz = torch.zeros(
                (b, 3),
                device=self.device,
            )

            xyz[..., :2] = xy
            xyz[..., 2] = self.cube_half_size

            self.obj.set_pose(
                Pose.create_from_pq(
                    p=xyz,
                    q=[1, 0, 0, 0],
                )
            )


            goal_offset = (
                torch.rand(
                    b,
                    device=self.device,
                ) * 0.02
                + 0.24
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