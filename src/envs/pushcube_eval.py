import numpy as np
import torch
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose


@register_env("PushCubeEval-v1", max_episode_steps=50)
class PushCubeEvalEnv(PushCubeEnv):

    VALID_CONDITIONS = {
        "normal",
        "cube_ood",
        "goal_near",
        "goal_far",
        "qpos_shift",
        "combined",
    }

    def __init__(
        self,
        *args,
        condition="normal",
        **kwargs,
    ):
        if condition not in self.VALID_CONDITIONS:
            raise ValueError(
                f"Unknown condition: {condition}. "
                f"Expected one of {self.VALID_CONDITIONS}"
            )

        self.condition = condition

        # 只有这两种 condition 扩大机器人初始姿态扰动
        if condition in {"qpos_shift", "combined"}:
            self.eval_qpos_noise = 0.06
        else:
            self.eval_qpos_noise = 0.02

        super().__init__(
          *args,
          robot_init_qpos_noise=self.eval_qpos_noise,
           **kwargs,
        )

    def _sample_baseline_cube(self, b):
        xy = torch.rand((b, 2), device=self.device) * 0.2 - 0.1
        return xy

    def _sample_cube_outer_band(self, b):
        """
        Make sure at least one axis is outside DR training range:
        abs(axis) in [0.13, 0.16].
        """
        xy = (
            torch.rand((b, 2), device=self.device) * 0.32
            - 0.16
        )

        # 每个环境随机选择 x 或 y 作为 OOD 轴
        axis = torch.randint(
            0, 2, (b,), device=self.device
        )

        sign = torch.where(
            torch.rand(b, device=self.device) < 0.5,
            -1.0,
            1.0,
        )

        magnitude = (
            torch.rand(b, device=self.device) * 0.03
            + 0.13
        )

        rows = torch.arange(b, device=self.device)
        xy[rows, axis] = sign * magnitude

        return xy

    def _sample_goal_offset(self, b):
        if self.condition == "goal_near":
            low, high = 0.14, 0.16

        elif self.condition in {"goal_far", "combined"}:
            low, high = 0.24, 0.26

        else:
            #其他condition与官方保持一致
            nominal_offset = 0.1 + self.goal_radius

            return torch.full(
            (b,),
            nominal_offset,
            device=self.device,
        )

        return (
        torch.rand(b, device=self.device)
        * (high - low)
        + low
         )

    def _initialize_episode(
        self,
        env_idx: torch.Tensor,
        options: dict,
    ):
        if self.condition == "normal":
            return super()._initialize_episode(
                env_idx,
                options,
            )

        with torch.device(self.device):
            b = len(env_idx)

            # 初始化 robot/table
            self.table_scene.initialize(env_idx)

            if self.condition in {"cube_ood", "combined"}:
                xy = self._sample_cube_outer_band(b)
            else:
                xy = self._sample_baseline_cube(b)

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

            # -------------------------
            # Goal
            # -------------------------
            goal_offset = self._sample_goal_offset(b)

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