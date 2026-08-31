import torch

from mani_skill.utils.registration import register_env

from src.envs.pushcube_depth_goal_dr import (
    PushCubeDepthGoalDREnv,
)


@register_env(
    "PushCubeDepthGoalContactDR-v1",
    max_episode_steps=50,
)
class PushCubeDepthGoalContactDREnv(
    PushCubeDepthGoalDREnv
):
    """
    Joint Episode DR + Contact-maintenance Reward.

    No progress reward is used in this environment.
    """

    def __init__(
        self,
        *args,
        contact_coef=0.02,
        contact_threshold=0.05,
        progress_tolerance=0.002,
        **kwargs,
    ):
        self.contact_coef = contact_coef
        self.contact_threshold = contact_threshold
        self.progress_tolerance = progress_tolerance

        self._prev_cube_goal_dist = None

        super().__init__(*args, **kwargs)

    def _initialize_episode(
        self,
        env_idx: torch.Tensor,
        options: dict,
    ):
        # 原来的 Cube / Goal / Qpos Episode DR
        super()._initialize_episode(
            env_idx,
            options,
        )

        cube_goal_dist = torch.linalg.norm(
            self.obj.pose.p[..., :2]
            - self.goal_region.pose.p[..., :2],
            dim=-1,
        )

        if self._prev_cube_goal_dist is None:
            self._prev_cube_goal_dist = torch.zeros(
                self.num_envs,
                device=self.device,
            )

        self._prev_cube_goal_dist[env_idx] = (
            cube_goal_dist[env_idx]
        )

    def compute_dense_reward(
        self,
        obs,
        action,
        info,
    ):
        # ------------------------------------------
        # 1. 官方 PushCube reward
        # ------------------------------------------
        reward = super().compute_dense_reward(
            obs,
            action,
            info,
        )

        # ------------------------------------------
        # 2. Cube -> Goal distance
        # ------------------------------------------
        cube_goal_dist = torch.linalg.norm(
            self.obj.pose.p[..., :2]
            - self.goal_region.pose.p[..., :2],
            dim=-1,
        )

        # ------------------------------------------
        # 3. TCP -> Cube distance
        # ------------------------------------------
        tcp_cube_dist = torch.linalg.norm(
            self.agent.tcp.pose.p
            - self.obj.pose.p,
            dim=-1,
        )

        # ------------------------------------------
        # 4. TCP 是否靠近 Cube
        # ------------------------------------------
        near_cube = (
            tcp_cube_dist
            < self.contact_threshold
        ).float()

        # ------------------------------------------
        # 5. Cube 是否没有明显被推离 Goal
        #
        # delta > 0:
        #   Cube 比上一时刻离 Goal 更远
        # ------------------------------------------
        delta = (
            cube_goal_dist
            - self._prev_cube_goal_dist
        )

        valid_direction = (
            delta <= self.progress_tolerance
        ).float()

        # ------------------------------------------
        # 6. 成功以后停止 shaping
        # ------------------------------------------
        if "success" in info:
            not_success = (
                ~info["success"]
            ).float()
        else:
            not_success = torch.ones_like(
                cube_goal_dist
            )

        # ------------------------------------------
        # 7. Contact maintenance reward
        # ------------------------------------------
        contact_reward = (
            self.contact_coef
            * near_cube
            * valid_direction
            * not_success
        )

        reward = reward + contact_reward

        # ------------------------------------------
        # 8. 更新历史距离
        # ------------------------------------------
        self._prev_cube_goal_dist = (
            cube_goal_dist.detach().clone()
        )

        return reward