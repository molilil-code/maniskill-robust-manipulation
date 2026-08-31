import torch

from mani_skill.utils.registration import register_env

from src.envs.pushcube_depth_goal_dr import PushCubeDepthGoalDREnv


@register_env(
    "PushCubeDepthGoalProgressDR-v1",
    max_episode_steps=50,
)
class PushCubeDepthGoalProgressDREnv(PushCubeDepthGoalDREnv):

    def __init__(
        self,
        *args,
        progress_coef=10.0,
        contact_threshold=0.05,
        **kwargs,
    ):
        self.progress_coef = progress_coef
        self.contact_threshold = contact_threshold

        self._prev_cube_goal_dist = None

        super().__init__(*args, **kwargs)

    def _initialize_episode(
    self,
    env_idx: torch.Tensor,
    options: dict,
):
        # 先执行原来的 Episode DR 初始化
        super()._initialize_episode(env_idx, options)

        # 当前 cube → goal 的 XY 距离
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

        # 注意只重置当前 reset 的 env
        self._prev_cube_goal_dist[env_idx] = cube_goal_dist[env_idx]

    def compute_dense_reward(
    self,
    obs,
    action,
    info,
):
        # --------------------------------------------------
        # 1. 保留原来的 ManiSkill PushCube reward
        # --------------------------------------------------
        reward = super().compute_dense_reward(
            obs,
            action,
            info,
        )

        # --------------------------------------------------
        # 2. 当前 Cube → Goal 距离
        # --------------------------------------------------
        cube_goal_dist = torch.linalg.norm(
            self.obj.pose.p[..., :2]
            - self.goal_region.pose.p[..., :2],
            dim=-1,
        )

        # --------------------------------------------------
        # 3. 当前 TCP → Cube 距离
        # --------------------------------------------------
        tcp_cube_dist = torch.linalg.norm(
            self.agent.tcp.pose.p
            - self.obj.pose.p,
            dim=-1,
        )

        # --------------------------------------------------
        # 4. 本 timestep 的推动进度
        # positive = toward goal
        # negative = away from goal
        # --------------------------------------------------
        progress = (
            self._prev_cube_goal_dist
            - cube_goal_dist
        )

        progress = torch.clamp(
            progress,
            min=-0.02,
            max=0.02,
        )

        # --------------------------------------------------
        # 5. TCP 足够靠近 Cube 才认为是在有效推动
        # --------------------------------------------------
        near_cube = (
            tcp_cube_dist < self.contact_threshold
        ).float()

        push_progress = near_cube * progress

        not_success = (~info["success"]).float()

        push_progress = (
            near_cube
            * progress
            * not_success
        )

        # --------------------------------------------------
        # 6. 加入 shaping reward
        # --------------------------------------------------
        reward = (
            reward
            + self.progress_coef * push_progress
        )

        # --------------------------------------------------
        # 7. 更新历史距离
        # detach 非常重要
        # --------------------------------------------------
        self._prev_cube_goal_dist = (
            cube_goal_dist.detach().clone()
        )

        return reward