from mani_skill.utils.registration import register_env

from .pushcube_dr import PushCubeDREnv


class PushCubeDepthGoalAblationBaseEnv(PushCubeDREnv):
    """
    Base environment for factor-wise Episode DR ablations.

    Observation:
        Depth + goal_pos + robot proprioception

    The policy does NOT receive cube position directly.
    """

    def _get_obs_extra(self, info):
        # PushCubeDREnv provides tcp_pose
        obs = super()._get_obs_extra(info)

        # Depth+Goal policy additionally receives task goal
        obs["goal_pos"] = self.goal_region.pose.p

        return obs


# 1. Cube DR only

@register_env("PushCubeDepthGoalCubeDR-v1", max_episode_steps=50)
class PushCubeDepthGoalCubeDREnv(PushCubeDepthGoalAblationBaseEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,

            # Only Cube is randomized
            cube_xy_range=0.13,

            # Goal stays nominal
            goal_offset_range=(0.20, 0.20),

            # Qpos stays nominal
            robot_init_qpos_noise=0.02,

            **kwargs,
        )


# 2. Goal DR only

@register_env("PushCubeDepthGoalGoalDR-v1", max_episode_steps=50)
class PushCubeDepthGoalGoalDREnv(PushCubeDepthGoalAblationBaseEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,

            # Cube stays baseline
            cube_xy_range=0.10,

            # Only Goal is randomized
            goal_offset_range=(0.17, 0.23),

            # Qpos stays nominal
            robot_init_qpos_noise=0.02,

            **kwargs,
        )


# 3. Qpos DR only

@register_env("PushCubeDepthGoalQposDR-v1", max_episode_steps=50)
class PushCubeDepthGoalQposDREnv(PushCubeDepthGoalAblationBaseEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,

            # Cube stays baseline
            cube_xy_range=0.10,

            # Goal stays nominal
            goal_offset_range=(0.20, 0.20),

            # Only Qpos is randomized
            robot_init_qpos_noise=0.04,

            **kwargs,
        )