from mani_skill.utils.registration import register_env

from src.envs.pushcube_dr import PushCubeDREnv


@register_env(
    "PushCubeDepthGoalDR-v1",
    max_episode_steps=50,
)
class PushCubeDepthGoalDREnv(PushCubeDREnv):

    def _get_obs_extra(self, info):
        obs = super()._get_obs_extra(info)


        obs["goal_pos"] = self.goal_region.pose.p

        return obs