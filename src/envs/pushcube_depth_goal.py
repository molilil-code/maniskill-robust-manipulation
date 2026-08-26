from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.utils.registration import register_env


@register_env("PushCubeDepthGoal-v1", max_episode_steps=50)
class PushCubeDepthGoalEnv(PushCubeEnv):

    def _get_obs_extra(self, info):
        obs = super()._get_obs_extra(info)

        obs["goal_pos"] = self.goal_region.pose.p

        return obs