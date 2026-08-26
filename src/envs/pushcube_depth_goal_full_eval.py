from mani_skill.utils.registration import register_env

from src.envs.pushcube_full_eval import PushCubeFullEvalEnv


@register_env(
    "PushCubeDepthGoalFullEval-v1",
    max_episode_steps=50,
)
class PushCubeDepthGoalFullEvalEnv(
    PushCubeFullEvalEnv
):

    def _get_obs_extra(self, info):
        obs = super()._get_obs_extra(info)

        obs["goal_pos"] = self.goal_region.pose.p

        return obs