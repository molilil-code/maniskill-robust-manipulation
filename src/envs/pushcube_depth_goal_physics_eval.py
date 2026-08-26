from mani_skill.utils.registration import register_env

from src.envs.pushcube_physics_eval import PushCubePhysicsEvalEnv


@register_env(
    "PushCubeDepthGoalPhysicsEval-v1",
    max_episode_steps=50,
)
class PushCubeDepthGoalPhysicsEvalEnv(
    PushCubePhysicsEvalEnv
):

    def _get_obs_extra(self, info):
        obs = super()._get_obs_extra(info)

        obs["goal_pos"] = self.goal_region.pose.p

        return obs