from mani_skill.utils.registration import register_env

from src.envs.pushcube_eval import PushCubeEvalEnv


@register_env("PushCubeDepthGoalEval-v1", max_episode_steps=50)
class PushCubeDepthGoalEvalEnv(PushCubeEvalEnv):

    def _get_obs_extra(self, info):
        obs = super()._get_obs_extra(info)

        # 这里不要猜变量名：
        # 直接复制你 PushCubeDepthGoal-v1 中已经验证成功的这一行
        obs["goal_pos"] = self.goal_region.pose.p

        return obs