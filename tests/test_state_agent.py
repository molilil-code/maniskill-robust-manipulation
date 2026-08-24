import gymnasium as gym
import mani_skill.envs
import torch

from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from src.models.agent import Agent


env = gym.make(
    "PushCube-v1",
    num_envs=1,
    obs_mode="state",
    sim_backend="physx_cpu",
)

if isinstance(env.action_space, gym.spaces.Dict):
    env = FlattenActionSpaceWrapper(env)

env = ManiSkillVectorEnv(
    env,
    num_envs=1,
    ignore_terminations=True,
    record_metrics=True,
)

obs, _ = env.reset(seed=1)

agent = Agent(
    env,
    obs_mode="state",
)

# 1. Critic
value = agent.get_value(obs)

# 2. Deterministic Actor
action = agent.get_action(
    obs,
    deterministic=True,
)

# 3. PPO Actor + Critic
sampled_action, logprob, entropy, value2 = (
    agent.get_action_and_value(obs)
)

print("obs shape:", obs.shape)
print("action shape:", action.shape)
print("value shape:", value.shape)
print("sampled action shape:", sampled_action.shape)
print("logprob shape:", logprob.shape)
print("entropy shape:", entropy.shape)

assert value.shape == (1, 1)
assert sampled_action.shape == action.shape
assert logprob.shape == (1,)
assert entropy.shape == (1,)

print("State Agent test PASSED.")

env.close()