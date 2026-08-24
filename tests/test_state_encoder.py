import gymnasium as gym
import mani_skill.envs

from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from src.models.encoders import StateEncoder


env = gym.make(
    "PushCube-v1",
    num_envs=1,
    obs_mode="state",
    sim_backend="physx_cpu",
)

# 和 train.py 保持一致
env = ManiSkillVectorEnv(
    env,
    num_envs=1,
    ignore_terminations=True,
    record_metrics=True,
)

obs, _ = env.reset(seed=1)

encoder = StateEncoder(
    env.single_observation_space
)

feature = encoder(obs)

print("obs shape:", obs.shape)
print("feature shape:", feature.shape)
print("encoder output_dim:", encoder.output_dim)

assert feature.shape[0] == 1
assert feature.shape[1] == encoder.output_dim

print("State encoder test PASSED.")

env.close()