import gymnasium as gym
import src.envs

from src.models.encoders import DepthEncoder

env = gym.make(
    "PushCubeDepthGoal-v1",
    num_envs=1,
    obs_mode="depth",
    sim_backend="physx_cuda",
)

obs, _ = env.reset(seed=1)

encoder = DepthEncoder()

feature = encoder(obs)

obs, info = env.reset(seed=1)

print("extra keys:", obs["extra"].keys())
print("goal_pos:", obs["extra"]["goal_pos"].shape)

print("depth:",
      obs["sensor_data"]["base_camera"]["depth"].shape)

print("qpos:",
      obs["agent"]["qpos"].shape)

print("feature:", feature.shape)
print("encoder output_dim:", encoder.output_dim)

assert feature.shape == (1, encoder.output_dim)
assert encoder.output_dim == 284

env.close()

print("Depth Encoder test PASSED.")