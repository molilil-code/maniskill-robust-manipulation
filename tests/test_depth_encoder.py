import gymnasium as gym
import mani_skill.envs

from src.models.encoders import DepthEncoder


env = gym.make(
    "PushCube-v1",
    num_envs=1,
    obs_mode="rgb+depth",
    sim_backend="physx_cpu",
    render_backend="sapien_cpu",
)

obs, _ = env.reset(seed=1)

encoder = DepthEncoder()

feature = encoder(obs)

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