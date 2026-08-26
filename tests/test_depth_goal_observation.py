import gymnasium as gym
import src.envs

env = gym.make(
    "PushCubeDepthGoal-v1",
    num_envs=16,
    obs_mode="depth",
    sim_backend="physx_cuda",
)

obs, _ = env.reset(seed=1)

print("depth:",
      obs["sensor_data"]["base_camera"]["depth"].shape)

print("qpos:",
      obs["agent"]["qpos"].shape)

print("tcp:",
      obs["extra"]["tcp_pose"].shape)

print("goal:",
      obs["extra"]["goal_pos"].shape)

print("extra keys:", obs["extra"].keys())

env.close()