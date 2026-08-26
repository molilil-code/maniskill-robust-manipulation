import gymnasium as gym
import torch

import src.envs

from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from src.models.agent import Agent


print("Creating Depth + Goal environment...")


# --------------------------------------------------
# Create environment
# --------------------------------------------------

env = gym.make(
    "PushCubeDepthGoal-v1",
    num_envs=1,
    obs_mode="depth",
    sim_backend="physx_cuda",
)

env = ManiSkillVectorEnv(
    env,
    num_envs=1,
    ignore_terminations=True,
    record_metrics=True,
)

print("Environment created.")


# --------------------------------------------------
# Reset
# --------------------------------------------------

obs, info = env.reset(seed=1)

print("Environment reset.")


# --------------------------------------------------
# Check observation
# --------------------------------------------------

print("\n=== Observation ===")

print(
    "extra keys:",
    obs["extra"].keys(),
)

print(
    "depth:",
    obs["sensor_data"]["base_camera"]["depth"].shape,
)

print(
    "qpos:",
    obs["agent"]["qpos"].shape,
)

print(
    "qvel:",
    obs["agent"]["qvel"].shape,
)

print(
    "tcp_pose:",
    obs["extra"]["tcp_pose"].shape,
)

print(
    "goal_pos:",
    obs["extra"]["goal_pos"].shape,
)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = obs["sensor_data"]["base_camera"]["depth"].device

print("\n=== Device ===")
print("Observation device:", device)


# --------------------------------------------------
# Create Depth + Goal Agent
# --------------------------------------------------

agent = Agent(
    env,
    encoder_type="depth_goal"
).to(device)

agent.eval()

print("Agent device:", next(agent.parameters()).device)


# --------------------------------------------------
# Forward
# --------------------------------------------------

with torch.no_grad():

    feature = agent.encode(obs)

    deterministic_action = agent.get_action(
        obs,
        deterministic=True,
    )

    value = agent.get_value(obs)

    action, logprob, entropy, value2 = (
        agent.get_action_and_value(obs)
    )


# --------------------------------------------------
# Print Encoder results
# --------------------------------------------------

print("\n=== Encoder ===")

print(
    "feature:",
    feature.shape,
)

print(
    "encoder output_dim:",
    agent.encoder.output_dim,
)


# --------------------------------------------------
# Print Agent results
# --------------------------------------------------

print("\n=== Agent ===")

print(
    "deterministic action:",
    deterministic_action.shape,
)

print(
    "sampled action:",
    action.shape,
)

print(
    "logprob:",
    logprob.shape,
)

print(
    "entropy:",
    entropy.shape,
)

print(
    "value:",
    value.shape,
)

print(
    "encoder class:",
    type(agent.encoder).__name__,
)

# --------------------------------------------------
# Assertions
# --------------------------------------------------

assert obs["sensor_data"]["base_camera"]["depth"].shape == (
    1, 128, 128, 1
)

assert obs["agent"]["qpos"].shape == (1, 9)

assert obs["agent"]["qvel"].shape == (1, 9)

assert obs["extra"]["tcp_pose"].shape == (1, 7)

assert obs["extra"]["goal_pos"].shape == (1, 3)


# 256 visual + 9 qpos + 9 qvel + 7 tcp + 3 goal
assert agent.encoder.output_dim == 284

assert feature.shape == (
    1,
    284,
)

assert deterministic_action.shape == (1, 8)

assert action.shape == deterministic_action.shape

assert logprob.shape == (1,)

assert entropy.shape == (1,)

assert value.shape == (1, 1)


# numerical sanity checks
assert torch.isfinite(feature).all()
assert torch.isfinite(action).all()
assert torch.isfinite(logprob).all()
assert torch.isfinite(entropy).all()
assert torch.isfinite(value).all()
assert torch.isfinite(value2).all()


env.close()

print("\nDepth + Goal Agent forward test PASSED.")