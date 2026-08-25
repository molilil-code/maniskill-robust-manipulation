import gymnasium as gym
import torch

import mani_skill.envs
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from src.models.agent import Agent


print("Creating Depth environment...")

env = gym.make(
    "PushCube-v1",
    num_envs=1,
    obs_mode="rgb+depth",
    sim_backend="physx_cpu",
    render_backend="sapien_cpu",
)

env = ManiSkillVectorEnv(
    env,
    ignore_terminations=True,
    record_metrics=True,
)

print("Environment created.")

obs, info = env.reset(seed=1)

print("Environment reset.")


# --------------------------------------------------
# Create Depth Agent
# --------------------------------------------------

agent = Agent(
    env,
    encoder_type="depth",
)

agent.eval()


# --------------------------------------------------
# Encoder output
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


print("\n=== Observation ===")

print(
    "depth:",
    obs["sensor_data"]["base_camera"]["depth"].shape
)

print(
    "qpos:",
    obs["agent"]["qpos"].shape
)


print("\n=== Encoder ===")

print("feature:", feature.shape)
print(
    "encoder output_dim:",
    agent.encoder.output_dim
)


print("\n=== Agent ===")

print(
    "deterministic action:",
    deterministic_action.shape
)

print(
    "sampled action:",
    action.shape
)

print(
    "logprob:",
    logprob.shape
)

print(
    "entropy:",
    entropy.shape
)

print(
    "value:",
    value.shape
)


# --------------------------------------------------
# Assertions
# --------------------------------------------------

assert feature.shape == (
    1,
    agent.encoder.output_dim,
)

assert agent.encoder.output_dim == 281

assert deterministic_action.shape[0] == 1
assert action.shape == deterministic_action.shape

assert logprob.shape == (1,)
assert entropy.shape == (1,)
assert value.shape == (1, 1)

assert torch.isfinite(feature).all()
assert torch.isfinite(action).all()
assert torch.isfinite(logprob).all()
assert torch.isfinite(entropy).all()
assert torch.isfinite(value).all()


env.close()

print("\nDepth Agent forward test PASSED.")