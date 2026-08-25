import gymnasium as gym
import mani_skill.envs


def print_structure(x, prefix="obs"):
    if isinstance(x, dict):
        for k, v in x.items():
            print_structure(v, f"{prefix}.{k}")
    else:
        print(
            f"{prefix}: "
            f"type={type(x).__name__}, "
            f"shape={getattr(x, 'shape', None)}, "
            f"dtype={getattr(x, 'dtype', None)}, "
            f"device={getattr(x, 'device', None)}"
        )


print("Creating environment...")

env = gym.make(
    "PushCube-v1",
    num_envs=1,
    obs_mode="rgb+depth",
    sim_backend="physx_cpu",
    render_backend="sapien_cpu",
)

print("Environment created.")

obs, info = env.reset(seed=1)

print("Environment reset.")
print_structure(obs)

env.close()

print("Depth observation test PASSED.")

depth = obs["sensor_data"]["base_camera"]["depth"]

print("\nDepth statistics:")
print("shape:", depth.shape)
print("dtype:", depth.dtype)
print("min:", depth.min().item())
print("max:", depth.max().item())
print("mean:", depth.float().mean().item())
print("zero ratio:", (depth == 0).float().mean().item())