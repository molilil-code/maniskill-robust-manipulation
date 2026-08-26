import gymnasium as gym
import torch
import src.envs


for condition in [
    "normal",
    "goal_far",
    "cube_ood",
]:

    env = gym.make(
        "PushCubeDepthGoalEval-v1",
        num_envs=1,
        obs_mode="depth",
        sim_backend="physx_cuda",
        condition=condition,
    )

    all_depth = []

    for seed in range(1000, 1020):
        obs, _ = env.reset(seed=seed)

        depth = (
            obs["sensor_data"]
            ["base_camera"]
            ["depth"]
            .float()
            / 1000.0
        )

        all_depth.append(depth.reshape(-1))

    depth = torch.cat(all_depth)

    print("\n", condition)
    print("min:", depth.min().item())
    print("max:", depth.max().item())

    print(
        "ratio >= 2m:",
        (depth >= 2.0).float().mean().item()
    )

    print(
        "ratio < 2m:",
        (depth < 2.0).float().mean().item()
    )

    print(
        "p50:",
        torch.quantile(depth, 0.50).item()
    )

    print(
        "p90:",
        torch.quantile(depth, 0.90).item()
    )

    print(
        "p95:",
        torch.quantile(depth, 0.95).item()
    )

    env.close()