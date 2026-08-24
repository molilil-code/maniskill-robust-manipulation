from src.models.agent import Agent

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import src.envs
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
# ============================================================
# Evaluation cases
# ============================================================

EVAL_CASES = {

    "normal": {
        "env_id": "PushCubeEval-v1",
        "condition": "normal",
    },

    "cube_ood": {
        "env_id": "PushCubeEval-v1",
        "condition": "cube_ood",
    },

    "goal_near": {
        "env_id": "PushCubeEval-v1",
        "condition": "goal_near",
    },

    "goal_far": {
        "env_id": "PushCubeEval-v1",
        "condition": "goal_far",
    },

    "qpos_shift": {
        "env_id": "PushCubeEval-v1",
        "condition": "qpos_shift",
    },

    "episode_combined": {
        "env_id": "PushCubeEval-v1",
        "condition": "combined",
    },

    # Physics OOD
    "mass_low": {
        "env_id": "PushCubePhysicsEval-v1",
        "condition": "mass_low",
    },

    "mass_high": {
        "env_id": "PushCubePhysicsEval-v1",
        "condition": "mass_high",
    },

    "friction_low": {
        "env_id": "PushCubePhysicsEval-v1",
        "condition": "friction_low",
    },

    "friction_high": {
        "env_id": "PushCubePhysicsEval-v1",
        "condition": "friction_high",
    },

    "physics_combined": {
        "env_id": "PushCubePhysicsEval-v1",
        "condition": "physics_combined",
    },

    # Full OOD
    "full_combined": {
        "env_id": "PushCubeFullEval-v1",
        "condition": "full_combined",
    },
}


# ============================================================
# Create evaluation environment
# ============================================================

def make_eval_env(
    env_id,
    condition,
    sim_backend="physx_cpu",
    control_mode=None,
):
    """
    创建与官方 PPO 兼容的 evaluation environment。

    第一版固定 num_envs=1，方便 CPU 调试和保证统计清晰。
    """

    env_kwargs = {
        "obs_mode": "state",
        "reward_mode": "normalized_dense",
        "sim_backend": sim_backend,
        "condition": condition,
    }

    if control_mode is not None:
        env_kwargs["control_mode"] = control_mode

    env = gym.make(
        env_id,
        num_envs=1,
        **env_kwargs,
    )

    # PPO Agent要求连续 Box action space
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)

    # 与官方 PPO evaluation 的处理保持一致
    env = ManiSkillVectorEnv(
        env,
        num_envs=1,

        # 不因为 success 提前结束，
        # 一直跑到 episode horizon，
        # 这样得到 success_at_end
        ignore_terminations=True,

        # 自动统计 return、reward、
        # success_at_end、episode_len 等
        record_metrics=True,
    )

    return env


# ============================================================
# Load checkpoint
# ============================================================

def load_agent(
    env,
    checkpoint,
    device,
):
    agent = Agent(env).to(device)
    

    state_dict = torch.load(
        checkpoint,
        map_location=device,
    )

    # 你目前 PPO 保存的是：
    # torch.save(agent.state_dict(), path)
    agent.load_state_dict(state_dict)

    agent.eval()

    return agent


# ============================================================
# Evaluate one condition
# ============================================================

def evaluate_condition(
    checkpoint,
    case_name,
    num_episodes=100,
    seed=1000,
    sim_backend="physx_cpu",
    device="cpu",
    control_mode=None,
):
    set_seed(seed)

    if case_name not in EVAL_CASES:
        raise ValueError(
            f"Unknown evaluation case: {case_name}"
        )

    case = EVAL_CASES[case_name]

    env_id = case["env_id"]
    condition = case["condition"]

    print()
    print("=" * 60)
    print(f"Evaluation case : {case_name}")
    print(f"Environment     : {env_id}")
    print(f"Condition       : {condition}")
    print("=" * 60)

    env = make_eval_env(
        env_id=env_id,
        condition=condition,
        sim_backend=sim_backend,
        control_mode=control_mode,
    )

    device = torch.device(device)

    agent = load_agent(
        env=env,
        checkpoint=checkpoint,
        device=device,
    )

    # 同一个 seed 用于不同 policy，
    # 尽量保证公平比较
    obs, _ = env.reset(seed=seed)

    metrics = defaultdict(list)

    completed_episodes = 0

    # 安全上限，避免环境逻辑出错导致无限循环
    max_steps = num_episodes * 100

    steps = 0

    while completed_episodes < num_episodes:
        steps += 1

        if steps > max_steps:
            raise RuntimeError(
                "Evaluation exceeded max_steps. "
                "Check whether episodes are terminating/truncating correctly."
            )

        action_space_low = torch.from_numpy(
        env.single_action_space.low
        ).to(device)

        action_space_high = torch.from_numpy(
        env.single_action_space.high
        ).to(device)

 
        with torch.no_grad():
            action = agent.get_action(
                obs,
                deterministic=True,
            )

            action = torch.clamp(
                action,
                action_space_low,
                action_space_high,
            )


        obs, reward, terminated, truncated, infos = env.step(
            action
        )

        # ManiSkillVectorEnv会在episode结束时
        # 把统计信息放进final_info
        if "final_info" not in infos:
            continue

        done_mask = infos["_final_info"]

        num_finished = int(
            done_mask.sum().item()
        )

        if num_finished == 0:
            continue

        episode_info = infos["final_info"]["episode"]

        for key, value in episode_info.items():
            # value通常shape=[num_envs]
            selected = value[done_mask]

            values = (
                selected
                .detach()
                .cpu()
                .float()
                .reshape(-1)
                .tolist()
            )

            metrics[key].extend(values)

        completed_episodes += num_finished

        print(
            f"\rEpisodes: "
            f"{min(completed_episodes, num_episodes)}"
            f"/{num_episodes}",
            end="",
        )

    print()

    env.close()

    # 防止最后一次超过目标episode数量
    for key in metrics:
        metrics[key] = metrics[key][:num_episodes]

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    result = {
        "condition": case_name,
        "num_episodes": num_episodes,
    }

    for key, values in metrics.items():
        values = np.asarray(
            values,
            dtype=np.float32,
        )

        result[f"{key}_mean"] = float(
            values.mean()
        )

        result[f"{key}_std"] = float(
            values.std()
        )

    print()
    print("Results")
    print("-" * 40)

    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:.4f}")
        else:
            print(f"{key:30s}: {value}")

    return result


# ============================================================
# Evaluate all conditions
# ============================================================

def evaluate_all(
    checkpoint,
    model_name,
    num_episodes,
    seed,
    sim_backend,
    device,
    control_mode,
):
    results = []

    for case_name in EVAL_CASES:
        result = evaluate_condition(
            checkpoint=checkpoint,
            case_name=case_name,
            num_episodes=num_episodes,
            seed=seed,
            sim_backend=sim_backend,
            device=device,
            control_mode=control_mode,
        )

        result["model"] = model_name

        results.append(result)

    return results



def save_results(
    results,
    output_path,
):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = []

    for result in results:
        for key in result.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(results)

    print()
    print(
        f"Results saved to: {output_path}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to PPO checkpoint",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="ppo",
    )

    parser.add_argument(
        "--case",
        type=str,
        default="normal",
        choices=[
            *EVAL_CASES.keys(),
            "all",
        ],
    )

    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--sim-backend",
        type=str,
        default="physx_cpu",
        choices=[
            "physx_cpu",
            "physx_cuda",
        ],
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    parser.add_argument(
        "--control-mode",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation.csv",
    )

    args = parser.parse_args()

    if args.case == "all":

        results = evaluate_all(
            checkpoint=args.checkpoint,
            model_name=args.model_name,
            num_episodes=args.num_episodes,
            seed=args.seed,
            sim_backend=args.sim_backend,
            device=args.device,
            control_mode=args.control_mode,
        )

    else:

        result = evaluate_condition(
            checkpoint=args.checkpoint,
            case_name=args.case,
            num_episodes=args.num_episodes,
            seed=args.seed,
            sim_backend=args.sim_backend,
            device=args.device,
            control_mode=args.control_mode,
        )

        result["model"] = args.model_name

        results = [result]

    save_results(
        results,
        args.output,
    )


if __name__ == "__main__":
    main()