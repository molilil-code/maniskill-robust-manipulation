import argparse
import csv
from pathlib import Path

import gymnasium as gym
import torch

import src.envs

from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from src.models.agent import Agent


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--condition",
        type=str,
        required=True,
        choices=[
            "goal_far",
            "cube_ood",
        ],
    )

    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
    )

    args = parser.parse_args()

    device = torch.device("cuda")

    video_dir = Path(
        f"videos/depth_goal/{args.condition}"
    )
    video_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==================================================
    # 1. Environment
    # ==================================================

    raw_env = gym.make(
        "PushCubeDepthGoalEval-v1",
        num_envs=1,
        obs_mode="depth",
        render_mode="rgb_array",
        sim_backend="physx_cuda",
        condition=args.condition,
    )

    # 保留真实 task env 引用，只用于诊断
    task_env = raw_env.unwrapped

    env = raw_env

    if isinstance(
        env.action_space,
        gym.spaces.Dict,
    ):
        env = FlattenActionSpaceWrapper(env)

    env = RecordEpisode(
        env,
        output_dir=str(video_dir),
        save_trajectory=False,
        save_video=True,
        info_on_video=True,
        max_steps_per_video=50,
        video_fps=20,
    )

    env = ManiSkillVectorEnv(
        env,
        num_envs=1,
        ignore_terminations=True,
        record_metrics=True,
    )

    # ==================================================
    # 2. Agent
    # ==================================================

    agent = Agent(
        env,
        encoder_type="depth_goal",
    ).to(device)

    state_dict = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )

    agent.load_state_dict(state_dict)
    agent.eval()

    action_low = torch.from_numpy(
        env.single_action_space.low
    ).to(device)

    action_high = torch.from_numpy(
        env.single_action_space.high
    ).to(device)

    # ==================================================
    # 3. Rollout
    # ==================================================

    obs, _ = env.reset(seed=args.seed)

    rows = []

    completed = 0
    episode_id = 1
    step_in_episode = 0

    while completed < args.num_episodes:

        step_in_episode += 1

        # ----------------------------------------------
        # 当前状态诊断
        # 注意：在 env.step() 前记录
        # ----------------------------------------------

        tcp_pos = obs["extra"]["tcp_pose"][0, :3]
        goal_pos = obs["extra"]["goal_pos"][0]

        # simulator privileged state
        # 仅用于分析，不输入 Policy
        cube_pos = task_env.obj.pose.p[0]

        d_tcp_cube = torch.linalg.norm(
            tcp_pos - cube_pos
        ).item()

        d_cube_goal = torch.linalg.norm(
            cube_pos - goal_pos
        ).item()

        d_tcp_goal = torch.linalg.norm(
            tcp_pos - goal_pos
        ).item()

        row = {
            "episode": episode_id,
            "step": step_in_episode,

            "d_tcp_cube": d_tcp_cube,
            "d_cube_goal": d_cube_goal,
            "d_tcp_goal": d_tcp_goal,

            "tcp_x": tcp_pos[0].item(),
            "tcp_y": tcp_pos[1].item(),

            "cube_x": cube_pos[0].item(),
            "cube_y": cube_pos[1].item(),

            "goal_x": goal_pos[0].item(),
            "goal_y": goal_pos[1].item(),

            "done": 0,
            "success_once": "",
            "success_at_end": "",
        }

        rows.append(row)

        # ----------------------------------------------
        # Policy action
        # ----------------------------------------------

        with torch.no_grad():
            action = agent.get_action(
                obs,
                deterministic=True,
            )

            action = torch.clamp(
                action,
                action_low,
                action_high,
            )

        obs, reward, terminated, truncated, infos = (
            env.step(action)
        )

        # ----------------------------------------------
        # Episode end
        # ----------------------------------------------

        if "final_info" not in infos:
            continue

        done_mask = infos["_final_info"]

        if not done_mask.any():
            continue

        ep = infos["final_info"]["episode"]

        success_once = (
            ep["success_once"][done_mask]
            .item()
        )

        success_at_end = (
            ep["success_at_end"][done_mask]
            .item()
        )

        # 给这一 episode 最后一条记录标记结果
        rows[-1]["done"] = 1
        rows[-1]["success_once"] = success_once
        rows[-1]["success_at_end"] = success_at_end

        completed += 1

        print(
            f"episode {completed}: "
            f"success_once={success_once:.0f}, "
            f"success_at_end={success_at_end:.0f}"
        )

        episode_id += 1
        step_in_episode = 0

    # ==================================================
    # 4. Save CSV
    # ==================================================

    csv_path = video_dir / "distances.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    env.close()

    print()
    print("Videos saved to:", video_dir)
    print("Distance log saved to:", csv_path)


if __name__ == "__main__":
    main()