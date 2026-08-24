import gymnasium as gym
import mani_skill.envs

import src.envs


env = gym.make(
    "PushCubeDR-v1",
    num_envs=1,
    obs_mode="state",
    sim_backend="physx_cpu",
    cube_xy_range=0.13,
    goal_offset_range=(0.17, 0.23),
    robot_init_qpos_noise=0.04,
)

base_env = env.unwrapped

print(
    "cube_xy_range:",
    base_env.cube_xy_range,
)

print(
    "goal_offset_range:",
    base_env.goal_offset_range,
)

for i in range(10):
    obs, info = env.reset()

    base_env = env.unwrapped

    cube_pos = base_env.obj.pose.p[0].cpu().numpy()
    goal_pos = base_env.goal_region.pose.p[0].cpu().numpy()

    goal_distance = goal_pos[0] - cube_pos[0]
    qpos = base_env.agent.robot.get_qpos()[0].cpu().numpy()
    print(
        f"Episode {i:02d} | "
        f"cube={cube_pos[:2]} | "
        f"goal={goal_pos[:2]} | "
        f"goal_offset={goal_distance:.3f}"
        f"qpos={qpos}"
    )


env.close()