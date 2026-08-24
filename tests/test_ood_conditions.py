import numpy as np
import gymnasium as gym
import mani_skill.envs

import src.envs


def test_condition(condition, num_episodes=100):
    env = gym.make(
        "PushCubeEval-v1",
        condition=condition,
        num_envs=1,
        obs_mode="state",
        sim_backend="physx_cpu",
    )

    base_env = env.unwrapped

    cube_xy_list = []
    goal_offset_list = []
    qpos_list = []

    for _ in range(num_episodes):
        env.reset()

        cube = (
            base_env.obj.pose.p[0]
            .cpu()
            .numpy()
        )

        goal = (
            base_env.goal_region.pose.p[0]
            .cpu()
            .numpy()
        )

        qpos = (
            base_env.agent.robot.get_qpos()[0]
            .cpu()
            .numpy()
        )

        cube_xy_list.append(cube[:2])

        goal_offset_list.append(
            goal[0] - cube[0]
        )

        qpos_list.append(qpos)

    env.close()

    cube_xy = np.asarray(cube_xy_list)
    goal_offset = np.asarray(goal_offset_list)
    qpos = np.asarray(qpos_list)

    print("\n==============================")
    print("Condition:", condition)
    print("==============================")

    print(
        "cube x range:",
        cube_xy[:, 0].min(),
        cube_xy[:, 0].max(),
    )

    print(
        "cube y range:",
        cube_xy[:, 1].min(),
        cube_xy[:, 1].max(),
    )

    print(
        "goal offset range:",
        goal_offset.min(),
        goal_offset.max(),
    )

    if condition in {"cube_ood", "combined"}:
        is_outer = (
        (np.abs(cube_xy[:, 0]) >= 0.13)
        | (np.abs(cube_xy[:, 1]) >= 0.13)
        )

        print("outer-band ratio:", is_outer.mean())

        assert is_outer.all()
        assert np.abs(cube_xy).max() <= 0.160001

        return cube_xy, goal_offset, qpos

    REST_QPOS = np.array([
    0.0,
    np.pi / 8,
    0.0,
    -5 * np.pi / 8,
    0.0,
    3 * np.pi / 4,
    np.pi / 4,
    0.04,
    0.04,
    ])

    if condition in {"qpos_shift", "combined"}:
        delta = qpos[:, :7] - REST_QPOS[:7]

        print("qpos noise mean:", delta.mean(axis=0))
        print("qpos noise std:", delta.std(axis=0))
 
if __name__ == "__main__":

    for condition in [
        "normal",
        "cube_ood",
        "goal_near",
        "goal_far",
        "qpos_shift",
        "combined",
    ]:
        test_condition(
            condition,
            num_episodes=1000,
        )

