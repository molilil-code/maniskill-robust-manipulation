import numpy as np
import gymnasium as gym
import mani_skill.envs
import sapien.physx as physx

import src.envs


env = gym.make(
    "PushCubeFullDR-v1",
    num_envs=1,
    obs_mode="state",
    sim_backend="physx_cpu",
)

task_env = env.unwrapped


# ==================================================
# Part 1
# Physics DR
# ==================================================

entity = task_env.obj._objs[0]

body = entity.find_component_by_type(
    physx.PhysxRigidDynamicComponent
)

mass = body.mass

material = (
    body.collision_shapes[0]
    .physical_material
)

friction = material.static_friction

print("\n==============================")
print("Physics DR")
print("==============================")

print("mass =", mass)
print("friction =", friction)


assert 0.0448 <= mass <= 0.0832
assert 0.21 <= friction <= 0.39


# ==================================================
# Part 2
# Episode DR
# ==================================================

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


cube_xy_list = []
goal_offset_list = []
qpos_list = []


for _ in range(1000):

    env.reset()

    cube = (
        task_env.obj.pose.p[0]
        .cpu()
        .numpy()
    )

    goal = (
        task_env.goal_region.pose.p[0]
        .cpu()
        .numpy()
    )

    qpos = (
        task_env.agent.robot
        .get_qpos()[0]
        .cpu()
        .numpy()
    )

    cube_xy_list.append(
        cube[:2]
    )

    goal_offset_list.append(
        goal[0] - cube[0]
    )

    qpos_list.append(
        qpos
    )


cube_xy = np.asarray(
    cube_xy_list
)

goal_offset = np.asarray(
    goal_offset_list
)

qpos = np.asarray(
    qpos_list
)


print("\n==============================")
print("Episode DR")
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
    "goal offset:",
    goal_offset.min(),
    goal_offset.max(),
)


delta = (
    qpos[:, :7]
    - REST_QPOS[:7]
)

print(
    "qpos std:",
    delta.std(axis=0)
)


env.close()