import numpy as np
import gymnasium as gym
import mani_skill.envs
import sapien.physx as physx

import src.envs


# Panda 官方 rest qpos
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


env = gym.make(
    "PushCubeFullEval-v1",
    condition="full_combined",
    num_envs=1,
    obs_mode="state",
    sim_backend="physx_cpu",
)

task_env = env.unwrapped


# ==================================================
# 1. 检查 Physics OOD
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

static_friction = material.static_friction
dynamic_friction = material.dynamic_friction


print("\n==============================")
print("Full Physics OOD")
print("==============================")

print("mass =", mass)
print("static friction =", static_friction)
print("dynamic friction =", dynamic_friction)


assert abs(mass - 0.096) < 1e-6

assert abs(
    static_friction - 0.15
) < 1e-6

assert abs(
    dynamic_friction - 0.15
) < 1e-6


# ==================================================
# 2. 收集 Episode OOD
# ==================================================

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


# ==================================================
# 3. Cube OOD
# ==================================================

is_outer = (
    (np.abs(cube_xy[:, 0]) >= 0.13)
    |
    (np.abs(cube_xy[:, 1]) >= 0.13)
)

outer_band_ratio = is_outer.mean()


print("\n==============================")
print("Full Episode OOD")
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
    "outer-band ratio:",
    outer_band_ratio,
)


assert is_outer.all()

assert (
    np.abs(cube_xy).max()
    <= 0.160001
)


# ==================================================
# 4. Goal OOD
# ==================================================

print(
    "goal offset range:",
    goal_offset.min(),
    goal_offset.max(),
)

assert goal_offset.min() >= 0.24
assert goal_offset.max() <= 0.26


# ==================================================
# 5. Qpos OOD
# ==================================================

delta = (
    qpos[:, :7]
    - REST_QPOS[:7]
)

qpos_std = delta.std(axis=0)

print(
    "qpos noise mean:",
    delta.mean(axis=0),
)

print(
    "qpos noise std:",
    qpos_std,
)


assert np.all(qpos_std > 0.05)
assert np.all(qpos_std < 0.07)


print()
print("Full OOD test PASSED.")

env.close()