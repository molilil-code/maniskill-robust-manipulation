import gymnasium as gym
import mani_skill.envs

import src.envs

# 这里的 component 类型，
# 请直接使用你 test_nominal_physics.py 中已经成功使用的那个
import sapien.physx as physx


env = gym.make(
    "PushCubePhysicsDR-v1",
    num_envs=1,
    obs_mode="state",
    sim_backend="physx_cpu",
)

obs, info = env.reset()

task_env = env.unwrapped

# -----------------------------
# 1. 查看我们采样出来的参数
# -----------------------------
sampled_mass = task_env.sampled_masses[0]
sampled_friction = task_env.sampled_frictions[0]

print("sampled mass =", sampled_mass)
print("sampled friction =", sampled_friction)


# -----------------------------
# 2. 从真正的 PhysX 对象重新读取
# -----------------------------
entity = task_env.obj._objs[0]

body = entity.find_component_by_type(
    physx.PhysxRigidDynamicComponent
)

actual_mass = body.mass

shape = body.collision_shapes[0]
material = shape.physical_material

actual_static_friction = material.static_friction
actual_dynamic_friction = material.dynamic_friction

print()
print("actual mass =", actual_mass)
print("actual static friction =", actual_static_friction)
print("actual dynamic friction =", actual_dynamic_friction)


# -----------------------------
# 3. 自动验证
# -----------------------------
assert 0.0448 <= actual_mass <= 0.0832

assert 0.21 <= actual_static_friction <= 0.39
assert 0.21 <= actual_dynamic_friction <= 0.39

assert abs(actual_mass - sampled_mass) < 1e-6
assert abs(actual_static_friction - sampled_friction) < 1e-6
assert abs(actual_dynamic_friction - sampled_friction) < 1e-6

print()
print("Physics DR test PASSED.")

env.close()