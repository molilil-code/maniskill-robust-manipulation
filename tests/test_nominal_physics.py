import gymnasium as gym
import mani_skill.envs
import sapien.physx as physx


env = gym.make(
    "PushCube-v1",
    num_envs=1,
    obs_mode="state",
    sim_backend="physx_cpu",
)

env.reset()

task_env = env.unwrapped

# ManiSkill Actor 管理的底层 SAPIEN entity
entity = task_env.obj._objs[0]

body = entity.find_component_by_type(
    physx.PhysxRigidDynamicComponent
)

print("mass =", body.mass)

for i, shape in enumerate(body.collision_shapes):
    material = shape.physical_material

    print(f"shape {i}")
    print("static friction =", material.static_friction)
    print("dynamic friction =", material.dynamic_friction)
    print("restitution =", material.restitution)

env.close()