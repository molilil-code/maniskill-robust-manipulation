import gymnasium as gym
import mani_skill.envs
import sapien.physx as physx

import src.envs


conditions = [
    "normal",
    "mass_low",
    "mass_high",
    "friction_low",
    "friction_high",
    "physics_combined",
]


for condition in conditions:

    env = gym.make(
        "PushCubePhysicsEval-v1",
        condition=condition,
        num_envs=1,
        obs_mode="state",
        sim_backend="physx_cpu",
    )

    env.reset()

    task_env = env.unwrapped

    entity = task_env.obj._objs[0]

    body = entity.find_component_by_type(
        physx.PhysxRigidDynamicComponent
    )

    mass = body.mass

    material = body.collision_shapes[0].physical_material

    static_friction = material.static_friction
    dynamic_friction = material.dynamic_friction

    expected_mass, expected_friction = expected[condition]

    assert abs(mass - expected_mass) < 1e-6

    assert abs(

    static_friction - expected_friction
    ) < 1e-6

    assert abs(
    dynamic_friction - expected_friction
    ) < 1e-6

    print("\n==============================")
    print("Condition:", condition)
    print("==============================")

    print("mass =", mass)
    print("static friction =", static_friction)
    print("dynamic friction =", dynamic_friction)

    env.close()