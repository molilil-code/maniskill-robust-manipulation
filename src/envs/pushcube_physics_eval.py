from mani_skill.utils.registration import register_env

from .pushcube_physics_dr import PushCubePhysicsDREnv


@register_env("PushCubePhysicsEval-v1", max_episode_steps=50)
class PushCubePhysicsEvalEnv(PushCubePhysicsDREnv):

    VALID_CONDITIONS = {
        "normal",
        "mass_low",
        "mass_high",
        "friction_low",
        "friction_high",
        "physics_combined",
    }

    def __init__(
        self,
        *args,
        condition="normal",
        **kwargs,
    ):
        if condition not in self.VALID_CONDITIONS:
            raise ValueError(
                f"Unknown condition: {condition}"
            )

        self.condition = condition

        # Official nominal physics
        nominal_mass = 0.064
        nominal_friction = 0.30

        # ----------------------------
        # 默认：完全官方 physics
        # ----------------------------
        mass = nominal_mass
        friction = nominal_friction

        # ----------------------------
        # 单因素 OOD
        # ----------------------------
        if condition == "mass_low":
            mass = 0.032

        elif condition == "mass_high":
            mass = 0.096

        elif condition == "friction_low":
            friction = 0.15

        elif condition == "friction_high":
            friction = 0.45

        # ----------------------------
        # 组合 Physics OOD
        # ----------------------------
        elif condition == "physics_combined":
            mass = 0.096
            friction = 0.15

        # low == high
        # → 不再随机，而是固定测试值
        super().__init__(
            *args,
            mass_range=(mass, mass),
            friction_range=(friction, friction),
            **kwargs,
        )