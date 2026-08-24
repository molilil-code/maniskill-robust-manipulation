import numpy as np
import sapien
import sapien.physx as physx

from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor


@register_env("PushCubePhysicsDR-v1", max_episode_steps=50)
class PushCubePhysicsDREnv(PushCubeEnv):

    def __init__(
        self,
        *args,
        mass_range=(0.0448, 0.0832),
        friction_range=(0.21, 0.39),
        **kwargs,
):
        self.mass_range = mass_range
        self.friction_range = friction_range

        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):

        self.table_scene = TableSceneBuilder(
            env=self,
            robot_init_qpos_noise=self.robot_init_qpos_noise,
        )

        self.table_scene.build()

        # 每个env单独创建cube

        cubes = []

        self.sampled_masses = []
        self.sampled_frictions = []

        for i in range(self.num_envs):

            builder = self.scene.create_actor_builder()

            builder.set_scene_idxs([i])

            builder.add_box_collision(
                half_size=[self.cube_half_size] * 3
            )

            builder.add_box_visual(
                half_size=[self.cube_half_size] * 3,
                material=sapien.render.RenderMaterial(
                    base_color=np.array(
                        [12, 42, 160, 255]
                    ) / 255
                ),
            )

            builder.set_initial_pose(
                sapien.Pose(
                    p=[0, 0, self.cube_half_size]
                )
            )

            cube = builder.build(
                name=f"cube_{i}"
            )

            # 单独 object 不作为最终 state-dict actor
            self.remove_from_state_dict_registry(cube)

            #  采样physics parameters

            rng = self._batched_episode_rng[i]

            mass = rng.uniform(
                low=self.mass_range[0],
                high=self.mass_range[1],
            )

            friction = rng.uniform(
                self.friction_range[0],
                self.friction_range[1],
            )

            # 修改这个env的cube physic

            entity = cube._objs[0]

            rigid_body = entity.find_component_by_type(
                physx.PhysxRigidDynamicComponent
            )

            rigid_body.mass = float(mass)

            for shape in rigid_body.collision_shapes:
                material = shape.physical_material

                material.static_friction = float(friction)
                material.dynamic_friction = float(friction)
                material.restitution = 0.0

            self.sampled_masses.append(float(mass))
            self.sampled_frictions.append(float(friction))  

            cubes.append(cube)

        # 把所有cube合成一个 ManiSkill Actor

        self.obj = Actor.merge(
            cubes,
            name="cube",
        )

        self.add_to_state_dict_registry(
            self.obj
        )



        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(
                p=[0, 0, 1e-3]
            ),
        )