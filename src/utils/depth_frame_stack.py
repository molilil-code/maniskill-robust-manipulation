import torch


class DepthFrameStacker:
    """
    Maintain temporal Depth stacks for ManiSkill parallel environments.

    Raw depth:
        [B, H, W, 1]

    Stacked depth:
        [B, H, W, K]

    The stacked depth is added as:
        obs["depth_stack"]
    """

    def __init__(self, num_frames=4):
        self.num_frames = num_frames
        self.depth_stack = None

    @staticmethod
    def _get_depth(obs):
        """
        Adjust this path only if your current observation uses
        a different camera name.
        """
        return obs["sensor_data"]["base_camera"]["depth"]

    @staticmethod
    def _add_stack_to_obs(obs, depth_stack):
       return {
        "depth_stack": depth_stack,

        "agent": {
            "qpos": obs["agent"]["qpos"],
            "qvel": obs["agent"]["qvel"],
        },

        "extra": {
            "tcp_pose": obs["extra"]["tcp_pose"],
            "goal_pos": obs["extra"]["goal_pos"],
        },
    }
    def reset(self, obs):
        """
        Full reset.

        D0 -> [D0, D0, D0, D0]
        """
        depth = self._get_depth(obs)

        # [B,H,W,1] -> [B,H,W,K]
        self.depth_stack = depth.repeat(
            1, 1, 1, self.num_frames
        )

        return self._add_stack_to_obs(
            obs,
            self.depth_stack,
        )

    def make_terminal_obs(
        self,
        terminal_obs,
        done_mask,
    ):
        """
        Construct the correct frame stack for terminal observations.

        Before step:
          [D_t-3, D_t-2, D_t-1, D_t]

        terminal observation:
          D_terminal

        Result:
          [D_t-2, D_t-1, D_t, D_terminal]

        terminal_obs contains ONLY the done environments.
        done_mask refers to positions in the full vector env.
        """
        terminal_depth = self._get_depth(
            terminal_obs
        )

        old_stack = self.depth_stack[done_mask]

        terminal_stack = torch.cat(
            [
                old_stack[..., 1:],
                terminal_depth,
            ],
            dim=-1,
        )

        return self._add_stack_to_obs(
            terminal_obs,
            terminal_stack,
        )

    def step(
        self,
        obs,
        done_mask,
    ):
        """
        Process observation returned by ManiSkillVectorEnv.step().

        Important:
        For done environments, obs already contains the FIRST
        observation of the automatically reset new episode.
        """
        depth = self._get_depth(obs)

        # --------------------------------------------------
        # Normal environments:
        #
        # [D_t-3,D_t-2,D_t-1,D_t]
        # ->
        # [D_t-2,D_t-1,D_t,D_t+1]
        # --------------------------------------------------
        new_stack = torch.cat(
            [
                self.depth_stack[..., 1:],
                depth,
            ],
            dim=-1,
        )

        # --------------------------------------------------
        # Partial-reset environments:
        #
        # ManiSkill has already reset them.
        #
        # Do NOT keep old episode history.
        #
        # [old, old, old, D_new]   WRONG
        #
        # [D_new,D_new,D_new,D_new] CORRECT
        # --------------------------------------------------
        if torch.any(done_mask):
            new_depth = depth[done_mask]

            new_stack[done_mask] = (
                new_depth.repeat(
                    1,
                    1,
                    1,
                    self.num_frames,
                )
            )

        self.depth_stack = new_stack

        if torch.any(done_mask):

            ids = torch.where(done_mask)[0]

            done_stack = new_stack[
                done_mask
            ]

            assert torch.equal(
                done_stack[..., 0],
                done_stack[..., 1],
            )

            assert torch.equal(
                done_stack[..., 1],
                done_stack[..., 2],
            )

            assert torch.equal(
                done_stack[..., 2],
                done_stack[..., 3],
            )


        return self._add_stack_to_obs(
            obs,
            self.depth_stack,
        )