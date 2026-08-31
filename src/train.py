from collections import defaultdict
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import yaml
from torch.utils.tensorboard import SummaryWriter

# ManiSkill specific imports
import mani_skill.envs
import src.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from src.models.agent import Agent
from src.utils.depth_frame_stack import DepthFrameStacker

@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=True`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""

    # Algorithm specific arguments
    env_id: str = "PushCube-v1"
    """the id of the environment"""
    total_timesteps: int = 300000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel environments"""
    num_eval_envs: int = 1
    """the number of parallel evaluation environments"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 200
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 2
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.1
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    eval_freq: int = 200
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    finite_horizon_gae: bool = False


    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    obs_mode: str = "state"
    reward_mode: str = "normalized_dense"
    sim_backend: str = "physx_cpu"

    env_kwargs: dict | None = None
    encoder_type: str = "state" 
    render_backend: str | None = None

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        self.writer.close()

#读取 YAML 
def load_config(config_path):
    args = Args()

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        if not hasattr(args, key):
            raise ValueError(f"Unknown config key: {key}")

        setattr(args, key, value)

    return args

def make_obs_buffer(obs, num_steps):
    if isinstance(obs, dict):
        return {
            k: make_obs_buffer(v, num_steps)
            for k, v in obs.items()
        }

    return torch.empty(
        (num_steps,) + tuple(obs.shape),
        dtype=obs.dtype,
        device=obs.device,
    )


def store_obs(buffer, step, obs):
    if isinstance(obs, dict):
        for k in obs:
            store_obs(buffer[k], step, obs[k])
    else:
        buffer[step].copy_(obs)


def flatten_obs_buffer(buffer):
    if isinstance(buffer, dict):
        return {
            k: flatten_obs_buffer(v)
            for k, v in buffer.items()
        }

    # [num_steps, num_envs, ...]
    # ->
    # [num_steps*num_envs, ...]
    return buffer.reshape(
        (-1,) + buffer.shape[2:]
    )


def index_obs(obs, indices):
    if isinstance(obs, dict):
        return {
            k: index_obs(v, indices)
            for k, v in obs.items()
        }

    return obs[indices]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config file")
    cli_args = parser.parse_args()

    args = load_config(cli_args.config)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name   
    if args.sim_backend == "physx_cpu" and args.num_envs != 1:
        raise ValueError(
            "physx_cpu only supports num_envs=1."
        )


    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        sim_backend=args.sim_backend,
    )

    if args.render_backend is not None:
        env_kwargs["render_backend"] = args.render_backend

    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode

    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    print("=" * 50)
    print("Experiment Configuration")
    print("=" * 50)

    print(f"Environment:      {args.env_id}")
    print(f"Observation:      {args.obs_mode}")
    print(f"Backend:          {args.sim_backend}")
    print(f"Device:           {device}")
    print(f"Num envs:         {args.num_envs}")
    print(f"Num steps:        {args.num_steps}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Total timesteps:  {args.total_timesteps}")

    print("=" * 50)
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.evaluate, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    use_frame_stack = (
        args.encoder_type == "depth_goal_stack4"
    )

    if use_frame_stack:
        train_frame_stacker = DepthFrameStacker(
            num_frames=4
        )

        eval_frame_stacker = DepthFrameStacker(
            num_frames=4
        )
    else:
        train_frame_stacker = None
        eval_frame_stacker = None
        
    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=False)
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group="PPO",
                tags=["ppo", "walltime_efficient"]
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    agent = Agent(envs,encoder_type=args.encoder_type,).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    next_obs, _ = envs.reset(seed=args.seed)

    if use_frame_stack:
        next_obs = train_frame_stacker.reset(
            next_obs
        )

    obs = make_obs_buffer(
        next_obs,
        args.num_steps,
    )
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)


    global_step = 0
    start_time = time.time()
    eval_obs, _ = eval_envs.reset(seed=args.seed)

    if use_frame_stack:
        eval_obs = eval_frame_stacker.reset(
            eval_obs
        )
    next_done = torch.zeros(args.num_envs, device=device)
    print(f"####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print(f"####")
    action_space_low, action_space_high = torch.from_numpy(envs.single_action_space.low).to(device), torch.from_numpy(envs.single_action_space.high).to(device)
    def clip_action(action: torch.Tensor):
        return torch.clamp(action.detach(), action_space_low, action_space_high)

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))

    for iteration in range(1, args.num_iterations + 1):
        print(f"Epoch: {iteration}, global_step={global_step}")
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()
        if iteration % args.eval_freq == 1:
            print("Evaluating")
            eval_obs, _ = eval_envs.reset()
            if use_frame_stack:
                eval_obs = eval_frame_stacker.reset(
                    eval_obs
                )
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_action = agent.get_action(
                    eval_obs,
                    deterministic=True,
                    )

                    eval_action = clip_action(eval_action)

                    (
                    raw_eval_obs,
                    eval_rew,
                    eval_terminations,
                    eval_truncations,
                    eval_infos,
                    ) = eval_envs.step(eval_action)

                    eval_reset_mask = torch.zeros_like(
                        eval_truncations,
                        dtype=torch.bool,
                    )

                    if "final_info" in eval_infos:
                        eval_reset_mask = eval_infos[
                            "_final_info"
                        ].bool()

                    if use_frame_stack:
                        eval_obs = eval_frame_stacker.step(
                            raw_eval_obs,
                            eval_reset_mask,
                        )
                    else:
                        eval_obs = raw_eval_obs
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes")
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
                print(f"eval_{k}_mean={mean}")
            if args.evaluate:
                break
        if args.save_model and iteration % args.eval_freq == 1:
            model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        rollout_time = time.time()
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            store_obs(
                obs,
                step,
                next_obs,
            )
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

          
            (
                raw_next_obs,
                reward,
                terminations,
                truncations,
                infos,
            ) = envs.step(
                clip_action(action)
            )

            # PPO 的 done
            next_done_bool = torch.logical_or(
                terminations,
                truncations,
            )

            next_done = next_done_bool.to(
                torch.float32
            )

            rewards[step] = (
                reward.view(-1)
                * args.reward_scale
            )

            # --------------------------------------------------
            # 哪些环境真正结束并发生 reset
            # --------------------------------------------------
            reset_mask = torch.zeros_like(
                next_done_bool,
                dtype=torch.bool,
            )

            # --------------------------------------------------
            # Terminal observation / final value
            # 必须在 frame stack 更新之前处理
            # --------------------------------------------------
            if "final_info" in infos:

                final_info = infos["final_info"]
                final_info_mask = infos[
                    "_final_info"
                ].bool()

                reset_mask = final_info_mask

                for k, v in final_info["episode"].items():
                    logger.add_scalar(
                        f"train/{k}",
                        v[final_info_mask]
                        .float()
                        .mean(),
                        global_step,
                    )

                with torch.no_grad():

                    done_env_ids = torch.arange(
                        args.num_envs,
                        device=device,
                    )[final_info_mask]

                    final_obs = index_obs(
                        infos["final_observation"],
                        final_info_mask,
                    )

                    # ------------------------------------------
                    # FrameStack terminal observation:
                    #
                    # [Dt-3,Dt-2,Dt-1,Dt]
                    # +
                    # Dterminal
                    #
                    # ->
                    #
                    # [Dt-2,Dt-1,Dt,Dterminal]
                    # ------------------------------------------
                    if use_frame_stack:

                        final_obs = (
                            train_frame_stacker
                            .make_terminal_obs(
                                final_obs,
                                final_info_mask,
                            )
                        )

                    # IMPORTANT:
                    # non-frame-stack 也必须计算！
                    final_values[
                        step,
                        done_env_ids
                    ] = agent.get_value(
                        final_obs
                    ).view(-1)


            # --------------------------------------------------
            # Update frame stack
            #
            # 必须每一个 timestep 都执行
            # --------------------------------------------------
            if use_frame_stack:

                next_obs = train_frame_stacker.step(
                    raw_next_obs,
                    reset_mask,
                )

            else:

                next_obs = raw_next_obs

        rollout_time = time.time() - rollout_time
        # bootstrap value according to termination and truncation
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t] # t instead of t+1
                # next_not_done means nextvalues is computed from the correct next_obs
                # if next_not_done is 1, final_values is always 0
                # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                if args.finite_horizon_gae:
                   
                    if t == args.num_steps - 1: # initialize
                        lam_coef_sum = 0.
                        reward_term_sum = 0. # the sum of the second term
                        value_term_sum = 0. # the sum of the third term
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
            returns = advantages + values

        # flatten the batch
        b_obs = flatten_obs_buffer(obs)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_time = time.time()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mb_obs = index_obs(
                    b_obs,
                    mb_inds,
                )

                _, newlogprob, entropy, newvalue = (
                    agent.get_action_and_value(
                        mb_obs,
                        b_actions[mb_inds],
                    )
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        update_time = time.time() - update_time

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
        logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        logger.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)
    if not args.evaluate:
        if args.save_model:
            model_path = f"runs/{run_name}/final_ckpt.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        logger.close()
    envs.close()
    eval_envs.close()
