import os

from omegaconf import OmegaConf
import hydra

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.env import Env
from legged_gym.rl import OnPolicyRunner
from legged_gym.utils import Logger
from legged_gym.utils.helpers import get_load_path, export_policy_as_jit
import torch

OmegaConf.register_new_resolver("eval", eval)

EXPORT_POLICY = True


@hydra.main(config_path="../configs/", config_name="defaults", version_base=None)
def play(cfg):
    cfg.env.num_envs = min(cfg.env.num_envs, 50)
    cfg.terrain.num_rows = 6
    cfg.terrain.num_cols = 6
    cfg.terrain.curriculum = True
    cfg.terrain.mesh_type = "trimesh"
    cfg.terrain.terrain_proportions = [0.0, 0.0, 0.0, 0.0, 1.0]  # maximum difficulty
    cfg.env.priv_observe_contact_forces = False
    cfg.env.priv_observe_base_lin_vel = False
    cfg.noise.add_noise = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.randomize_gravity = False
    cfg.domain_rand.push_robots = False
    cfg.commands.heading_command = False
    cfg.domain_rand.randomize_base_com = False
    cfg.domain_rand.lag_timesteps = 6
    cfg.domain_rand.randomize_lag_timesteps = True
    cfg.control.control_type = "actuator_net"

    env = Env(cfg)
    ppo = OnPolicyRunner(env, cfg.ppo, log_dir=None, device=cfg.experiment.device, wandb=None)

    # code for resuming
    if cfg.ppo.runner.load_run != -1:
        model_path = get_load_path(cfg.ppo.runner.load_run)
        print("==== LOADING MODEL:", model_path)
        ppo.load(model_path)
    else:
        raise Exception(
            "You need to specify which model you wanna load. Add `ppo.runner.load_run=./logs/DATETIME` "
            "to the command line call."
        )

    policy = ppo.get_inference_policy(device=env.device)
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, cfg.ppo.runner.load_run)
        export_policy_as_jit(ppo.alg.actor_critic, path)
        print("=== Exported policy as jit script to: ", path)

    ## eval loop with logging and plotting

    logger = Logger(cfg.sim.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    stop_state_log = 100  # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1  # number of steps before print average episode rewards

    # ===  hardcoded test command
    x_vel_cmd, y_vel_cmd, yaw_vel_cmd, body_height_cmd = 0.3, 0.0, 0.0, 0.0
    print(f"=== RUNNING COMMAND [{x_vel_cmd}, {y_vel_cmd}, {yaw_vel_cmd}, {body_height_cmd}]")

    obs = env.get_observations()

    for i in range(10 * int(env.max_episode_length)):
        actions = policy(obs.detach())
        env.commands[:, 0] = x_vel_cmd
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd
        env.commands[:, 3] = body_height_cmd
        obs, _, rews, dones, infos = env.step(actions.detach())

        if i < stop_state_log:
            # print("proj grav : ", env.projected_gravity)
            logger.log_states(
                {
                    "dof_pos_target": actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    "dof_pos": env.dof_pos[robot_index, joint_index].item(),
                    "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                    "dof_torque": env.torques[robot_index, joint_index].item(),
                    "command_x": env.commands[robot_index, 0].item(),
                    "command_y": env.commands[robot_index, 1].item(),
                    "command_yaw": env.commands[robot_index, 2].item(),
                    "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                    "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                    "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                    "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                    "contact_forces_z": env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                    "projected_gravity_x": env.projected_gravity[robot_index, 0].item(),
                    "projected_gravity_y": env.projected_gravity[robot_index, 1].item(),
                    "projected_gravity_z": env.projected_gravity[robot_index, 2].item(),
                }
            )
        elif i == stop_state_log:
            # print("proj grav : ", env.projected_gravity)
            # print("base vel x : ", env.base_lin_vel[robot_index, 0])
            # print("base vel x item : ", env.base_lin_vel[robot_index, 0].item()
            logger.plot_states()
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()


if __name__ == "__main__":
    play()
