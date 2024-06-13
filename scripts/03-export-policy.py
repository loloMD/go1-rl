import os
from omegaconf import OmegaConf
import hydra
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.env import Env
from legged_gym.rl import OnPolicyRunner
from legged_gym.utils.helpers import get_load_path, export_policy_as_jit

OmegaConf.register_new_resolver("eval", eval)


@hydra.main(config_path="../configs/", config_name="defaults", version_base=None)
def play(cfg):
    cfg.env.num_envs = min(cfg.env.num_envs, 50)
    cfg.terrain.num_rows = 6
    cfg.terrain.num_cols = 6
    cfg.terrain.curriculum = False

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

    # export policy as a jit module (used to run it from C++)

    path = os.path.join(LEGGED_GYM_ROOT_DIR, cfg.ppo.runner.load_run)
    export_policy_as_jit(ppo.alg.actor_critic, path)
    print("=== Exported policy as jit script to: ", path)


if __name__ == "__main__":
    play()
