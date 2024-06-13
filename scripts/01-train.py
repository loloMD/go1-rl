import wandb
from omegaconf import OmegaConf
import hydra

from legged_gym.env import Env
from legged_gym.rl import OnPolicyRunner
from legged_gym.utils.helpers import setSeed, createLogDir, get_load_path

OmegaConf.register_new_resolver("eval", eval)


@hydra.main(config_path="../configs/", config_name="defaults", version_base=None)
def train(cfg):
    # print(OmegaConf.to_yaml(cfg))
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, config=config)

    setSeed(cfg.experiment.seed)
    logdir = createLogDir(cfg)

    env = Env(cfg)
    ppo = OnPolicyRunner(env, cfg.ppo, log_dir=logdir, device=cfg.experiment.device, wandb=wandb)

    # code for resuming
    if cfg.ppo.runner.load_run != -1:
        model_path = get_load_path(cfg.ppo.runner.load_run)
        print("==== RESUMING TRAINING OF MODEL:", model_path)
        ppo.load(model_path)

    ppo.learn(num_learning_iterations=cfg.ppo.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    train()
