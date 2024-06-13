from omegaconf import OmegaConf
import hydra
from legged_gym.env import Env
import torch

OmegaConf.register_new_resolver("eval", eval)

NO_ENVS = 10

@hydra.main(config_path="../configs/", config_name="defaults", version_base=None)
def eval(cfg):
    cfg.env.num_envs = min(cfg.env.num_envs, NO_ENVS)
    env = Env(cfg)
    for i in range(int(NO_ENVS * env.max_episode_length)):
        actions = 0. * torch.ones(env.num_envs, env.num_actions, device=env.device)
        obs, _, rew, done, info = env.step(actions)
    print("Done")


if __name__ == "__main__":
    eval()
