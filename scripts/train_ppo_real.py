from environments.old.ARSL_env_camera_2 import MicrorobotEnv
import yaml
import gymnasium as gym
from stable_baselines3 import PPO


def main(config_file):
    with open(config_file, 'r') as f:
        env_config = yaml.safe_load(f)

    env = MicrorobotEnv(save_path_experiment= "/home/m4/Documents/PPO_Real_Experiments_Online")  # Save_path
    #env = gym.wrappers.TimeLimit(env, max_episode_steps=env_config['max_episode_steps'])
    #env = gym.wrappers.MaxAndSkipEnv(env, 2)

    model = PPO("CnnPolicy", env, verbose=1, n_epochs=16)
    #model.set_parameters(env_config['model_path']

    model.learn()