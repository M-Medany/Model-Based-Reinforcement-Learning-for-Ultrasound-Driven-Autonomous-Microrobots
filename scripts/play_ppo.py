import pathlib

from environments import MicrorobotEnvGame8Act, MicrorobotEnvGameByTheWall
from environments.game_env_dreamer_cont import MicrorobotEnvContGame
from environments.game_env_dreamer_rand_freq import MicrorobotEnvContGameFreq
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from environments.costum_wrappers.MaxandSkip import MaxAndSkipEnv
from environments.costum_wrappers.RateTargetReached import RateTargetReachedCallback, RateTargetReachedWrapper, RewardWrapper
from environments.costum_wrappers.NewApi import NewApi
import os
import tqdm, yaml
import cProfile

EXP_NAME = "PPO_vascular_no_collision_play"
ckp = "/media/m4/Backup/logdir/PPO_vascular_8_actions_by_wall_no_flow/PPO_vascular_8_actions_by_wall_no_flow_20100000_steps.zip"
folder = "/media/m4/Backup/logdir/PPO_vascular_8_actions_by_wall_no_flow"
LOGDIR = '/media/m4/Backup/play_logdir'


def main(episodes=10000):
    truncation = 100
    path = pathlib.Path(f'{LOGDIR}').absolute().resolve()
    assert not pathlib.Path(path, EXP_NAME).exists(), f"Change experiment name, {path}/{EXP_NAME} already exists"
    assert "PPO" in EXP_NAME, "Change experiment name, must contain PPO"
    assert "play" in EXP_NAME, "Change experiment name, must contain play"
    os.mkdir(f"{path}/{EXP_NAME}")
    env_config = yaml.load(open(f"{folder}/env_config.yaml", "r"), Loader=yaml.FullLoader)
    full_config = yaml.load(open(env_config["config"], "r"), Loader=yaml.FullLoader)
    env = MicrorobotEnvGameByTheWall(**env_config)
    def wrap_env(env, eval="train"):
        env = MaxAndSkipEnv(env, 4)
        env = RateTargetReachedWrapper(env, 100, logdir=f"{path}/{EXP_NAME}/rate_target_reached_{eval}.csv")
        env = RewardWrapper(env, logdir=f"{path}/{EXP_NAME}/reward_{eval}.csv")
        env = NewApi(env)
        env = gym.wrappers.StepAPICompatibility(env)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=truncation)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    env = wrap_env(env, "play")
    yaml.dump(env_config, open(f"{path}/{EXP_NAME}/env_config.yaml", "w"))
    yaml.dump(full_config, open(f"{path}/{EXP_NAME}/full_config.yaml", "w"))
    os.system(f"cp {env_config['config']} {path}/{EXP_NAME}")
    open(f"{path}/{EXP_NAME}/checkpoint.txt", "w").write("Checkpoint: " + ckp)

    try:
        model = PPO.load(ckp, env=env)
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint {ckp} not found")
    
    for i in tqdm.tqdm(range(episodes), total=episodes, leave=True):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            total_reward += reward
        tqdm.tqdm.write(f"Episode: {i}, Total reward: {total_reward}")  # Add this line

if __name__ == "__main__":
    # cProfile.run('main()', 'output.prof')
    main()