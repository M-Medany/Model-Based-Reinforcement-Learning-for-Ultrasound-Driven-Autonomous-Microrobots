import pathlib

from environments import MicrorobotEnvGame8Act, MicrorobotEnvGameByTheWall, MicrorobotEnvGameNoCollision
from environments.game_env_dreamer_cont import MicrorobotEnvContGame
from environments.game_env_dreamer_rand_freq import MicrorobotEnvContGameFreq
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from environments.costum_wrappers.MaxandSkip import MaxAndSkipEnv
from environments.costum_wrappers.RateTargetReached import RateTargetReachedCallback, RateTargetReachedWrapper, RewardWrapper
from environments.costum_wrappers.NewApi import NewApi
import os, yaml

EXP_NAME = "PPO_racetrack_collision"
ckp = ""  # "/media/m4/Backup/logdir/PPO_vascular_8_actions_by_wall_no_flow/PPO_vascular_8_actions_by_wall_no_flow_3500000_steps.zip" #"/home/m4/DQN_for_Microrobot_control/logdir/PPO/racetrack_3/racetrack_3_200000_steps.zip"
LOGDIR = '/media/m4/Backup/logdir'

env_config={
    "name": (0, 0),
    "config": "scripts/config_sim_6_envs.yaml",
    "render_mode": None,
    "render_fps": 0,
    "max_envs": 1,
    "image_string": 'binary_images/default_mask_racetrack_segmented.png', # binary_images/closing1.png
    "subepisode_sampling": True,
    "subepisode_length": 2,
    }
# env_config["inv_img_path"] = "binary_images/vascular_dilated.png"
eval_env_config = env_config.copy()


def main():
    truncation = 100
    path = pathlib.Path(f'{LOGDIR}').absolute().resolve()
    assert not pathlib.Path(path, EXP_NAME).exists(), f"Change experiment name, {path}/{EXP_NAME} already exists"
    assert "PPO" in EXP_NAME, "Change experiment name, must contain PPO"
    os.mkdir(f"{path}/{EXP_NAME}")
    env = MicrorobotEnvGame8Act(**env_config)
    eval_env = MicrorobotEnvGame8Act(**eval_env_config)
    def wrap_env(env, eval="train"):
        env = MaxAndSkipEnv(env, 4)
        env = RateTargetReachedWrapper(env, 100, logdir=f"{path}/{EXP_NAME}/rate_target_reached_{eval}.csv")
        env = RewardWrapper(env, logdir=f"{path}/{EXP_NAME}/reward_{eval}.csv")
        env = NewApi(env)
        env = gym.wrappers.StepAPICompatibility(env)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=truncation)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    env = wrap_env(env, "train")
    eval_env = wrap_env(eval_env, "eval")
    callback = CallbackList([
        CheckpointCallback(save_freq=100000, save_path=f"{path}/{EXP_NAME}", name_prefix=EXP_NAME),
        EvalCallback(eval_env, best_model_save_path=f"{path}/{EXP_NAME}", log_path=f"{path}/{EXP_NAME}", eval_freq=2000, deterministic=True, render=True,
                     n_eval_episodes=25
                     ),
        RateTargetReachedCallback(),
    ])
    env_config_copy = env_config.copy()
    config = env_config_copy["config"]
    with open(config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    config_path = f"{path}/{EXP_NAME}/config.yaml"
    with open(config_path, 'w') as stream:
        try:
            yaml.safe_dump(config, stream)
        except yaml.YAMLError as exc:
            print(exc)
    with open(f"{path}/{EXP_NAME}/env_config.yaml", 'w') as stream:
        try:
            yaml.safe_dump(env_config, stream)
        except yaml.YAMLError as exc:
            print(exc)

    if ckp != "":
        model = PPO.load(ckp, env=env)
    else:
        model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=f"{path}/{EXP_NAME}", device="cuda")

    model.learn(tb_log_name=EXP_NAME, total_timesteps=100e6, progress_bar=True, callback=callback, reset_num_timesteps=False)

if __name__ == "__main__":
    main()
