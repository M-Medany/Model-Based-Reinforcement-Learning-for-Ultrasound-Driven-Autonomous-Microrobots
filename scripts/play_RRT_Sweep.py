from environments.env_camera_sweeping_rrt import RRTEnvSweeping
from environments.costum_wrappers.RateTargetReached import RateTargetReachedWrapper
from environments.costum_wrappers.MaxandSkip import MaxAndSkipEnv
import os
import argparse
import signal, sys
from utils import close_arduino


env_config={
    "config": "/home/m4/git/DQN_for_Microrobot_control/scripts/config_sweep_rrt.yaml",
    "default_image_path": "binary_images/default_overlay_vascular_rotate.png",  #"/home/m4/git/DQN_for_Microrobot_control/binary_images/default_mask_racetrack.png",
    "default_mask": "binary_images/default_segmentation_vascular_4_closed.png",
    "roi": (750, 210, 642, 642), # racetrack (686, 269, 550, 550),
            }

NAME = "vascular_sweeping_RRT"
MAIN_PATH = f"/media/m4/Mahmoud_T7_Touch/Dreamer_Real_v2/{NAME}"
LOGDIR = f"/media/m4/Backup/logdir/{NAME}"

def signal_handler(a, b):
    print("Ctrl+C pressed. Turning off the Arduino...")
    close_arduino()
    sys.exit(0)

def main(config_file):
    signal.signal(signal.SIGINT, signal_handler)
    os.makedirs(LOGDIR, exist_ok=True)

    env = make_env(4)
    obs = env.reset()
    i = 0
    
    while True:
        i+=1
        obs, rew, done, info = env.step(None)
        if i == 200:
            obs = env.reset()
            i = 0
        if done:
            i = 0
            obs = env.reset()


def make_env(skip):  # i is for parallel envs (when you have multiple like in simulation)
    env = RRTEnvSweeping(save_path_experiment=MAIN_PATH, **env_config)
    env = MaxAndSkipEnv(env, skip=skip)
    env = RateTargetReachedWrapper(env, dreamer=True, logdir=f"{LOGDIR}/rate_target_reached.csv")
    return env


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='scripts/config.yaml', help='Path to config file')
    args = parser.parse_args()
    main(args.config)
