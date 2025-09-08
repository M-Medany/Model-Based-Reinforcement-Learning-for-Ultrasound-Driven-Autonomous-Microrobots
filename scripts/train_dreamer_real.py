import warnings
import dreamerv3
from dreamerv3 import embodied
from dreamerv3.embodied.run.parallel import actor, learner
from dreamerv3.embodied.run.parallel import env as env_thread
from dreamerv3.embodied.envs import from_gym, from_dm, atari
from environments.ARSL_env_camera_dreamer import MicrorobotEnv
from environments.env_camera_continous import MicrorobotEnvContinous
from environments.game_env_dreamer_rand_freq import MicrorobotEnvContGameFreq, MicrorobotEnvContinousGame
from environments.env_camera_sweeping import MicrorobotEnvSweeping
from environments.env_camera_sweeping_rrt import RRTEnvSweeping
from environments.costum_wrappers.RateTargetReached import RateTargetReachedWrapper
from environments.costum_wrappers.MaxandSkip import MaxAndSkipEnv
import yaml
import os
import argparse
from gymnasium.wrappers import TimeLimit
from functools import partial as bind
import signal, sys
from utils import close_arduino

env_config={
    "config": "/home/mahmoud/git/DQN_for_Microrobot_control/scripts/config.yaml",
    "default_image_path": "binary_images/binary_images/vascular_cropped.png",  #"/home/mahmoud/git/DQN_for_Microrobot_control/binary_images/default_mask_racetrack.png",
    "default_mask": "binary_images/default_segmentation_vascular_large.png",
    "roi": (700, 180, 831, 831)#(750, 210, 642, 642), # racetrack (686, 269, 550, 550),
            }

env_config_continous = env_config.copy()
env_config_continous.update({
    "config": "/home/mahmoud/git/DQN_for_Microrobot_control/scripts/config_sweep.yaml",
})

NAME = "vascular_real_flow"
MAIN_PATH = f"/home/mahmoud/logdir/{NAME}/images/"
# MAIN_PATH = f"/media/mahmoud/Mahmoud_T7_Touch/Dreamer_Real_v2/{NAME}"
# MAIN_PATH = f"/media/mahmoud/Backup/Backup_ML/Dreamer_Real_v2/{NAME}"
LOGDIR = f"/home/mahmoud/logdir/{NAME}"

def signal_handler(signal, frame):
    print("Ctrl+C pressed. Turning off the Arduino...")
    close_arduino()
    sys.exit(0)

def main(config_file):
    warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')
    signal.signal(signal.SIGINT, signal_handler)

    # See configs.yaml for all options.
    config = embodied.Config(dreamerv3.configs['defaults'])
    config = config.update(dreamerv3.configs['large'])
    config = config.update({
        'logdir': LOGDIR,
        'run.train_ratio': -1, # Run at maximum speed.
        'run.log_every': 60,  # Seconds
        'batch_size': 16,
        'jax.prealloc': True,
        'encoder.mlp_keys': '^$',#'piezo', #'agent_position|target_position',
        'decoder.mlp_keys': '^$',#'piezo', #'agent_position|target_position',
        'encoder.cnn_keys': 'image',
        'decoder.cnn_keys': 'image',
        'jax.platform': 'gpu',
        'run.log_keys_mean': '^log_.*',
        'run.log_keys_max': '^log_.*',
        'run.actor_batch': 1,
        # 'expl_behavior': 'Explore',  # Still have to run with 'Explore' for now
        # 'wrapper': 
    })
    # config["wrapper"].update({"length": 10})
    config = embodied.Flags(config).parse()
    logdir = embodied.Path(config.logdir)
    os.makedirs(logdir, exist_ok=True)
    if os.path.exists(logdir / 'config.yaml'):
        print(f'\033[1;31mExperiment already exists at {logdir}.\033[0m')
    else:
        yaml.dump(config, open(logdir / 'config.yaml', 'w'))
        yaml.dump(env_config, open(logdir / 'env_config.yaml', 'w'))
    
    step = embodied.Counter()
    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.TensorBoardOutput(logdir),
        # embodied.logger.WandBOutput(logdir.name, config),
        # embodied.logger.MLFlowOutput(logdir.name),
    ])
    
    args = embodied.Config(
        **config.run, logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length)
    
    replay = embodied.replay.Uniform(
        config.batch_length, config.replay_size, logdir / 'replay', online=True)
    
    run_parallel(config, logger, args, step, replay, skip=4)
    # run(config, args, logger, step, replay, skip=4)


def run_parallel(config, logger, args, step, replay, skip=4):
    env = make_env(config, skip, fake_env=True)
    env = embodied.BatchEnv([env], parallel=False)
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    env.close()
    embodied.run.parallel(agent, replay, logger, bind(make_env, config, skip, False), 1, args)
    # parallel(agent, replay, logger, bind(make_env, config, skip, True), 1, args, "acting")
    # parallel(agent, replay, logger, bind(make_env, config, skip, True), 1, args, "learning")
    # parallel(agent, replay, logger, bind(make_env, config, skip, True), 1, args, "enviroment")


# def parallel(agent, replay, logger, make_env, num_envs, args, mode):
#   step = logger.step
#   timer = embodied.Timer()
#   timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
#   timer.wrap('replay', replay, ['add', 'save'])
#   timer.wrap('logger', logger, ['write'])
#   workers = []
#   if mode == 'acting':
#     workers.append(embodied.distr.Thread(
#       actor, step, agent, replay, logger, args.actor_addr, args))
#     workers.append(embodied.distr.Thread(
#       env_thread, make_env, args.actor_addr, 0, args, timer))
#     # workers.append(embodied.distr.Thread(
#     #   learner, step, agent, replay, logger, timer, args))
#   elif mode == 'enviroment':
#     workers.append(embodied.distr.Thread(
#       env_thread, make_env, args.actor_addr, 0, args, timer))
#   elif mode == 'learning':
#     workers.append(embodied.distr.Thread(
#       learner, step, agent, replay, logger, timer, args))

#   embodied.distr.run(workers)


def run(config, args, logger, step, replay, skip=4):
    env = make_env(config, skip, fake_env=False)
    env = embodied.BatchEnv([env], parallel=False)

    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)

    embodied.run.train(agent, env, replay, logger, args)
    # checkpoint = embodied.Checkpoint(f"{args.logdir}/checkpoint.ckpt")
    # args = args.update({"from_checkpoint": f"{args.logdir}/checkpoint.ckpt"})
    # embodied.run.eval_only(agent, env, logger, args)

def make_env(config, skip, fake_env=False, i=0):  # i is for parallel envs (when you have multiple like in simulation)
    # print("Making the environment. \n params: ", config, skip, fake_env)
    if fake_env:
        config_fake = env_config_continous.copy()
        # config_fake.update({"image_string": 'binary_images/default_mask_racetrack_segmented.png',
        #                     "config": "scripts/config_sim.yaml",
        #                     "name": (0,0),
        #                     "max_envs": 1,})
        env = MicrorobotEnvSweeping(save_path_experiment=f"{MAIN_PATH}_fake", fake=True, **config_fake)
    else:
        env = MicrorobotEnvSweeping(save_path_experiment=MAIN_PATH, **env_config_continous)
    env = MaxAndSkipEnv(env, skip=skip)
    if not fake_env:
        env = RateTargetReachedWrapper(env, dreamer=True, logdir=f"{LOGDIR}/rate_target_reached.csv")
    env = from_gym.FromGym(env, obs_key='image')
    env = dreamerv3.wrap_env(env, config)
    # env = embodied.BatchEnv([env], parallel=False)
    return env


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='scripts/config.yaml', help='Path to config file')
    args = parser.parse_args()
    main(args.config)