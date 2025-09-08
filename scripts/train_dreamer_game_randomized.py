import sys
import os

# Add the path to the library
library_path = os.path.abspath("/home/mahmoud/git/dreamer/dreamerv3")
if library_path not in sys.path:
    sys.path.append(library_path)


import warnings
import dreamerv3
import embodied
import dreamerv3.embodied.envs # import embodied.envs
from dreamerv3.embodied.envs import from_gym, from_dm, atari # from embodied.envs import from_gym, from_dm, atari
from environments import MicrorobotEnvGameRayWrappedCont, MicrorobotEnvContGame, MicrorobotEnvGameByTheWall, MicrorobotEnvGame8Act, MicrorobotEnvGameRandomized
from environments import env_cls_dict
import yaml
from environments.costum_wrappers.MaxandSkip import MaxAndSkipEnv
from environments.costum_wrappers.RateTargetReached import RateTargetReachedWrapper, RewardWrapper


LOGDIR = '/home/mahmoud/logdir/randomized_envs_debug12'
ENV_NAME = "MicrorobotEnvGameByTheWall"
num_envs = 1
yaml_config = "config_sim_no_collision"

env_config_racetrack={
    "env_name": ENV_NAME,
    "image_string": 'binary_images/default_mask_racetrack_segmented.png', # binary_images/closing1.png
    "config": f"scripts/{yaml_config}.yaml",
    "max_envs": 1,
    "render_mode": "human",
    "name": (0,0),
    # "random_env_config": {
    #     'RECTANGLES': {
    #         'MIN_WIDTH': 5,
    #         'MAX_WIDTH': 20,
    #         "NUM": 12
    #     },
    #     'CIRCLES': {
    #         'MIN_RADIUS': 5,
    #         'MAX_RADIUS': 20,
    #         "NUM": 10
    #     },
    #     'NARROW_PASSAGES': {
    #         'MIN_WIDTH': 2,
    #         'MAX_WIDTH': 7,
    #         'MIN_DISTANCE': 10,
    #         'MAX_DISTANCE': 75,
    #         "NUM": 5
    #     },
    #     "REFINE_RATIO": 3,
    #     "DILATE_KERNEL": (13, 13),
    #     "ERODE_KERNEL": (3, 3)
    # }
    }
# env_config_racetrack["inv_img_path"] = 'binary_images/default_mask_racetrack_segmented_inv_dil.png'

env_config_4_squares=env_config_racetrack.copy()
env_config_4_squares["image_string"] = 'binary_images/4_squares.png'
# env_config_4_squares["inv_img_path"] = 'binary_images/4_squares_inv_dil.png'
env_config_4_squares["name"] = (0,0)

env_config_simple=env_config_racetrack.copy()
env_config_simple["image_string"] = 'binary_images/simple.png'
env_config_simple["inv_img_path"] = 'binary_images/simple_inv_dil.png'
env_config_simple["name"] = (0,0)

env_config_vascular=env_config_racetrack.copy()
env_config_vascular["image_string"] = 'binary_images/default_segmentation_vascular.png'
env_config_vascular["name"] = (0,0)

env_config_vascular_real=env_config_racetrack.copy()
env_config_vascular_real["image_string"] = 'binary_images/default_segmentation_vascular_4_closed.png'
env_config_vascular_real["name"] = (0,0)

env_config_vascular_bis=env_config_racetrack.copy()
env_config_vascular_bis["image_string"] = 'binary_images/default_segmentation_vascular_fake.png'
env_config_vascular_bis["inv_img_path"] = 'binary_images/default_segmentation_vascular_fake_inv_dil.png'
env_config_vascular_bis["name"] =(0,0)

env_config_9_squares=env_config_racetrack.copy()
env_config_9_squares["image_string"] = 'binary_images/closing_cleaned.png'
env_config_9_squares["inv_img_path"] = 'binary_images/closing_cleaned_inv_dil.png'
env_config_9_squares["name"] = (0,0)

env_config_maze_1=env_config_racetrack.copy()
env_config_maze_1["image_string"] = 'binary_images/maze_1.png'
env_config_maze_1["inv_img_path"] = 'binary_images/maze_1_inv_dil.png'
env_config_maze_1["name"] = (0,0)

env_config_maze_2=env_config_racetrack.copy()
env_config_maze_2["image_string"] = 'binary_images/maze_2.png'
env_config_maze_2["inv_img_path"] = 'binary_images/maze_2_inv_dil.png'
env_config_maze_2["name"] = (0,0)

env_config_maze_3=env_config_racetrack.copy()
env_config_maze_3["image_string"] = 'binary_images/maze_3.png'
env_config_maze_3["inv_img_path"] = 'binary_images/maze_3_inv_dil.png'
env_config_maze_3["name"] = (0,0)

env_config_maze_4=env_config_racetrack.copy()
env_config_maze_4["image_string"] = 'binary_images/maze_4.png'
env_config_maze_4["inv_img_path"] = 'binary_images/maze_4_inv_dil.png'
env_config_maze_4["name"] = (0,0)

env_config_large_vascular=env_config_racetrack.copy()
env_config_large_vascular["image_string"] = "binary_images/default_segmentation_vascular_large.png"

env_config_large_vascular_by_wall=env_config_racetrack.copy()
env_config_large_vascular_by_wall["image_string"] = "binary_images/default_segmentation_vascular_large.png"
# env_config_large_vascular_by_wall["inv_img_path"] = "binary_images/vascular_dilated.png"

env_config_spa=env_config_racetrack.copy()
env_config_spa["image_string"] = "example_new_images/segmented/spa_segmented.png"

env_config_4_out=env_config_racetrack.copy()
env_config_4_out["image_string"] = "example_new_images/segmented/vascular_multiout_segmented.png"

env_config_self_segmented = env_config_racetrack.copy()
env_config_self_segmented["image_string"] = '/home/mahmoud/git/DQN_for_Microrobot_control/binary_images/5_Bright_(69,0,930,426)_processed.png'
env_config_self_segmented["name"] = (0, 0)


def main(env_configs):
    warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

    # See configs.yaml for all options.
    config = embodied.Config(dreamerv3.Agent.configs['defaults'])
    config = config.update(dreamerv3.Agent.configs['defaults']) #size12m
    config = config.update(dreamerv3.Agent.configs['debug'])
    config = config.update({
        'logdir': LOGDIR,
        'run.train_ratio': 100,
        'run.log_every': 200,  # Seconds
        'run.save_every': 200,  # Seconds
        'batch_size': 16,
        'jax.prealloc': True,
        # 'envs.checks' : True,
        # 'wrapper.checks': True,
        # 'jax.debug': True,
        # 'envs.restart': False,
        #'enc.spaces': 'image', #'agent_position|target_position',
        #'dec.spaces': 'image', #'agent_position|target_position',
        # 'encoder.cnn_keys': 'image',
        # 'decoder.cnn_keys': 'image',
        'jax.platform': 'gpu',
        'run.actor_batch': num_envs,
        #'run.log_keys_avg': '^log_.*',
        #'run.log_keys_max': '^log_.*',
        # Still have to run with 'Explore' for now
    })
    config["wrapper"].update({"length": 1000})
    config = embodied.Flags(config).parse()
    logdir = embodied.Path(config.logdir)
    os.makedirs(logdir, exist_ok=True)

    if os.path.exists(logdir / 'config.yaml'):
        print(f'\033[1;31mExperiment already exists at {logdir}.\033[0m')
    else:
        yaml.dump(config, open(logdir / 'config.yaml', 'w'))
        yaml.dump(env_config_racetrack, open(logdir / 'env_config.yaml', 'w'))
    
    step = embodied.Counter()
    # logger = embodied.Logger(step, [
    #     embodied.logger.TerminalOutput(),
    #     embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
    #     embodied.logger.TensorBoardOutput(logdir),
    #     # embodied.logger.WandBOutput(logdir.name, config),
    #     # embodied.logger.MLFlowOutput(logdir.name),
    # ])
    # envs = envs_gen(dreamer_config=config, configs=env_configs)
    # # env = embodied.Batch([*envs])
    # env = envs.__next__()
    # env.close()
    
    for config_env in env_configs:
        config_ = config_env["config"]
        name = config_env["image_string"]
        name = name.split("/")[-1].split(".")[0]
        with open(config_, 'r') as stream:
            try:
                config_ = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        config_path = f"{logdir}/general_config.yaml"
        with open(config_path, 'w') as stream:
            try:
                yaml.safe_dump(config_, stream)
            except yaml.YAMLError as exc:
                print(exc)
        with open(f"{logdir}/env_config_{name}.yaml", 'w') as stream:
            try:
                yaml.safe_dump(config_, stream)
            except yaml.YAMLError as exc:
                print(exc)
    
    def make_agent():
        env = make_env()
        env.close()
        return dreamerv3.Agent(env.obs_space, env.act_space, config)
    def make_replay():
        return embodied.replay.Replay(
            config.batch_length, config.replay.size, logdir / 'replay')
    def make_logger():
        logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.TensorBoardOutput(logdir),
    ])
        return logger
    def make_env(i=0):
        envs = envs_gen(dreamer_config=config, configs=env_configs)
        env = envs.__next__()
        return env
    
    # agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
    replay = embodied.replay.Replay(
        config.batch_length, config.replay.size, logdir / 'replay')
    args = embodied.Config(
        **config.run, logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length)
    
    # envs_gen_cls_ = envs_gen_cls(dreamer_config=config, configs=env_configs)
    
    # embodied.run.parallel(agent, replay, logger, envs_gen_cls_, num_envs, args)
    args = embodied.Config(
    **config.run,
    logdir=config.logdir,
    batch_size=config.batch_size,
    batch_length=config.batch_length,
    batch_length_eval=config.batch_length_eval,
    replay_context=config.replay_context,
  )
    embodied.run.train(
        make_agent, make_replay, make_env, make_logger, args)
    
    # checkpoint = embodied.Checkpoint(f"{args.logdir}/checkpoint.ckpt")
    # args = args.update({"from_checkpoint": f"{args.logdir}/checkpoint.ckpt"})
    # embodied.run.eval_only(agent, env, logger, args)

def envs_gen(dreamer_config, configs):
    for config in configs:
        env = env_cls_dict(**config)  # Replace this with your Gym env.
        env = MaxAndSkipEnv(env, 4)
        env = RateTargetReachedWrapper(env, 100, dreamer=True, verbose=1, logdir=f"{LOGDIR}/rate_target_reached.csv")
        env = RewardWrapper(env, logdir=f"{LOGDIR}/reward.csv")
        env = from_gym.FromGym(env)
        env = dreamerv3.wrap_env(env, dreamer_config)
        yield env

class envs_gen_cls():
    def __init__(self, dreamer_config, configs):
        self.envs = []
        self.count = -1
        self.dreamer_config = dreamer_config
        self.configs = configs
    
    def __call__(self, i, *args, **kwds):
        self.count += 1
        print(f"envs_gen_cls: {self.count}")
        
        config = self.configs[i]
        env = env_cls_dict(**config)
        env = MaxAndSkipEnv(env, 4)
        env = RateTargetReachedWrapper(env, 100, dreamer=True, verbose=1, logdir=f"{LOGDIR}/rate_target_reached_env_{i}.csv")
        env = RewardWrapper(env, logdir=f"{LOGDIR}/reward_{i}.csv")
        env = from_gym.FromGym(env)
        env = dreamerv3.wrap_env(env, self.dreamer_config)
        return env


if __name__ == '__main__':
    # config = [env_config_simple, env_config_4_squares, env_config_9_squares, env_config_racetrack, env_config_vascular_real, env_config_vascular_bis] 
    # config.extend([env_config_maze_1, env_config_maze_2, env_config_maze_3, env_config_maze_4])
    # config = [env_config_simple, env_config_simple, env_config_simple, env_config_simple]
    # ([env_config_racetrack, env_config_large_vascular, env_config_maze_1, env_config_maze_2, env_config_maze_3, env_config_maze_4]
    # config = [env_config_vascular_real] # , env_config_vascular_bis, env_config_vascular_real, env_config_vascular_real, env_config_vascular_real]
    # config = [env_config_vascular_real for _ in range(num_envs)]
    # config = [env_config_large_vascular for _ in range(num_envs)]
    # config = [env_config_large_vascular_by_wall for _ in range(num_envs)]
    #config = [env_config_racetrack for _ in range(num_envs)]
    config = [env_config_self_segmented]
    main(config)