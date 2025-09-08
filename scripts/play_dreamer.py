import warnings
import dreamerv3
from dreamerv3 import embodied
from dreamerv3.embodied.envs import from_gym, from_dm, atari
from environments import MicrorobotEnvGameRayWrappedCont, MicrorobotEnvContGame, MicrorobotEnvGameNoCollision, MicrorobotEnvGameByTheWall, MicrorobotEnvGame8Act
import yaml, glob
import os
from environments.costum_wrappers.MaxandSkip import MaxAndSkipEnv
from environments.costum_wrappers.RateTargetReached import RateTargetReachedWrapper, RewardWrapper


LOGDIR = '/home/mahmoud/play_logdir/vascular_by_wall_Large_faster_flow_far_3'
ckp = "/home/mahmoud/logdir/vascular_by_wall_Large_faster_flow"
num_envs = 1
yaml_config = "config_sim_by_wall"

env_config_racetrack={
    "name": (0,0),
    "config": f"/home/mahmoud/logdir/vascular_by_wall_Large/env_config_default_segmentation_vascular_large.yaml",
    "max_envs": 1,
    "render_mode": "human",
    "image_string": 'binary_images/default_mask_racetrack_segmented.png', # binary_images/closing1.png
    "subepisode_sampling": True,
    "subepisode_length": 2,  
    }

env_config_4_squares=env_config_racetrack.copy()
env_config_4_squares["image_string"] = 'binary_images/4_squares.png'
env_config_4_squares["name"] = (0,0)

env_config_simple=env_config_racetrack.copy()
env_config_simple["image_string"] = 'binary_images/simple.png'
env_config_simple["name"] = (0,0)

env_config_vascular=env_config_racetrack.copy()
env_config_vascular["image_string"] = 'binary_images/default_segmentation_vascular.png'
env_config_vascular["name"] = (0,0)

env_config_vascular_real=env_config_racetrack.copy()
env_config_vascular_real["image_string"] = 'binary_images/default_segmentation_vascular_4_closed.png'
env_config_vascular_real["name"] = (0,0)

env_config_vascular_bis=env_config_racetrack.copy()
env_config_vascular_bis["image_string"] = 'binary_images/default_segmentation_vascular_fake.png'
env_config_vascular_bis["name"] =(0,0)

env_config_9_squares=env_config_racetrack.copy()
env_config_9_squares["image_string"] = 'binary_images/closing_cleaned.png'
env_config_9_squares["name"] = (0,0)

env_config_maze_1=env_config_racetrack.copy()
env_config_maze_1["image_string"] = 'binary_images/maze_1.png'
env_config_maze_1["name"] = (0,0)

env_config_maze_2=env_config_racetrack.copy()
env_config_maze_2["image_string"] = 'binary_images/maze_2.png'
env_config_maze_2["name"] = (0,0)

env_config_maze_3=env_config_racetrack.copy()
env_config_maze_3["image_string"] = 'binary_images/maze_3.png'
env_config_maze_3["name"] = (0,0)

env_config_maze_4=env_config_racetrack.copy()
env_config_maze_4["image_string"] = 'binary_images/maze_4.png'
env_config_maze_4["name"] = (0,0)

env_config_large_vascular=env_config_racetrack.copy()
env_config_large_vascular["image_string"] = "binary_images/default_segmentation_vascular_large.png"

env_config_large_vascular_by_wall=env_config_racetrack.copy()
env_config_large_vascular_by_wall["image_string"] = "binary_images/default_segmentation_vascular_large.png"
env_config_large_vascular_by_wall["inv_img_path"] = "binary_images/vascular_dilated.png"


def main(env_config):
    warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

    # See configs.yaml for all options.
    config = embodied.Config(dreamerv3.configs['defaults'])
    config = config.update(dreamerv3.configs['large'])
    # config = config.update(dreamerv3.configs['debug'])
    config = config.update({
        'logdir': LOGDIR,
        'run.train_ratio': 1,
        'run.log_every': 360,  # Seconds
        'batch_size': 16,
        'jax.prealloc': True,
        'encoder.mlp_keys': '$^', #'agent_position|target_position',
        'decoder.mlp_keys': '$^', #'agent_position|target_position',
        'encoder.cnn_keys': 'image',
        'decoder.cnn_keys': 'image',
        'jax.platform': 'gpu',
        'run.actor_batch': num_envs,
        'run.log_keys_mean': '^log_.*',
        'run.log_keys_max': '^log_.*',
         # Still have to run with 'Explore' for now
    })
    config["wrapper"].update({"length": 50})
    config = embodied.Flags(config).parse()
    logdir = embodied.Path(config.logdir)
    os.makedirs(logdir)
    csv_files = glob.glob(f"{ckp}/*.csv")
    txt_files = glob.glob(f"{ckp}/*.txt")
    yaml_files = glob.glob(f"{ckp}/*.yaml")

    paths = csv_files + txt_files + yaml_files
    for path in paths:
        os.system(f"cp {path} {logdir}")
    
    step = embodied.Counter()
    envs = envs_gen(dreamer_config=config, configs=[env_config])
    env = embodied.BatchEnv([*envs], parallel=False)

    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.TensorBoardOutput(logdir),
        # embodied.logger.WandBOutput(logdir.name, config),
        # embodied.logger.MLFlowOutput(logdir.name),
    ])
    
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    replay = embodied.replay.Uniform(
        config.batch_length, config.replay_size, logdir / 'replay')
    args = embodied.Config(
        **config.run, logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length)
    
    # embodied.run.train(agent, env, replay, logger, args)
    checkpoint = embodied.Checkpoint(f"{ckp}/checkpoint.ckpt")
    args = args.update({"from_checkpoint": f"{ckp}/checkpoint.ckpt"})
    embodied.run.eval_only(agent, env, logger, args)

def envs_gen(dreamer_config, configs):
    for config in configs:
        env = MicrorobotEnvGameByTheWall(**config)  # Replace this with your Gym env.
        env = MaxAndSkipEnv(env, 4)
        env = RateTargetReachedWrapper(env, 100, dreamer=True, verbose=1, logdir=f"{LOGDIR}/tg_reached_eval.csv")
        env = RewardWrapper(env, logdir=f"{LOGDIR}/reward.csv")
        env = from_gym.FromGym(env)
        env = dreamerv3.wrap_env(env, dreamer_config)
        yield env


if __name__ == '__main__':
    main(env_config_large_vascular)