from gymnasium import spaces
import numpy as np
import os
from utils.actuator import Arduino, FunctionGenerator_1
from utils.tracking_CSRT import CSRT_tracker
from utils.image_postprocessing import create_binary_bitmap, plot_cluster_on_image_blue, resize_and_crop_frame, find_legal_point_target_close, detect_largest_cluster
from utils import ImageSegmentation
from microrobot_env import BaseMicrorobotEnv, PIEZO_DIRECTIONS
from ARSL_env_camera_dreamer import MicrorobotEnv
import cv2
import time
import csv
import json


class MicrorobotEnvCont(MicrorobotEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100, "reward_types": ["sparse", "dense"]}  

    def __init__(self, timeout=50, save_path_experiment: str=None, threshold=100, 
                 parameters: str='rgb_on_rest_default', run: int=0, subepisode_sampling: bool=False,
                 subepisode_length=10, reward_function: str='linear', reward_target_reached: int=1, 
                 reward_collision: int=-1, reward_step=-0.01, const=-0.01):
        
        super().__init__(timeout=timeout, save_path_experiment=save_path_experiment, threshold=threshold, 
                         parameters=parameters, run=run, subepisode_sampling=subepisode_sampling,
                         subepisode_length=subepisode_length, reward_function=reward_function, 
                         reward_target_reached=reward_target_reached, reward_collision=reward_collision, 
                         reward_step=reward_step, const=const)
        self.action_space = spaces.Dict({"Continous": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32,),
                                         "Discrete": spaces.Discrete(4)})

    def step(self, action):
        discrete_action = action['Discrete']
        continuous_action = action['Continuous']

        vpp = self.function_generator.set_vpp_from_action(discrete_action)
        freq = self.function_generator.set_frequency(continuous_action)
        piezo = self.arduino.set_piezo_from_action(discrete_action)

        time.sleep(self.config['Action_space_settings']['STEP_DURATION']) 
        return super()._post_step(piezo, vpp, freq)
