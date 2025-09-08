from typing import Tuple, List, Literal, Union, Optional, Dict
from pathlib import Path
from gymnasium import Env
from gymnasium.utils import seeding
from gymnasium import spaces
import numpy as np
import os
import time
import serial
from settings import *
from utils.image_processing import *
import utils.tektronix_func_gen as tfg
import pymmcore
from utils.old.path_planning import *
import math
import yaml
from utils.actuator import ArduinoActuator, FunctionGenerator, CameraHammamatsu




#TODO: Check values on this
# When normalizing the state vector to [0, 1], we need to know the boundaries for the atmospheric conditions.
# If no such boundaries are supplied, these defaults will be used.
# DEFAULT_BOUNDARIES = {
#     "bubble_size": (0.0, 20.0),
#     "amplitude": (0.0, 360.0),
#     "frequency": (-1, 1),
# }



class MicroSwimmerCameraEnv(Env):
    """
    MicroSwimmerEnv is an OpenAI gym environment that simulates a Microbubble swimmer as a reinforcement learning problem
    with dynamic physics simulations. It either uses COMSOLE to simulate the environment at each time step, 
    or uses the real microscope images of the environment."""
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(
            self,
            config: Dict[str, Union[str, int, float]] = None,
            seed: Optional[int] = None,
            actuator_layout: Optional[Dict[str, List[float]]] = None,
            observe_images: bool = True,
            time_delta: float = 1.0,
            normalize_observations: bool = True,
            perturbation_scale: float = 0.0,

    ):
        # Load the YAML file
        with open(f'scripts/config.yaml', 'r') as yaml_file:
            self.config = yaml.safe_load(yaml_file)
       
        # random seeding
        self._np_random, self._seed = self.seed(seed)
        
        # initialize setup
        self.arduino = ArduinoActuator()
        self.func_gen = FunctionGenerator()
        self.camera = CameraHammamatsu()
        #self.mmc = pymmcore.CMMCore()
        self.initial_obs = self._get_obs()
        self.path_planner = RRTStar(self.initial_obs["image"], self.initial_obs["robot_location"], self.initial_obs["robot_location"]) # TODO: Check arguments
        self.path = self.path_planner.plan()

        self.observation_space = spaces.Box(low=0, high=255, shape=(self.config['Layout_settings']['IMG_SIZE'], self.config['Layout_settings']['IMG_SIZE'], 3), dtype=np.uint8)

        self.action_space = spaces.Discrete(config["NUMBER_PIEZOS"] * config["NUMBER_FREQUENCIES"] * config["NUMBER_AMPLITUDES"])
        self.actuator = ActuatorClass() #TODO: What is this supposed to do?

        self.culminative_reward = 0
        self.estimate_new_target = True

    def _get_obs(self):
        """
        Get the observation of the current state, observation is a dict
        """
        image, segm_image, bubble_coordinates = self.camera.get_image_and_bubble_coordinates() #TODO: Rewrite it such that we print only onto the image
        
        if self.estimate_new_target:
            current_target_point = self._estimate_new_target(segm_image, bubble_coordinates, self.config["min_target_distance"])
            self.estimate_new_target = False

        return {'image': image, 'robot_location': current_target_point, 'robot_location': bubble_coordinates}

    #This does not work with the current action space I think. TODO: Make them compatible
    def _apply_action(self, action):
        """
        Apply the action to the actuators
        """
        act_num = self.actuator._get_act_from_action_space(action)
        freq = self.actuator._get_freq_from_action_space(action)
        ampl = self.actuator._get_ampl_from_action_space(action)

        self.func_gen.set_frequency(action[0]) #This takes a float
        self.func_gen.set_vpp(action[1]) #This takes a float
        self.arduino.move(action[2]) #This takes an int, not a float TODO: Check

    def step(self, action):
        """
        :type action: np.array #TODO: Check data type
        """
       
        self._apply_action(action)
        observation = self._get_obs() #TODO: How quickly should we get one?


        terminated = observation['robot_location'] == observation['target_location'] 
        
        if observation['robot_location'][0] < X_MIN or observation['robot_location'][0] > X_MAX or observation['robot_location'][1] < Y_MIN or observation['robot_location'][1] > Y_MAX:
            truncated = True
        else:
            truncated = False


        reward = 1 if terminated else 0  # Binary sparse rewards TODO: implement euclidian distance reward
        self.culminative_reward += reward
        

        info = {
            'robot_location': observation['robot_location'],
            'target_location': observation['target_location'],
            'reward': reward,
            'culminative reward': self.culminative_reward
        } #TODO: What do we want to return here?

        if self.render_mode == "human":
            self.render()#TODO: Implement render function


        return observation, reward, terminated, truncated, info

    def reset(self):
        """
        Reset the environment to begin a new episode
        """
        super().reset(seed=self._seed) #TODO
        self.culminative_reward = 0
        self.func_gen.reset()
        self.estimate_new_target = True

        observation = self._get_obs()

        info = {
            'robot_location': observation['robot_location'],
            'target_location': observation['target_location'],
            'culminative reward': self.culminative_reward
        } #TODO: What do we want to return here?


        if self.render_mode == "human":
            self._render_frame() #TODO: Implement render function

        return observation, info


        # #TODO: Do we want to use this in the observation?
        # # place the actuators in the environment, if a layout is given
        # if actuator_layout is not None:
        #     if isinstance(actuator_layout, dict):
        #         for actuator, coordinates in actuator_layout.items():
        #             for i, coordinate in enumerate(coordinates):
        #                 self.model.parameter(f"{actuator}_{i}", coordinate) 
        #             self.actuators.append(Actuator(frequency=0, amplitude=0))
        #     self.n_actuators = len(actuator_layout)

        # #TODO: Adjust this
        # # initialize the action space to 2 normalized intervals [-1; 1] for each actuator (one is the 
        # # frequency, the other is the amplitude)
        # ones = np.array([1.0 for _ in range(self.n_actuators*2)], dtype=np.float32)
        # self.action_space = Box(-ones, ones, dtype=np.float32)
        # self.action_space.seed(self._seed)
        # self.current_flow_points = None
        # self._current_flow = None

    
        # self._perturbed_observations = None
        # if perturbation_scale > 0.0:
        #     pass
        #     # TODO
        #     # self._perturbation_scale = [(x['max'] - x['min']) * perturbation_scale for x in self.observed_variables]
        #     # self._noise = MVGaussianNoiseProcess(len(self._perturbed_observations))
        # self.observation_space.seed(self._seed)

        # self.visualization = None
        # self.state = self._get_state()

    @property
    def parameters(self) -> Dict[str, List[float]]:
        """
        All parameters in the setup
        """
        return self.model.parameters
    
    @staticmethod
    def _estimate_new_target(segm_image, bubble_coordinates, min_distance):
        """
        Estimate the new target point based on the segmentation image and the bubble coordinates
        """

        valid_points = np.argwhere(segm_image)
        distances = np.linalg.norm(valid_points - bubble_coordinates, axis=1)
        valid_points = valid_points[distances >= min_distance]
        if len(valid_points) == 0:
            raise ValueError("No valid points found")
        return valid_points[np.random.randint(len(valid_points))]
       

    # def _getmodel_measurement(self, measurement):
    #     return self.model.evaluate(measurement)

    # #TODO: Adapt
    # def _get_state(self):
    #     self.model.solve()
    #     res = self._getmodel_measurement(['x', 'y'], "particle")
    #     # TODO: check name of the measurement

    #     return np.array(res)

    def _generate_noise(self):
        pass  # [self._get_measurement_point_data(d) for d in self._observed_variables]

    def seed(self, seed=None):
        return seeding.np_random(seed)

    # def step(self, action):
    #     """
    #     :type action: np.array
    #     """
    #     #TODO:Data Type of action items 
    #     action = np.array(action, dtype=np.float32)
    #     if not self.action_space.contains(action):
    #         action = np.minimum(np.maximum(action, -1.0), 1.0)

    #     done = False
    #     self._apply_action(action)
    #     self.update_model_actuators()
        
    #     self.state = self._get_state()

    #     reward = np.sum(self.model.evaluate("particle")) # TODO: decide reward function
    #     if np.isnan(reward):
    #         reward = 0
    #     reward *= self._reward_scaling_factor
    #     return self.state, reward, done, {}


    # # Implementing a method from the base class. This method resets the environment to begin a new experiment
    # def reset(self):

    #     self.model.reset()
    #     self.state = self._get_state()
    #     return self.state

    # # Implementing a method from the base class. This method renders the environment for the user.
    # def render(self, mode='human'):

    #     if self.state is None:
    #         return None

    #     if self.visualization is None:
    #         self.visualization = None # TODO: add visualization

    #     return self.visualization.render(return_rgb_array=mode == 'rgb_array')

    # Implementing a method from the base class. This method finalized the environment when it is not used anymore.
    def close(self):
        if self.visualization is not None:
            self.visualization.close()
        self.wind_process.close()
    
    #TODO: Is this needed?
    def update_model_actuators(self):
        for actuator in self.actuators:
            self.model.parameter(f"{actuator.id}_frequency", actuator.frequency)
            self.model.parameter(f"{actuator.id}_amplitude", actuator.amplitude)


class ActuatorClass:
    def __init__(self, config, **kwargs):
        self.config = config
        self.NUMBER_PIEZOS = kwargs.get('NUMBER_PIEZOS', self.config['DEFAULT_NUMBER_PIEZOS'])
        self.NUMBER_FREQUENCIES = kwargs.get('NUMBER_FREQUENCIES', self.config['DEFAULT_NUMBER_FREQUENCIES'])
        self.NUMBER_AMPLITUDES = kwargs.get('NUMBER_AMPLITUDES', self.config['DEFAULT_NUMBER_AMPLITUDES'])
        self.MIN_FREQUENCY = kwargs.get('MIN_FREQUENCY', self.config['DEFAULT_MIN_FREQUENCY'])
        self.MAX_FREQUENCY = kwargs.get('DEFAULT_MAX_FREQUENCY', self.config['DEFAULT_MAX_FREQUENCY'])
        self.MIN_AMPLITUDE = kwargs.get('MIN_AMPLITUDE', self.config['DEFAULT_MIN_AMPLITUDE'])
        self.MAX_AMPLITUDE = kwargs.get('MAX_AMPLITUDE', self.config['DEFAULT_MAX_AMPLITUDE'])


    def _get_act_from_action_space(self, action):
        act_num = action % self.NUMBER_PIEZOS
        return act_num

    def _get_freq_from_action_space(self, action):
        action //= self.NUMBER_PIEZOS
        freq_index = action % self.NUMBER_FREQUENCIES
        # Calculate the actual frequency value based on the index
        freq_value = self.MIN_FREQUENCY + (freq_index * (self.MAX_FREQUENCY - self.MIN_FREQUENCY) / (self.NUMBER_FREQUENCIES - 1))
        return freq_value

    def _get_ampl_from_action_space(self, action):
        action //= (self.NUMBER_PIEZOS * self.NUMBER_FREQUENCIES)
        ampl_index = action % self.NUMBER_AMPLITUDES
        ampl_value = self.MIN_AMPLITUDE + (ampl_index * (self.MAX_AMPLITUDE - self.MIN_AMPLITUDE) / (self.NUMBER_AMPLITUDES - 1))            
        return ampl_value

