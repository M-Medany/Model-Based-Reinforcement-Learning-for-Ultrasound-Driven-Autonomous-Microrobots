import numpy as np
from typing import Dict, List, Optional, Union
from gym import Env, spaces
from utils import ArduinoActuator, FunctionGenerator, CameraHammamatsu
from utils.old.path_planning import RRTStar


class MicroSimulatedEnv(Env):
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
        """
        Initialize an instance of MicroSwimmerEnv.

        :param seed: random seed
        :param actuator_layout: positions (in meters) of the speakers/actuators:
        {'act_1': [x_0, y_0, z_0], 'act_2': [x_1, y_1, z_1]}, if None, the positions will be 4 corners of the tank
        :param time_delta: time interval between two consecutive time steps, in seconds
        :observe_images: if True, the environment will return images instead of observations (TODO: Is the image not part of the observation)
        :param perturbation_scale: perturbation noise scale, relative to the observation scale. If the observation is
        normalized, than zero-mean Gaussian variables with standard deviation of @perturbation_scale are added to
        each of the perturbed observations; for non-normalized observations, this noise is appropriately rescaled
        """
        # random seeding
        self._np_random, self._seed = self.seed(seed)
        self.config = config
        
        # initialize setup
        self.arduino = None # TODO: Implement arduino class from pygame
        self.camera = None # TODO: Implement camera class from pygame
        self.initial_obs = self._get_obs()
        self.path_planner = RRTStar(self.initial_obs["image"], self.initial_obs["robot_location"], self.initial_obs["robot_location"]) # TODO: Check arguments
        self.path = self.path_planner.plan()

        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=1.0, high=2.0, shape=(config["IMG_SIZE"], config["IMG_SIZE"]), dtype=np.float32),
            'target_location': spaces.Box(low=[config["X_MIN"], config["Y_MIN"]], high=[config["X_MAX"], config["Y_MAX"]], shape=(2,), dtype=np.float32),
            'robot_location': spaces.Box(low=[config["X_MIN"], config["Y_MIN"]], high=[config["X_MAX"], config["Y_MAX"]], shape=(2,), dtype=np.float32)
        })

        self.action_space = spaces.Discrete(4)
        self.actuator = ActuatorClass()

        self.culminative_reward = 0
        self.estimate_new_target = True

    def _get_obs(self):
        """
        Get the observation of the current state, observation is a dict
        """
        image, segm_image, bubble_coordinates = None # TODO: Implement camera class from pygame
        
        if self.estimate_new_target:
            current_target_point = self._estimate_new_target(segm_image, bubble_coordinates, self.config["min_target_distance"])
            self.estimate_new_target = False

        return {'image': image, 'robot_location': current_target_point, 'robot_location': bubble_coordinates}

    #This does not work with the current action space I think. TODO: Make them compatible
    def _apply_action(self, action):
        """
        Apply the action to the actuators
        """

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


    # TODO: Adapt
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
