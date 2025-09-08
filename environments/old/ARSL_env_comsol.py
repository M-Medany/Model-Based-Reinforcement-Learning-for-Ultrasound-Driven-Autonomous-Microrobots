from typing import Tuple, List, Literal, Union, Optional, Dict, Path
from pathlib import Path
import mph

from gymnasium import Env
from gymnasium.utils import seeding
from gymnasium.spaces import Box, Discrete

import numpy as np
import os


# When normalizing the state vector to [0, 1], we need to know the boundaries for the atmospheric conditions.
# If no such boundaries are supplied, these defaults will be used.
DEFAULT_BOUNDARIES = {
    "bubble_size": (0.0, 20.0),
    "amplitude": (0.0, 360.0),
    "frequency": (-1, 1),
}


class MicroSwimmerEnv(Env):
    """
    MicroSwimmerEnv is an OpenAI gym environment that simulates a Microbubble swimmer as a reinforcement learning problem
    with dynamic pysics simulations. It either uses COMSOLE to simulate the environment at each time step, 
    or uses the real microscope images of the environment."""
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(
            self,
            seed: Optional[int] = None,
            comsole: Optional[Union[Path, str]] = None,
            actuator_layout: Optional[Dict[str, List[float]]] = None,
            observe_images: bool = False,
            time_delta: float = 1.0,
            normalize_observations: bool = True,
            perturbation_scale: float = 0.05
    ):
        """
        Initialize an instance of MicroSwimmerEnv.

        :param seed: random seed
        :param comsole: either a path to an input .mph file to initialize COMSOLE; if None,
        connection to the real microscope will be used
        :param actuator_layout: positions (in meters) of the speakers/actuators:
        {'act_1': [x_0, y_0, z_0], 'act_2': [x_1, y_1, z_1]}, if None, the positions will be 4 corners of the tank
        :param time_delta: time interval between two consecutive time steps, in seconds
        :observe_images: if True, the environment will return images instead of observations
        :param perturbation_scale: perturbation noise scale, relative to the observation scale. If the observation is
        normalized, than zero-mean Gaussian variables with standard deviation of @perturbation_scale are added to
        each of the perturbed observations; for non-normalized observations, this noise is appropriately rescaled
        """
        # random seeding
        self._np_random, self._seed = self.seed(seed)

        # initialize the floris interface depending on the argument type
        if comsole is None:
            # attempt connection to real microscope
            # TODO: add code to interface with microscope
            pass

        elif isinstance(comsole, Path):
            # attempt to start COMSOLE
            if not comsole.exists():
                raise FileNotFoundError(f"File {comsole} does not exist")
            self.client = mph.start(comsole)

        elif isinstance(comsole, str):
            # attempt to start COMSOLE
            if not os.path.exists(comsole):
                raise FileNotFoundError(f"File {comsole} does not exist")
            self.client = mph.start()
            self.model = self.client.load(comsole)

        self.actuators = []
        # place the actuators in the environment, if a layout is given
        if actuator_layout is not None:
            if isinstance(actuator_layout, dict):
                for actuator, coordinates in actuator_layout.items():
                    for i, coordinate in enumerate(coordinates):
                        self.model.parameter(f"{actuator}_{i}", coordinate)
                    self.actuators.append(Actuator(frequency=0, amplitude=0))
            self.n_actuators = len(actuator_layout)


        # initialize the action space to 2 normalized intervals [-1; 1] for each actuator (one is the 
        # frequency, the other is the amplitude)
        ones = np.array([1.0 for _ in range(self.n_actuators*2)], dtype=np.float32)
        self.action_space = Box(-ones, ones, dtype=np.float32)
        self.action_space.seed(self._seed)
        self.current_flow_points = None
        self._current_flow = None

        # initialize the observation space
        if observe_images:
            self.observation_space = Box(0, 255, shape=(3, 512, 512), dtype=np.uint8)
        else:
            self.observation_space = Box(-ones, ones, dtype=np.float32)

        self._perturbed_observations = None
        if perturbation_scale > 0.0:
            pass
            # TODO
            # self._perturbation_scale = [(x['max'] - x['min']) * perturbation_scale for x in self.observed_variables]
            # self._noise = MVGaussianNoiseProcess(len(self._perturbed_observations))
        self.observation_space.seed(self._seed)

        self.visualization = None
        self.state = self._get_state()

    @property
    def parameters(self) -> Dict[str, List[float]]:
        """
        All parameters in the setup
        """
        return self.model.parameters

    def _apply_action(self, action):
        """
        Apply the action to the actuators
        """
        for actuator, value, i in enumerate(zip(self.actuators, action)):
            # even indices are frequencies, odd indices are amplitudes
            if i % 2 == 0:
                actuator.frequency = value
            else:
                actuator.amplitude = value

    def _getmodel_measurement(self, measurement):
        return self.model.evaluate(measurement)


    def _get_state(self):
        self.model.solve()
        res = self._getmodel_measurement(['x', 'y'], "particle")
        # TODO: check name of the measurement

        return np.array(res)

    def _generate_noise(self):
        pass  # [self._get_measurement_point_data(d) for d in self._observed_variables]

    def seed(self, seed=None):
        return seeding.np_random(seed)

    def step(self, action):
        """
        :type action: np.array
        """
        action = np.array(action, dtype=np.float32)
        if not self.action_space.contains(action):
            action = np.minimum(np.maximum(action, -1.0), 1.0)

        done = False
        self._apply_action(action)
        self.update_model_actuators()
        
        self.state = self._get_state()

        reward = np.sum(self.model.evaluate("particle")) # TODO: decide reward function
        if np.isnan(reward):
            reward = 0
        reward *= self._reward_scaling_factor
        return self.state, reward, done, {}


    # Implementing a method from the base class. This method resets the environment to begin a new experiment
    def reset(self):

        self.model.reset()
        self.state = self._get_state()
        return self.state

    # Implementing a method from the base class. This method renders the environment for the user.
    def render(self, mode='human'):

        if self.state is None:
            return None

        if self.visualization is None:
            self.visualization = None # TODO: add visualization

        return self.visualization.render(return_rgb_array=mode == 'rgb_array')

    # Implementing a method from the base class. This method finalized the environment when it is not used anymore.
    def close(self):
        if self.visualization is not None:
            self.visualization.close()
        self.wind_process.close()
    
    def update_model_actuators(self):
        for actuator in self.actuators:
            self.model.parameter(f"{actuator.id}_frequency", actuator.frequency)
            self.model.parameter(f"{actuator.id}_amplitude", actuator.amplitude)


class Actuator(object):
    id = None
    position = [0, 0, 0]

    def __init__(self, frequency, amplitude) -> None:
        """Initialize the actuator with the given frequency and amplitude"""

        self.frequency = frequency
        self.amplitude = amplitude