import numpy as np
from .game_env_dreamer_rand_freq import MicrorobotEnvContGameFreqResampled
from .microrobot_env import PIEZO_DIRECTIONS


class MicrorobotEnvGame8Act(MicrorobotEnvContGameFreqResampled):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._action_to_direction.update({
            PIEZO_DIRECTIONS8.OFF: np.array([0, 0]),
            PIEZO_DIRECTIONS8.UP: np.array([0, -1]),
            PIEZO_DIRECTIONS8.RIGHT: np.array([1, 0]),
            PIEZO_DIRECTIONS8.LEFT: np.array([-1, 0]),
            PIEZO_DIRECTIONS8.DOWN: np.array([0, 1]),
            PIEZO_DIRECTIONS8.DOWN_RIGHT: np.array([np.sqrt(2)/2, np.sqrt(2)/2]),
            PIEZO_DIRECTIONS8.DOWN_LEFT: np.array([-np.sqrt(2)/2, np.sqrt(2)/2]),
            PIEZO_DIRECTIONS8.UP_RIGHT: np.array([np.sqrt(2)/2, -np.sqrt(2)/2]),
            PIEZO_DIRECTIONS8.UP_LEFT: np.array([-np.sqrt(2)/2, -np.sqrt(2)/2]),
        })
        
        self._direction_to_action = {
            tuple(v): k for k, v in self._action_to_direction.items()
        }

    def step(self, action):
        act_num = action % self.config['Action_space_settings']['NUMBER_PIEZOS']
        direction = self._action_to_direction[int(act_num)+1]
        amplitude = self._amplitude_from_action(action, act_num)
        return super()._post_step(direction, amplitude)


class PIEZO_DIRECTIONS8(PIEZO_DIRECTIONS):
    DOWN_RIGHT = 5
    DOWN_LEFT = 6
    UP_RIGHT = 7
    UP_LEFT = 8

    def convert(direction):
        if direction <= 4:    
            return PIEZO_DIRECTIONS.convert(direction)
        if direction == PIEZO_DIRECTIONS8.DOWN_RIGHT:
            return "DOWN_RIGHT"
        if direction == PIEZO_DIRECTIONS8.DOWN_LEFT:
            return "DOWN_LEFT"
        if direction == PIEZO_DIRECTIONS8.UP_RIGHT:
            return "UP_RIGHT"
        if direction == PIEZO_DIRECTIONS8.UP_LEFT:
            return "UP_LEFT"
        return f"ERROR, direction {direction} not found in PIEZO_DIRECTIONS8"
            