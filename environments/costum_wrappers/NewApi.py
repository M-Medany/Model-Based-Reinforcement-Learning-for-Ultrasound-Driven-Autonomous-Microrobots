import gymnasium as gym
import numpy as np


class NewApi(gym.Wrapper[np.ndarray, int, np.ndarray, int]):

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        
    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        return obs, {}

    def render(self):
        return
