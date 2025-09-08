from typing import Any, Dict, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time
# from stable_baselines3.common.type_aliases import AtariResetReturn, AtariStepReturn


class MaxAndSkipEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Return only every ``skip``-th frame (frameskipping)
    and return the max between the two last frames.

    :param env: Environment to wrap
    :param skip: Number of ``skip``-th frame
        The same action will be taken ``skip`` times.
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        assert env.observation_space['image'].dtype is not None, "No dtype specified for the observation space"
        assert env.observation_space['image'].shape is not None, "No shape defined for the observation space"
        self._obs_buffer = np.zeros((3, *env.observation_space['image'].shape), dtype=env.observation_space['image'].dtype)
        self._skip = skip

    def step(self, action: int):
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        # comment arduino lines in if using the real environment
        
        total_reward = 0.0
        done = False
        # arduino = self.env.unwrapped.arduino
        # arduino.set_piezo_by_number(0)
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            
            if i == self._skip - 3:
                self._obs_buffer[0] = obs['image']
            if i == self._skip - 2:
                self._obs_buffer[1] = obs['image']
            if i == self._skip - 1:
                self._obs_buffer[2] = obs['image']
            total_reward += float(reward)
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        # arduino.set_piezo_by_number(0)
        max_frame = self._obs_buffer.max(axis=0)
        
        out_obs = {'image': max_frame}
        out_obs.update(obs)
        return out_obs, total_reward, done, info
    


class MaxAndSkipEnv_BoxObs(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Return only every ``skip``-th frame (frameskipping)
    and return the max between the two last frames.

    :param env: Environment to wrap
    :param skip: Number of ``skip``-th frame
        The same action will be taken ``skip`` times.
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        assert isinstance(env.observation_space, spaces.Box), "Observation space must be of type Box (e.g. image)"
        self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action: int):
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        total_reward = 0.0
        #done = False
        terminated = False
        truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)
            if terminated or truncated:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)
        
        #out_obs = {'image': max_frame}

        return max_frame, total_reward, terminated, truncated, info #TODO: Check that this is correct
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._obs_buffer = np.zeros((2, *self.env.observation_space.shape), dtype=self.env.observation_space.dtype)
        return super().reset(seed=seed, options=options)