import gymnasium as gym
import numpy as np
import cv2


class NormalizedObs(gym.ObservationWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Box(
            -1.0,
            1.0,
            #shape=self.observation_space.shape,
            shape=(64, 64, 3), #maybe this was leading to issues
            dtype=np.float32,
        )
    
    # def observation(self, observation):
    #     #observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY) # Make it a grayscale image
    #     image_2 = observation.astype(np.float32) / 255.0
    #     return image_2
    

    # Divide by scale and center around 0.0, such that observations are in the range
    # of -1.0 and 1.0.
    def observation(self, observation):
        #observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY) # Make it a grayscale image
        return (observation.astype(np.float32) / 128.0) - 1.0

class GrayScaleObs(gym.ObservationWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Box(
            0, # make it 0.0 instead of -1.0 to keep 0.0-1.0 range
            255,
            shape=(self.observation_space.shape[0], self.observation_space.shape[1], 1),
            dtype=np.uint8,
        )
    
    def observation(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY) # Make it a grayscale image
        return observation.astype(np.uint8)