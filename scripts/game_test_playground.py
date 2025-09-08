from environments.old.game_test_env import MicrorobotEnv
import numpy as np
import gymnasium as gym
from gymnasium.utils.play import play, PlayPlot


env = MicrorobotEnv(render_mode="rgb_array", microbubble_radius=5, image='binary_images/closing1.png')

def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
       return [rew, action, terminated]
plotter = PlayPlot(callback, 30, ["reward", "action", "terminated"])


keys_to_action = {
    (ord('d'), ): 0,
    (ord('s'), ): 1,
    (ord('a'), ): 2,
    (ord('w'), ): 3,
}

if __name__ == "__main__":
    play(env, keys_to_action=keys_to_action, fps=30, callback=plotter.callback)