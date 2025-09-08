import numpy as np
# from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
import csv
import os


# class RateTargetReachedCallback(BaseCallback):
#     """
#     Custom callback for plotting additional values in tensorboard.
#     """

#     def __init__(self, verbose=0, log_frequency=100):
#         super().__init__(verbose)
#         self._log_frequency = log_frequency

#     def _on_step(self) -> bool:
#         # print(self.training_env.get_attr("rate_target_reached")[0])

#         if self.model.num_timesteps % self._log_frequency == 0:
#             value = self.training_env.get_attr("rate_target_reached")[0]
#             self.logger.record("eval/rate_target_reached", value)
#         return True


class RateTargetReachedWrapper(gym.Wrapper):

    def __init__(self, env, frequency=100, verbose=0, dreamer=False, logdir=None):
        super().__init__(env)
        self._log_frequency = frequency
        self.rate_target_reached = np.array([0,]).astype(np.float32)
        self.verbose = verbose
        self.episode_count = 0
        self.count_target = 0
        self.dreamer = dreamer
        self._steps_rate_target_reacher = 0
        self.logdir = logdir
        if self.logdir:
            assert logdir.endswith(".csv")
            if not os.path.exists(self.logdir):
                print(f"Creating log file at {self.logdir}")
                with open(self.logdir, 'w', newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["steps", "rate_target_reached"])
            else:
                try:
                    self._steps_rate_target_reacher = int(open(self.logdir, 'r').readlines()[-1].split(",")[0])
                    print(f"Log file already exists at {self.logdir}, resuming from step {self._steps_rate_target_reacher}")
                except:
                    print(f"Log file already exists at {self.logdir}, but it is empty, overwriting it.")
                    with open(self.logdir, 'w', newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(["steps", "rate_target_reached"])
                    
        if dreamer:
            self.observation_space['log_rate_target_reached'] = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._steps_rate_target_reacher += 1
        if info['target_reached']:
            self.count_target += 1
        if self.dreamer:
            obs["log_rate_target_reached"] = self.rate_target_reached
        return obs, reward, done, info

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        self.episode_count += 1
        self._steps_rate_target_reacher += 1
        if self.episode_count >= self._log_frequency:
            self.rate_target_reached = np.array((self.count_target/self.episode_count,)).astype(np.float32)
            self.episode_count = 0
            self.count_target = 0
            if self.verbose > 0:
                print("\033[92m" + f"Rate target reached {self.rate_target_reached} times" + "\033[0m" + f" at step {self._steps_rate_target_reacher}, path {self.logdir}" )
            if self.logdir:
                with open(self.logdir, 'a', newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([self._steps_rate_target_reacher, self.rate_target_reached[0]])
        if self.dreamer:
            obs["log_rate_target_reached"] = self.rate_target_reached
        return obs

class RewardWrapper(gym.Wrapper):
    
    def __init__(self, env, logdir=None):
        super().__init__(env)
        self.logdir = logdir
        self._steps_reward = 0
        self._reward_total = 0
        self._lenght = 0
        if self.logdir:
            assert logdir.endswith(".csv")
            if not os.path.exists(self.logdir):
                print(f"Creating log file at {self.logdir}")
                with open(self.logdir, 'w', newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["steps", "reward", "lenght"])
            else:
                try:
                    self._steps_reward = int(open(self.logdir, 'r').readlines()[-1].split(",")[0])
                    print(f"Log file already exists at {self.logdir}, resuming from step {self._steps_reward}")
                except:
                    print(f"Log file already exists at {self.logdir}, but it is empty, overwriting it.")
                    with open(self.logdir, 'w', newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(["steps", "reward"])
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._steps_reward += 1
        self._lenght += 1
        self._reward_total += reward
        return obs, reward, done, info

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        if self.logdir:
            with open(self.logdir, 'a', newline="") as file:
                writer = csv.writer(file)
                writer.writerow([self._steps_reward, self._reward_total, self._lenght])
        self._reward_total = 0
        self._lenght = 0
        return obs
