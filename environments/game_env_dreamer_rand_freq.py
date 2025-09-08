import numpy as np
from scipy.special import softmax
from collections import defaultdict
from .game_env_dreamer_cont import MicrorobotEnvContGame
from gym import spaces


class MicrorobotEnvContGameFreq(MicrorobotEnvContGame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_actions = self.config['Action_space_settings']['TOTAL_ACTIONS']
        self.tot_steps = 0
        self._frequencies = -np.ones(self._num_actions)
        self.max_ampl = self.config['Action_space_settings']['MAX_AMPLITUDE']
        self.min_ampl = self.config['Action_space_settings']['MIN_AMPLITUDE']

    def step(self, action):
        act_num = action % self.config['Action_space_settings']['NUMBER_PIEZOS']
        direction = self._action_to_direction[int(act_num)+1]
        amplitude = self._amplitude_from_action(action)
        return super()._post_step(direction, amplitude)
    
    def _amplitude_from_action(self, action):
        self.tot_steps += 1
        self._frequencies[action] -= 1
        
        freq_prob = softmax(self._frequencies/(self.tot_steps))
        amplitude = self._num_actions * freq_prob[action] * (self.max_ampl - self.min_ampl) + self.min_ampl
        # if self.tot_steps % 100 == 0 and self.verbose > 0:
        #     print(f"Amplitude: {amplitude}, freq: {freq_prob[action]}")
        return np.clip(amplitude, self.min_ampl, self.max_ampl)


class MicrorobotEnvContGameFreqResampled(MicrorobotEnvContGame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_actions = self.config['Action_space_settings']['TOTAL_ACTIONS']
        self._num_piezos = self.config['Action_space_settings']['NUMBER_PIEZOS']
        self.min_ampl = self.config['Action_space_settings']['MIN_AMPLITUDE']
        self.max_ampl = self.config['Action_space_settings']['MAX_AMPLITUDE']
        self._action_per_piezo = self._num_actions // self._num_piezos
        self._act_freq = {k: [-1 for _ in range(self._action_per_piezo)]
                              for k in range(self._num_piezos)}
        self._tot_steps_per_piezo = {k: 0 for k in range(self._num_piezos)}
        self.prev_ampl = np.random.uniform(self.min_ampl, self.max_ampl)

    def step(self, action):
        act_num = action % self.config['Action_space_settings']['NUMBER_PIEZOS']
        direction = self._action_to_direction[int(act_num)+1]
        amplitude = self._amplitude_from_action(action, act_num)
        return super()._post_step(direction, amplitude)
    
    def _amplitude_from_action(self, action, act_num):
        # self._tot_steps_per_piezo[act_num] += 1
        # freq = (action // self._num_piezos) % self._action_per_piezo
        # self._act_freq[act_num][freq] -= 1

        # freq_prob = softmax(np.array(self._act_freq[act_num])/((self._tot_steps_per_piezo[act_num])/5))
        # amplitude = self._action_per_piezo * freq_prob[freq] * (self.max_ampl - self.min_ampl) + self.min_ampl
        
        amplitude = self.prev_ampl * np.random.uniform(0.5, 1.5)
        amplitude = np.clip(amplitude, self.min_ampl, self.max_ampl)
        self.prev_ampl = amplitude
        
        if np.random.rand() < 0.01:
            self.prev_ampl = np.random.uniform(self.min_ampl, self.max_ampl)

        # if self._tot_steps_per_piezo[act_num] % 100 == 0:
        #     print(f"Amplitude: {amplitude}, freq: {freq_prob[freq]}, act_num: {act_num}")

        return np.clip(amplitude, self.min_ampl, self.max_ampl)
    
class MicrorobotEnvContinousGame(MicrorobotEnvContGame):


    def __init__(self, min_freq, max_freq, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Box(low=min_freq, high=max_freq, shape=(1,), dtype=np.float32)
