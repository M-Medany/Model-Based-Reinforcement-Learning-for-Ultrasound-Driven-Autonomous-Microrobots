from .ARSL_env_camera_dreamer import MicrorobotEnv as DiscreteMicrorobotEnv
import numpy as np
import yaml
from gymnasium import spaces
import time
from .game_env_8_actions import PIEZO_DIRECTIONS8


class MicrorobotEnvSweeping(DiscreteMicrorobotEnv):
    def __init__(self, fake=False, *args, **kwargs):
        if fake:
            with open(kwargs["config"], 'r') as yaml_file:
                self.config = yaml.safe_load(yaml_file)
            self.action_space = spaces.Discrete(self.config['Action_space_settings']['TOTAL_ACTIONS'])
            self.observation_space = spaces.Dict({
                'image': spaces.Box(low=0, high=255, shape=(self.config['Layout_settings']['IMG_SIZE'], self.config['Layout_settings']['IMG_SIZE'], 3), dtype=np.uint8,),
                "agent_position": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
                "target_position": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
                })
            self.observation_space["piezo"] = spaces.Box(low=0, high=1, shape=(self.config['Action_space_settings']['NUMBER_PIEZOS'],), dtype=np.float32)
            self.observation_space['log_distance_reward'] = spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)
            self.observation_space['log_substep_reward'] = spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)
            self.observation_space['log_reached_reward'] = spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)
            self.observation_space['log_collision_reward'] = spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)
            return
        super().__init__(*args, **kwargs)
        self.function_generator.set_sweep_mode()
        self.function_generator.set_sweep_limits(self.config['Action_space_settings']['SWEEP_MIN'], self.config['Action_space_settings']['SWEEP_MAX'], "MHz")
    
    def step(self, action):
        area = self.tracker.get_bubble_area()
        vpp = self._vpp_from_area(area)
        self.function_generator.set_vpp(vpp)
        # freq = np.random.uniform(2.3, 2.5)
        # freq = self.function_generator.set_frequency(freq)
        piezo = self.arduino.set_piezo_from_action(action)
        freq = self.function_generator.get_frequency()
        print(f"Going direction: {PIEZO_DIRECTIONS8.convert(piezo)}")
        # if np.random.random() > 0.99:
        #     freq = self.function_generator.set_frequency_from_action(np.random.randint(0,16))
        #     print("random freq: ", freq)

        # time.sleep(self.config['Action_space_settings']['STEP_DURATION'])
        return self._post_step(piezo, vpp, freq)