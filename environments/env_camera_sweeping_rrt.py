from .env_camera_continous import MicrorobotEnvContinous as RRTEnv
import numpy as np
import yaml
from gymnasium import spaces
from .game_env_8_actions import PIEZO_DIRECTIONS8

class RRTEnvSweeping(RRTEnv):
    def __init__(self, fake=False, *args, **kwargs):
        if fake:
            with open(kwargs["config"], 'r') as yaml_file:
                self.config = yaml.safe_load(yaml_file)
            self.action_space = spaces.Box(low=self.config['Action_space_settings']['MIN_AMPLITUDE'], 
                                           high=self.config['Action_space_settings']['MAX_AMPLITUDE'],
                                           shape=(self.config['Action_space_settings']['NUMBER_PIEZOS'],),
                                           dtype=np.float32)
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
        self.action_space = spaces.Box(low=self.config['Action_space_settings']['MIN_AMPLITUDE'], 
                                high=self.config['Action_space_settings']['MAX_AMPLITUDE'],
                                shape=(self.config['Action_space_settings']['NUMBER_PIEZOS'],),
                                dtype=np.float32)
        self.function_generator.set_sweep_mode()
        self.function_generator.set_sweep_limits(self.config['Action_space_settings']['SWEEP_MIN'], self.config['Action_space_settings']['SWEEP_MAX'], "MHz")
    
    def step(self, action):
        area = self.tracker.get_bubble_area()
        vpp = self._vpp_from_area(area)
        self.function_generator.set_vpp(vpp)
        freq = self.function_generator.get_frequency()

        out = self.path(self.get_agent_pos(), self.target_reached)
        self.target_reached = False
        self.target_location = out["next_waypoint"]
        
        print(f"Going direction: {PIEZO_DIRECTIONS8.convert(out['piezo'])}")
        
        # print("raw action: ", action)
        
        # amplitude = action[1]
        # amplitude = np.clip(amplitude, self.config['Action_space_settings']['MIN_AMPLITUDE'], self.config['Action_space_settings']['MAX_AMPLITUDE'])

        # amplitude = amplitude[out["piezo"]-1]  # Piezos are shift by 1 (they are 1-indexed)

        self.arduino.set_piezo_by_number(out["piezo"])

        # time.sleep(self.config['Action_space_settings']['STEP_DURATION'])
        # print(f"Action: {action}, Piezo: {piezo}, Frequency: {action}, Substep amount: {substep_amount}")
        return self._post_step(vpp=vpp, freq=freq, **out)
