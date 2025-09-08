import numpy as np
from .game_env_8_actions import MicrorobotEnvGame8Act


class MicrorobotEnvGameNoCollision(MicrorobotEnvGame8Act):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reduce_radius = self.config['General_environment_settings']['REDUCE_RADIUS']
        print(f"reduce_radius: {self.reduce_radius}")
        
    def step(self, action):
        act_num = action % self.config['Action_space_settings']['NUMBER_PIEZOS']
        direction = self._action_to_direction[int(act_num)+1]
        amplitude = self._amplitude_from_action(action, act_num)
        return self._post_step(direction, amplitude)

    def _post_step(self, direction, amplitude):
        reward = 0
        if self.is_valid(direction, amplitude, radius=self.get_radius()+self.tolerance_collision):
            self.agent_location = self.move_agent(direction, amplitude)
        elif self.check_collision(*self.agent_location, self.get_radius()):
            self._update_radius(self.reduce_radius)
            if self.is_valid(direction, amplitude/4, radius=self.get_radius()/2):
                self.agent_location = self.move_agent(direction, amplitude/4)
            else:
                reward += self.reward_collision
        else:
            self.agent_location = self.move_agent(direction, self.get_radius()/1.25)
            self._update_radius(self.reduce_radius)

        self.terminated = False
        self.truncated = False
        self._elapsed_steps += 1
        tol_target = self.get_radius() + self.tolerance_target_reached

        if np.allclose(self.agent_location, self.target_location, atol=tol_target):
            reward = self.reward_target_reached
            self.terminated = True
            self.target_reached = True
            self.count_target += 1
        
        elif self.get_radius() == self.min_bubble_radius:
            self.terminated = True
            reward = self.reward_termination
            self._reset_radius()

        else:
            distance = self._get_norm_dist(self.agent_location, self.target_location)
            reward += self.reward_function(distance)
            if self.verbose == 2:
                print("distance: ", distance)
                print("reward: ", reward)

        done = self.terminated
        return self._get_obs(), reward, done, self._get_info()
