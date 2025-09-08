import pygame
import cv2
import numpy as np
from .game_env_no_collision import MicrorobotEnvGameNoCollision
from utils.path_planning_v2 import RRTStar
from PIL import Image


class MicrorobotEnvGameByTheWall(MicrorobotEnvGameNoCollision):
    def __init__(self, inv_img_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if inv_img_path is None:
            img = cv2.imread(kwargs["image_string"], cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            inv_dil_obs = RRTStar.inverse_dilation((np.asarray(img, dtype=np.uint8)/255).astype(np.uint8), 
                                                            self.config["General_environment_settings"]["DILATION"], 
                                                            True)
            img = Image.fromarray((inv_dil_obs*255).astype(np.uint8)).convert('RGB')
            img.save(f"{kwargs['image_string'][:-4]}_inv_dil.png")
            self.inv_dil_obs = pygame.image.load(f"{kwargs['image_string'][:-4]}_inv_dil.png")

        else:
            self.inv_dil_obs = pygame.image.load(inv_img_path)
        
        self.inv_dil_obs = pygame.transform.scale(self.inv_dil_obs, (self.img_size, self.img_size))
        self.inv_dil_obs = pygame.surfarray.pixels3d(self.inv_dil_obs)
        self.reward_center = self.config['Reward Settings']['reward_center']
        self.flow_direction = np.array(self.config['General_environment_settings']['FLOW_DIRECTION'])
        self._flow = any(self.flow_direction) or self.reward_center != 0

    def _post_step(self, direction, amplitude):
        reward = 0
        if self.is_valid(direction, amplitude, radius=self.get_radius()+self.tolerance_collision):
            self.agent_location = self.move_agent(direction, amplitude)
        elif self.check_collision(*self.agent_location, self.get_radius()):
            if self.is_valid(direction, amplitude/2, radius=self.get_radius()/4):
                # print(f"direc: {PIEZO_DIRECTIONS8.convert(self._direction_to_action[tuple(direction)])}, ampl: {amplitude}")
                self.agent_location = self.move_agent(direction, amplitude/2)
            else:
                reward += self.reward_collision
                self._update_radius(self.reduce_radius/2)
        else:
            self.agent_location = self.move_agent(direction, self.get_radius()/1)
        
        if self._flow:
            if self._eval_pixel_collision(self.agent_location[0].astype(int),
                                        self.agent_location[1].astype(int), self.inv_dil_obs):
                reward += self.reward_center
                if self.is_valid(self.flow_direction, amplitude/1.5, radius=self.get_radius()/1.5):
                    self.agent_location = self.move_agent(self.flow_direction, amplitude)
                else:
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