from typing import Any
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import yaml
import cv2
import time
from scipy.special import softmax
from environments.costum_wrappers.MaxandSkip import MaxAndSkipEnv_BoxObs
# from ray.rllib.algorithms.dreamerv3.utils.env_runner import NormalizedImageEnv


class MicrorobotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30, "reward_types": ["sparse", "dense"]}  

    def __init__(self, render_mode: str="human", microbubble_radius: int=5, 
                 image_string=None, timeout: int=100, subepisode_sampling: bool=False, subepisode_length: int=0, 
                 reward_function: str='linear', reward_target_reached: int=1, reward_collision: int=-1, reward_step=-0.01,
                 const=-0.01): 

        # Load the YAML file
        with open(f'/home/m4/git/DQN_for_Microrobot_control/scripts/config.yaml', 'r') as yaml_file:
            self.config = yaml.safe_load(yaml_file)

        self.microbubble_radius = microbubble_radius  # The size of the microbubble
        self.window_size = self.config['Layout_settings']['IMG_SIZE'] # The size of the PyGame window
        #self.image = image
        self.image_string = image_string  # The image to use as the environment
        self.obstacles = pygame.image.load(f"{image_string}") # The image to use as the obstacle environment
        self.obstacles = pygame.transform.scale(self.obstacles, (self.window_size, self.window_size))
        self.num_envs = 1
        self.display_size = self.config['Layout_settings']['IMG_UPSCALED_SIZE']
        self.obstacles_disp = pygame.transform.scale(self.obstacles, (self.display_size, self.display_size))
        self.verbose = 1
        self.agent_location = self.find_legal_point()
        # self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(64, 64, 3), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(self.config['Layout_settings']['IMG_SIZE'], self.config['Layout_settings']['IMG_SIZE'], 3), dtype=np.uint8),
            'agent_position': spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            'target_position': spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
        })
            # 'vector': spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
        # self.observation_space = spaces.Dict({
        #     'image': spaces.Box(low=0, high=255, shape=(self.config['Layout_settings']['IMG_SIZE'], self.config['Layout_settings']['IMG_SIZE'], 3), dtype=np.uint8)})

        self.timeout = timeout
        self.count_target = 0

        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            -1: np.array([0, 0]),
            0: np.array([1, 0]),
            1: np.array([0, -1]),
            2: np.array([-1, 0]),
            3: np.array([0, 1]),
        }

        # self.action_space = spaces.Discrete(32)
        # self._action_to_direction = {
        #     -1: np.array([0, 0]),
        #     0: np.array([0.25, 0]),
        #     1: np.array([0, 0.25]),
        #     2: np.array([-0.25, 0]),
        #     3: np.array([0, -0.25]),
        #     4: np.array([0.5, 0]),
        #     5: np.array([0, 0.5]),
        #     6: np.array([-0.5, 0]),
        #     7: np.array([0, -0.5]),
        #     8: np.array([0.75, 0]),
        #     9: np.array([0, 0.75]),
        #     10: np.array([-0.75, 0]),
        #     11: np.array([0, -0.75]),
        #     12: np.array([1, 0]),
        #     13: np.array([0, 1]),
        #     14: np.array([-1, 0]),
        #     15: np.array([0, -1]),
        #     16: np.array([1.25, 0]),
        #     17: np.array([0, 1.25]),
        #     18: np.array([-1.25, 0]),
        #     19: np.array([0, -1.25]),
        #     20: np.array([1.5, 0]),
        #     21: np.array([0, 1.5]),
        #     22: np.array([-1.5, 0]),
        #     23: np.array([0, -1.5]),
        #     24: np.array([1.75, 0]),
        #     25: np.array([0, 1.75]),
        #     26: np.array([-1.75, 0]),
        #     27: np.array([0, -1.75]),
        #     28: np.array([2, 0]),
        #     29: np.array([0, 2]),
        #     30: np.array([-2, 0]),
        #     31: np.array([0, -2]),
        # }
       

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        self.tolerance_collision = self.config['General_environment_settings']['COLLISION_TOLERANCE']
        self.tolerance_target_reached = self.config['General_environment_settings']['TARGET_REACHED_TOLERANCE'] # This decides how close the agent needs to be to the target to be considered reached
        # self.distance_to_new_target = self.config['General_environment_settings']['DISTANCE_TO_NEW_TARGET']
        self.size_threshold = self.config['General_environment_settings']['SIZE_THRESHOLD'] # This decides how much the bubble can shrink before we consider it lost
        self.collision_reset_tolerance = self.config['General_environment_settings']['COLLISION_RESET_TOLERANCE']
        self.random_moves = self.config['General_environment_settings']['RANDOM_MOVES']
        self.random_move_probability = self.config['General_environment_settings']['RANDOM_MOVE_PROBABILITY']
        self.elapsed_steps = 0
        self.collision = False
        self.cumulative_reward = 0
        self.last_reward = 0
        self.state = 0
        self.n_elapsed_episodes = -1 # This is a hack to make the first episode start at 0 because reset() is called to initialize the environment
        self.terminated = False
        self.truncated = False
        self.first_obs = True
        self.collision_reset_steps = False
        self.subepisode_sampling = subepisode_sampling
        self.n_subepisodes = subepisode_length
        self.target_reached = False
        self.flength = self.window_size 
        self.fwidth = self.window_size
        #self.reward_target_reached = self.config['General_environment_settings']["Reward_Shape"]['REWARD_TARGET_REACHED']
        #self.reward_collision = self.config['General_environment_settings']["Reward_Shape"]['REWARD_COLLISION']
        #self.reward_step = self.config['General_environment_settings']["Reward_Shape"]['REWARD_STEP']
        #self.reward_function = self.config['General_environment_settings']["Reward_Shape"]['REWARD_FUNCTION']
        self.reward_function = reward_function
        self.reward_target_reached = reward_target_reached
        self.reward_step = reward_step
        self.reward_collision = reward_collision
        print("reward_function: ", self.reward_function)
        print("reward_target_reached: ", self.reward_target_reached)
        print("reward_collision: ", self.reward_collision)
        print("reward_step: ", self.reward_step)
        print("const: ", const)
        if self.reward_function == 'linear':
            self.reward_function = lambda x: -self.reward_step*x + const
        elif self.reward_function == 'quadratic':
            self.reward_function = lambda x: -self.reward_step*x**2 + const
        elif self.reward_function == 'log':
            self.reward_function = lambda x: -self.reward_step*np.log(x) + const
        elif self.reward_function == 'inverse':
            self.reward_function = lambda x: self.reward_step*(1/(x+1)) + const
        elif self.reward_function == 'inverse_squared':
            self.reward_function = lambda x: self.reward_step*(1/(x**2+1)) + const
        elif self.reward_function == 'binary':
            self.reward_function = lambda x: self.reward_step*0*x + const
        else:
            raise Exception("Reward function not implemented")

        self.n_safe_steps = self.config['Action_space_settings']['SAFE_STEPS']
        self.new_target_set = False
        self.target_location = self.find_legal_point_target_close(self.agent_location/64)
        

    def _get_norm_dist(self, agent_location, target_location):
        aget_loc_norm = agent_location / np.array([self.fwidth, self.flength])
        target_loc_norm = target_location / np.array([self.fwidth, self.flength])
        return np.linalg.norm(aget_loc_norm - target_loc_norm, ord=2)


    def step(self, action):
        #print("action: ", action)
        direction = self._action_to_direction[int(action)]
        if self.is_valid(direction):
            self.agent_location = self.move_agent(direction)
        else:
            self.collision = True
            self.terminated = True
            self.agent_location = self.agent_location + direction
            observation = self._get_obs() # get state s* after taking action a
            info = self._get_info()
            self.agent_location = self.agent_location - direction
            return observation, self.reward_collision, True, info

        # Set done for state s* to False
        self.terminated = False
        self.truncated = False
        reward = 0
        self._elapsed_steps += 1
        self.state += 1 # move to s*
  
        #Is the if if structure good?
        if self.check_collision(self.agent_location[0], self.agent_location[1], self.tolerance_collision):
            reward = self.reward_collision
            self.terminated = True 
            self.collision = True

        elif np.allclose(self.agent_location, self.target_location, atol=self.tolerance_target_reached): #check if goal is reached by agent.
            reward = self.reward_target_reached
            self.terminated = True
            self.target_reached = True
            self.count_target += 1

        else:
            distance = self._get_norm_dist(self.agent_location, self.target_location)
            reward = self.reward_function(distance)
            if self.verbose == 2:
                print("distance: ", distance)
                print("reward: ", reward)

        # if self.check_goal_reached(self.agent_location[0], self.agent_location[1], self.microbubble_radius):
        #     reward = 1
        #     terminated = True
        if self._elapsed_steps >= self.timeout:
            self.truncated = True

        self.cumulative_reward += reward
        self.last_reward = reward

        #observation = self._get_obs() #np.stack((self.agent_location, self.target_location), axis=-1)
        observation = self._get_obs() # get state s* after taking action a
        info = self._get_info()
        done = self.truncated or self.terminated
        # print("terminated: ", self.terminated)

        return observation, reward, done, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if self.verbose == 2:
            print("Number of episodes completed: ", self.n_elapsed_episodes)
            print("Elapsed steps in last episode: ", self.elapsed_steps) # This is states-1 TODO: Do we need this in our data?
            print("Cumulative_reward: ", self.cumulative_reward)
            print("Resetting environment for next episode")
        self.n_elapsed_episodes += 1
        self._elapsed_steps = 0
        self.state = 0
        self.cumulative_reward = 0
        self.terminated = False

        if self.collision:
            for _ in range(self.config['Action_space_settings']['SAFE_STEPS']):
                safe_piezo = self._get_safe_direction_from_img(self.agent_location)
                direction = self._action_to_direction[int(safe_piezo)]
                self.agent_location = self.agent_location + direction
            self.collision = False  

        if self.random_moves > 0 and np.random.random() < self.random_move_probability:
            for _ in range(self.random_moves):
                action = self._sample_safe_action()
                self.move_agent(action)
        
        # if self.obstacles.get_at((int(self.agent_location[0]), int(self.agent_location[1]))) == (0, 0, 0, 255):
        #     self.agent_location = self.find_legal_point()
        #     self.target_location = self.find_legal_point_target_close(self.agent_location/64)
        #     self.n_elapsed_episodes = 0
        
        # if self.collision:
        #     self.target_location = self.find_legal_point_target_close(self.agent_location/64)
        #     self.new_target_set = True
        #     self.n_elapsed_episodes = 0

        if self.target_reached:
            #self.target_location = self.find_legal_point_target_close(self.agent_location, self.distance_to_new_target)
            self.target_location = self.find_legal_point_target_close(self.agent_location/64)
            self.target_reached = False

        else:
            if self.subepisode_sampling : #and not self.new_target_set:
                if self.n_elapsed_episodes % self.n_subepisodes == 0 or np.allclose(self.agent_location, self.target_location, atol=self.tolerance_target_reached):
                    self.target_location = self.find_legal_point_target_close(self.agent_location/64)
                    self.n_elapsed_episodes = 0
            else:
                self.target_location = self.find_legal_point_target_close(self.agent_location/64)

        observation = self._get_obs()
        info = self._get_info()
        self.episode_count += 1
        
        if self.episode_count % 100 == 0:
            print("\033[92m" + f"Rate target reached {self.count_target/self.episode_count} times" + "\033[0m")
            self.episode_count = 0
            self.count_target = 0
        self.new_target_set = False
        return observation

       
    #find random legal target point 
    def find_legal_point(self):
        location = self.np_random.integers(0, self.config['Layout_settings']['X_MAX'], size=2, dtype=int) #Add cushioning so no immediate edge cases get drawn
        if not self.check_collision(location[0], location[1], 3):
            return np.array([location[0], location[1]], dtype=int) 
        else:
            return self.find_legal_point()
    
    def find_legal_point_target_close(self, start_point: np.ndarray=None):
        min_x = start_point[1] - self.config['General_environment_settings']['MAX_DISTANCE_TARGET_POINT']
        max_x = start_point[1] + self.config['General_environment_settings']['MAX_DISTANCE_TARGET_POINT']
        min_y = start_point[0] - self.config['General_environment_settings']['MAX_DISTANCE_TARGET_POINT']
        max_y = start_point[0] + self.config['General_environment_settings']['MAX_DISTANCE_TARGET_POINT']
        min_x = np.clip(min_x, 0, 1)
        max_x = np.clip(max_x, 0, 1)
        min_y = np.clip(min_y, 0, 1)
        max_y = np.clip(max_y, 0, 1)
        location_x = np.random.uniform(min_x, max_x)
        location_y = np.random.uniform(min_y, max_y)
        location = np.array([location_y*self.window_size, location_x*self.window_size], dtype=int)

        if not np.allclose(start_point*64, location, atol=self.config['General_environment_settings']['MIN_DISTANCE_TO_NEW_TARGET']*64):
            if not self.check_collision(*location, self.tolerance_collision):
                return location
        # pixel_color = self.obstacles.get_at(location)
        # if pixel_color != (0, 0, 0, 255):
        #     return location # This is needed to make coordinates match the image  
        return self.find_legal_point_target_close(start_point)
        
    # def find_legal_point_target_close(self, agent_location, tolerance):
    #         location = self.np_random.integers(agent_location - tolerance, agent_location + tolerance + 1)
    #         if not self.check_collision(location[0], location[1], self.microbubble_radius):
    #             return np.array([location[0], location[1]], dtype=int)
    #         else:
    #             return self.find_legal_point_target_close(agent_location, tolerance)
        
        # location = self.np_random.integers(0, self.config['Layout_settings']['X_MAX'], size=2, dtype=int) #Add cushioning so no immediate edge cases get drawn
        # if not self.check_collision(location[0], location[1], 3):
        #     return np.array([location[0], location[1]], dtype=int) 
        # else:
        #     return self.find_legal_point()
        
    # def find_legal_target_point_normalized(self, agent_location_normalized:
    #     location = self.np_random.integers(0, self.config['Layout_settings']['X_MAX'], size=2, dtype=int) #Add cushioning so no immediate edge cases get drawn
    #     if not self.check_collision(location[0], location[1], 3):
    #         return np.array([location[0], location[1]], dtype=int) 
        # else:
        #     return self.find_legal_point()

    def check_collision(self, player_x, player_y, radius):
        if abs(player_x) + radius >= self.window_size or abs(player_y) + radius >= self.window_size:
            return True
        elif abs(player_x) <= radius or abs(player_y) <= radius:
            return True
        player_x = int(player_x)
        player_y = int(player_y)
        for x in range(player_x - radius, player_x + radius):
            for y in range(player_y - radius, player_y + radius):
                pixel_color = self.obstacles.get_at((x, y))
                if pixel_color == (0, 0, 0, 255):  # Check for black (obstacle) color
                    return True
        return False
    
    def _get_obs(self):
        # Get the rendered image
        #img = self.render()
        img = self._get_image()
        # img_rescaled = self.downscale_img(img)
        
        # Get the agent and target positions
        agent_pos = self.agent_location
        target_pos = self.target_location
        
        # Return a dictionary containing the image and positions
        #obs_dict = {'image': img, 'agent_position': agent_pos, 'target_position': target_pos}
        #return img_rescaled.astype(np.float32)/255.
        # img2 = (img_rescaled.astype(np.float32) / 128.0) - 1.0
        return {
                'image': img,
                'agent_position': (agent_pos/64).astype(np.float32),
                'target_position': (target_pos/64).astype(np.float32),
                }
        # return {
        #         'image': img}
                # 'vector': np.concatenate([agent_pos/64, target_pos/64], axis=0).astype(np.float32),
                
        
        
    #TODO: What info to return?
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self.agent_location - self.target_location, ord=2
                ),
            "TimeLimit.truncated": self.truncated,        
    }

    #TODO: Check if this is correct
    def render_2(self):
        self.render_mode = "rgb_array"
        img = self._render_frame()
        self.render_mode = "human"
        return img

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _get_image(self):
        agent_img, target_img = self._render_frame()
        # print("agent_img: ", agent_img.shape)
        # print("target_img: ", target_img.shape)
        obst = pygame.surfarray.pixels2d(self.obstacles)
        agent = pygame.surfarray.pixels2d(agent_img)
        target = pygame.surfarray.pixels2d(target_img)
        
        image = np.stack((agent, target, obst), axis=-1, dtype=np.uint8)
        # image = self.rescale_img(image)
        return image

    @staticmethod
    def rescale_img(img):
        img = img.astype(np.float32) / 260.0
        img = (img - 0.5) * 2
        img += np.random.normal(0, 0.01, img.shape)
        return img

    def _render_frame(self):
        
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.display_size, self.display_size) # this had 1024, 1024
            )
            pygame.display.set_caption("Microbubble Emulator")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # First we create a blank canvas to draw on
        canvas = pygame.Surface((self.display_size, self.display_size))
        canvas_agent = pygame.Surface((self.window_size, self.window_size))
        canvas_target = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        ratio = self.display_size/self.window_size
        # Draw the environment and player
        canvas.blit(self.obstacles_disp, (0, 0))
        pygame.draw.circle(canvas, (0, 0, 255), self.agent_location*ratio, self.microbubble_radius*ratio)
        pygame.draw.circle(canvas_agent, (0, 0, 255), self.agent_location, self.microbubble_radius)
        pygame.draw.circle(canvas, (255, 0, 0), self.target_location*ratio, self.microbubble_radius*ratio) #TODO: Check when to draw the target point
        pygame.draw.circle(canvas_target, (0, 0, 255), self.target_location, self.microbubble_radius) #TODO: Check when to draw the target point


        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            canvas = canvas.copy()
            self.window.blit(canvas, canvas.get_rect()) #TODO: Should this be (0,0)?
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        # else:  # rgb_array
        #     return np.transpose(
        #         np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2) #window reference maybe wrong
        #     )
        return canvas_agent, canvas_target

    def _sample_safe_action(self):
        action = self._randomly_sample_action()
        while not self.is_valid(action):
            action = self._randomly_sample_action()
        return action

    def is_valid(self, action):
        agent_location = self.agent_location + action
        return not self.check_collision(*agent_location, self.tolerance_collision)
    
    def stable_softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    # def _randomly_sample_action(self):
    #     pos = self.get_agent_pos()/64
    #     probabilities = softmax((pos-0.5)**2)  # More probable if away from the center

    #     # Choose randomly a direction in which to go
    #     direction = np.random.choice([0, 1], p=probabilities)
        
    #     probabilities = softmax([(pos[direction])**0.5, (1 - pos[direction])**0.5])  # Trick to make even more probable to go to the center

    #     # Then choose a orientation (left/right, up/down)
    #     orientation = np.random.choice([-1, 1], p=probabilities)
        
    #     out = [0, 0]
    #     out[direction] = orientation
    #     return out
    
    # def _randomly_sample_action(self):
    #     pos = self.get_agent_pos()/64
    #     probabilities = self.stable_softmax((pos-0.5)**2)  # More probable if away from the center

    #     # Choose randomly a direction in which to go
    #     direction = np.random.choice([0, 1], p=probabilities)
        
    #     probabilities = self.stable_softmax([(pos[direction])**0.5, (1 - pos[direction])**0.5])  # Trick to make even more probable to go to the center

    #     # Then choose a orientation (left/right, up/down)
    #     orientation = np.random.choice([-1, 1], p=probabilities)
        
    #     out = [0, 0]
    #     out[direction] = orientation
    #     return out
    
    def _randomly_sample_action(self):
        pos = self.get_agent_pos() / 64
        probabilities = self.stable_softmax((pos - 0.5) ** 2)  # More probable if away from the center

        # Choose randomly a direction in which to go
        direction = np.random.choice([0, 1], p=probabilities)

        # Ensure that the values are non-negative before applying the square root
        sqrt_pos = np.sqrt(np.maximum(0, pos[direction]))
        sqrt_1_minus_pos = np.sqrt(np.maximum(0, 1 - pos[direction]))

        probabilities = self.stable_softmax([sqrt_pos, sqrt_1_minus_pos])

        # Then choose an orientation (left/right, up/down)
        orientation = np.random.choice([-1, 1], p=probabilities)

        out = [0, 0]
        out[direction] = orientation
        out = np.array(out) * np.random.uniform(0.5, 4)
        return out


    def get_agent_pos(self):
        return self.agent_location
    
    def _set_agent_pos(self, pos):
        self.agent_location = pos

    def move_agent(self, action):
        pos = self.get_agent_pos()
        new_pos = pos + action
        self._set_agent_pos(new_pos)
        return self.agent_location
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()  
    
    def check_collisions(self, player_location_x: int, player_location_y: int):
        collisions = []
        player_location_x = int(player_location_x)
        player_location_y = int(player_location_y)
        for i in range(player_location_x - self.collision_reset_tolerance, player_location_x + self.collision_reset_tolerance):
            for j in range(player_location_y - self.collision_reset_tolerance, player_location_y + self.collision_reset_tolerance):
                if i < 64 and i > 0:
                    if j < 64 and j > 0:
                        pixel_color = self.obstacles.get_at((i, j))
                        if pixel_color == (0, 0, 0, 255):  # Assuming black pixels have a value of 0 pixel_color = self.obstacles.get_at((x, y))
                            if i > player_location_x:
                                collisions.append(PIEZO_DIRECTIONS.RIGHT)
                            elif i < player_location_x:
                                collisions.append(PIEZO_DIRECTIONS.LEFT)
                            if j > player_location_y:
                                collisions.append(PIEZO_DIRECTIONS.DOWN)
                            elif j < player_location_y:
                                collisions.append(PIEZO_DIRECTIONS.UP)
        return collisions     
    
    def _get_safe_direction_from_img(self, cluster_center):
        directions = self.check_collisions(cluster_center[0], cluster_center[1])
        if directions == []:
            return PIEZO_DIRECTIONS.OFF
        right_tot = len([i for i in directions if i == PIEZO_DIRECTIONS.RIGHT])
        left_tot = len([i for i in directions if i == PIEZO_DIRECTIONS.LEFT])
        up_tot = len([i for i in directions if i == PIEZO_DIRECTIONS.UP])
        down_tot = len([i for i in directions if i == PIEZO_DIRECTIONS.DOWN])
        horizontal_tot = abs(right_tot - left_tot)
        vertical_tot = abs(down_tot - up_tot)

        if right_tot > left_tot:
            safe_direction_hor = PIEZO_DIRECTIONS.LEFT
        else:
            safe_direction_hor = PIEZO_DIRECTIONS.RIGHT

        if down_tot > up_tot:
            safe_direction_ver = PIEZO_DIRECTIONS.UP
        else:
            safe_direction_ver = PIEZO_DIRECTIONS.DOWN
        
        if horizontal_tot > vertical_tot:
            safe_direction = safe_direction_hor
        elif vertical_tot > horizontal_tot:
            safe_direction = safe_direction_ver
        else:
            safe_direction = np.random.choice([safe_direction_hor, safe_direction_ver])

        if self.verbose == 2:
            print('\033[91mSafe direction: {}\033[0m'.format(PIEZO_DIRECTIONS.convert(safe_direction)))
            print('Safe direction hor: ', PIEZO_DIRECTIONS.convert(safe_direction_hor))
            print('Safe direction ver: ', PIEZO_DIRECTIONS.convert(safe_direction_ver))
            print('right_tot: ', right_tot)
            print('left_tot: ', left_tot)
            print('up_tot: ', up_tot)
            print('down_tot: ', down_tot)
            print()
        return safe_direction

    def downscale_img(self, img):
        return cv2.resize(img, (self.config['Layout_settings']['IMG_DOWNSIZED_SIZE'], self.config['Layout_settings']['IMG_DOWNSIZED_SIZE']), interpolation=cv2.INTER_AREA)
    

class PIEZO_DIRECTIONS:
    # RIGHT = 12 # 1, 2, 3, 4
    # UP = 15
    # LEFT = 14
    # DOWN = 13
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3
    OFF = -1
    def convert(direction):
        if direction == PIEZO_DIRECTIONS.RIGHT:
            return "RIGHT"
        elif direction == PIEZO_DIRECTIONS.UP:
            return "UP"
        elif direction == PIEZO_DIRECTIONS.DOWN:
            return "DOWN"
        elif direction == PIEZO_DIRECTIONS.LEFT:
            return "LEFT"
        elif direction == PIEZO_DIRECTIONS.OFF:
            return "OFF"
        else:
            return "ERROR"+str(direction)
        

class MicrorobotEnvGameRayWrapped(MicrorobotEnv):
    
    def __init__(self, env_config):
        super().__init__(env_config["render_mode"], env_config["microbubble_radius"], env_config["image_string"], 
                         env_config["timeout"], env_config["subepisode_sampling"], env_config["subepisode_length"], env_config["reward_function"],
                         env_config["reward_target_reached"], env_config["reward_collision"], env_config["reward_step"], env_config["const"])


class MicrorobotEnvGameRayWrappedMaS(MicrorobotEnv):
    # wrap with max and skip
    def __init__(self, env_config):
        self.env = MicrorobotEnv(env_config["render_mode"], env_config["microbubble_radius"], env_config["image_string"], 
                         env_config["timeout"], env_config["subepisode_sampling"], env_config["subepisode_length"], env_config["reward_function"],
                         env_config["reward_target_reached"], env_config["reward_collision"], env_config["reward_step"])
        self.env = MaxAndSkipEnv_BoxObs(self.env, skip=2)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
    
    def step(self, action):
        return self.env.step(action)
    
    def reset(self):
        return self.env.reset()

    def close(self):
        return self.env.close()