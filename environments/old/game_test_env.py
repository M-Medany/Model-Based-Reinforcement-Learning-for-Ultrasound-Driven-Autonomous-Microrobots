import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import yaml
import os


class MicrorobotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 90, "reward_types": ["sparse", "dense"]}  

    def __init__(self, render_mode="human", microbubble_radius=5, image=None, timeout=100): #TODO: we could include reward type. timeout in config?, include timeout in code

        # Load the YAML file
        with open(f'scripts/config.yaml', 'r') as yaml_file:
            self.config = yaml.safe_load(yaml_file)

        self.microbubble_radius = microbubble_radius  # The size of the microbubble
        self.window_size = self.config['Layout_settings']['IMG_SIZE'] # The size of the PyGame window
        self.image = image  # The image to use as the environment
        self.obstacles = pygame.image.load(f"{image}") # The image to use as the obstacle environment
        self.obstacles = pygame.transform.scale(self.obstacles, (self.config['Layout_settings']['IMG_SIZE'], self.config['Layout_settings']['IMG_SIZE']))
        self.num_envs = 1
        self.obstacles_disp = pygame.transform.scale(self.obstacles, (1024, 1024))
        # Do we need an initial position for the robot?
        # self.agent_location = np.array([self.config['Layout_settings']['X_MIN'], self.config['Layout_settings']['Y_MIN']])  # The initial location of the agent TODO: Check the coordinates

        
        # Check the observations here
        # self.observation_space = spaces.Dict({
        #     'image': spaces.Box(low=0, high=255, shape=(self.config['Layout_settings']['IMG_SIZE'],self.config['Layout_settings']['IMG_SIZE'], 3), dtype=np.float32), #Check that this is correct, maybe low=0, high=255?, also check colour channels
        #     'target_location': spaces.Box(low=[self.config['Layout_settings']['X_MIN'], self.config['Layout_settings']['Y_MIN']], high=[self.config['Layout_settings']['X_MAX'], self.config['Layout_settings']['Y_MAX']], shape=(2,), dtype=int),
        #     'robot_location': spaces.Box(low=[self.config['Layout_settings']['X_MIN'], self.config['Layout_settings']['Y_MIN']], high=[self.config['Layout_settings']['X_MAX'], self.config['Layout_settings']['Y_MAX']], shape=(2,), dtype=int)
        # })

        # Alternative to the above
        self.observation_space = spaces.Dict({
        'image': spaces.Box(low=0, high=255, shape=(self.config['Layout_settings']['IMG_SIZE'], self.config['Layout_settings']['IMG_SIZE'], 3), dtype=np.uint8)
        })
        # self.observation_space = spaces.Box(low=0, high=255, shape=(self.config['Layout_settings']['IMG_SIZE'], self.config['Layout_settings']['IMG_SIZE'], 3), dtype=np.uint8)
        # self.observation_space = spaces.Box(low=0, high=self.window_size-1, shape=(2,2), dtype=int)
        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)
        self.timeout = timeout
        self.count_target = 0
        self.episode_count = 0

        # We need to map actions to directions. This implies move distance of 1 pixel
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        

    def step(self, action):
        direction = self._action_to_direction[int(action)]
        self.agent_location = self.agent_location + direction
        reward = -0.01
        terminated = False
        truncated = False
        self._elapsed_steps += 1
        
        #Is the if if structure good?
        if self.check_collision(self.agent_location[0], self.agent_location[1], self.microbubble_radius):
            reward = -2.0 
            terminated = True 
            #Set info to collision ?
        elif np.allclose(self.agent_location, self.target_location, atol=self.microbubble_radius*2): #check if goal is reached by agent.
            reward = +10.0
            terminated = True
            self.count_target += 1
                         
        else:
            reward += (1/np.linalg.norm(self.agent_location - self.target_location, ord=2)*0.001)

        # if self.check_goal_reached(self.agent_location[0], self.agent_location[1], self.microbubble_radius):
        #     reward = 1
        #     terminated = True
        if self._elapsed_steps >= self.timeout:
            truncated = True

        observation = self._get_obs() # np.stack((self.agent_location, self.target_location), axis=-1)
        info = self._get_info()
        done = truncated or terminated

        return observation, reward, done, info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        self._elapsed_steps = 0

        # Choose the agent's location uniformly at random
        self.agent_location = self.find_legal_point()
        self.target_location = self.find_legal_point()

        observation = self._get_obs()
        info = self._get_info()
        self.episode_count += 1
        
        if self.episode_count % 100 == 0:
            print(f"Rate target reached {self.count_target/self.episode_count} times")   
            self.episode_count = 0
            self.count_target = 0
        # return observation, {}
        return observation
        
    #find random legal target point 
    def find_legal_point(self):
        location = self.np_random.integers(0, self.config['Layout_settings']['X_MAX'], size=2, dtype=int) #Add cushioning so no immediate edge cases get drawn
        if not self.check_collision(location[0], location[1], 3):
            return np.array([location[0], location[1]], dtype=int) 
        else:
            return self.find_legal_point()

    def check_collision(self, player_x, player_y, radius):
        if abs(player_x) + radius >= self.window_size or abs(player_y) + radius >= self.window_size:
            return True
        if abs(player_x) <= radius or abs(player_y) <= radius:
            return True
        for x in range(int(player_x - radius), int(player_x + radius)):
            for y in range(int(player_y - radius), int(player_y + radius)):
                pixel_color = self.obstacles.get_at((x, y))
                if pixel_color == (0, 0, 0, 255):  # Check for black (obstacle) color
                    return True
        return False
    
    def _get_obs(self):
        # Get the rendered image
        #img = self.render()
        img = self._get_image()
        
        # Get the agent and target positions
        agent_pos = self.agent_location
        target_pos = self.target_location
        
        # Return a dictionary containing the image and positions
        obs_dict = {'image': img, 'agent_position': agent_pos, 'target_position': target_pos}
        return {'image': img}

    #TODO: What info to return?
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self.agent_location - self.target_location, ord=2
                ),
            "TimeLimit.truncated": True        
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
                (1024, 1024)
            )
            pygame.display.set_caption("Microbubble Emulator")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # First we create a blank canvas to draw on TODO:  Check this
        canvas = pygame.Surface((1024, 1024))
        canvas_agent = pygame.Surface((self.window_size, self.window_size))
        canvas_target = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        ratio = 1024/self.window_size
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
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()  