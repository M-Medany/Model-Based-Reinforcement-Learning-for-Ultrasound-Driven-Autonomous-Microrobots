from typing import Any
import pygame
import numpy as np
import cv2
import cv2
from .microrobot_env import BaseMicrorobotEnv, PIEZO_DIRECTIONS


class MicrorobotEnvContGame(BaseMicrorobotEnv):
    metadata = {"render_modes": ["human", "rgb_array", None], "reward_types": ["sparse", "dense"]}  

    def __init__(self, config, name="sim", render_mode: str="human",
                 image_string=None, max_envs=4, render_fps=0, **kwargs: Any):
        super().__init__(config)
        
        print("extra kwargs: ", kwargs)
        
        self.name = name
        self.max_envs = max_envs
        self.window_size = self.config['Layout_settings']['IMG_SIZE'] # The size of the PyGame window
        self.img_size = self.config['Layout_settings']['IMG_UPSCALED_SIZE']
        self.obstacles = pygame.image.load(image_string) # The image to use as the obstacle environment
        self.obstacles = pygame.transform.scale(self.obstacles, (self.img_size, self.img_size))
        self._fast_obstacles = pygame.surfarray.pixels3d(self.obstacles)
        self.obstacles_cv2 = cv2.resize(cv2.imread(image_string), (self.img_size, self.img_size)).astype(np.uint8)
        self._obstacles_mask = self.obstacles_cv2 == [0, 0, 0]
        
        self.ratio = self.img_size/self.window_size
        self.upscaled_microbubble_radius = self.config['General_environment_settings']['BUBBLE_RADIUS']*self.ratio
        self.min_bubble_radius = self.upscaled_microbubble_radius*self.config['General_environment_settings']['MIN_BUBBLE_RADIUS']
        self.max_bubble_radius = self.upscaled_microbubble_radius*self.config['General_environment_settings']['MAX_BUBBLE_RADIUS']
        self.mean_bubble_radius = self.upscaled_microbubble_radius*self.config['General_environment_settings']['MEAN_BUBBLE_RADIUS']
        self.radius = self.upscaled_microbubble_radius
        self.agent_location = self.find_legal_point(radius=self.radius)
        self.count_episodes_reset_size=0

        # spaces.Dict({"Continous": spaces.Box(low=0, high=2, shape=(1,), dtype=np.float32),
        #                               "Discrete": spaces.Discrete(4),})

        self._action_to_direction = {
            PIEZO_DIRECTIONS.OFF: np.array([0, 0]),
            PIEZO_DIRECTIONS.UP: np.array([0, -1]),
            PIEZO_DIRECTIONS.RIGHT: np.array([1, 0]),
            PIEZO_DIRECTIONS.LEFT: np.array([-1, 0]),
            PIEZO_DIRECTIONS.DOWN: np.array([0, 1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.render_fps = render_fps

        self.window = None
        self.clock = None

        self.tolerance_collision = self.tolerance_collision*self.ratio
        self.tolerance_target_reached = self.tolerance_target_reached*self.ratio
        self.count_target = 0

        self.target_location = self.find_legal_point_target_close(self.agent_location/self.img_size, radius=self.radius)

    def _get_norm_dist(self, agent_location, target_location):
        aget_loc_norm = agent_location / np.array(self.img_size)
        target_loc_norm = target_location / np.array(self.img_size)
        return np.linalg.norm(aget_loc_norm - target_loc_norm, ord=2)
    
    def step(self, action):
        # print("direction: ", direction)
        # print("amplitude: ", amplitude)
        act_num = action % self.config['Action_space_settings']['NUMBER_PIEZOS']
        direction = self._action_to_direction[int(act_num)+1]
        amplitude = self.ratio*np.random.uniform(0.5, 1.5)
        return self._post_step(direction, amplitude)

    def _post_step(self, direction, amplitude):
        if self.is_valid(direction, radius=self.get_radius()+self.tolerance_collision):
            self.agent_location = self.move_agent(direction, amplitude)
        else:
            self.collision = True
            self.terminated = True
            self.agent_location = self.move_agent(direction, amplitude)
            observation = self._get_obs() # get state s* after taking action a
            info = self._get_info()
            self.agent_location = self.move_agent(direction, -amplitude)
            return observation, self.reward_collision, True, info

        # Set done for state s* to False
        self.terminated = False
        self.truncated = False
        self._elapsed_steps += 1
        tol_col = self.get_radius() + self.tolerance_collision
        tol_target = self.get_radius() + self.tolerance_target_reached

        if self.check_collision(*self.agent_location, tol_col):
            reward = self.reward_collision
            self.terminated = True 
            self.collision = True

        elif np.allclose(self.agent_location, self.target_location, atol=tol_target):
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
        
        # self._update_radius()
        done = self.terminated
        return self._get_obs(), reward, done, self._get_info()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._update_radius()
        self.count_episodes_reset_size += 1
        if self.count_episodes_reset_size % 30 == 0:
            self._reset_radius()
        if self.verbose == 2:
            print("Number of episodes completed: ", self.n_elapsed_episodes)
            print("Elapsed steps in last episode: ", self.elapsed_steps) # This is states-1 TODO: Do we need this in our data?
            print("Resetting environment for next episode")
        self.n_elapsed_episodes += 1
        self._elapsed_steps = 0
        self.terminated = False
        tol = self.get_radius() + self.tolerance_collision*2

        if self.collision:
            for _ in range(10):
                pos = self.get_agent_pos()
                safe_piezo = self._get_safe_direction_from_img([int(pos[0]), int(pos[1])], radius=tol)
                direction = self._action_to_direction[int(safe_piezo)]
                self.agent_location = self.move_agent(direction, 1)
            self.collision = False  

        if self.random_move_probability > 0 and np.random.random() < self.random_move_probability:
            self.agent_location = self.find_legal_point(radius=tol)

        if self.target_reached:
            self.target_location = self.find_legal_point_target_close(self.get_agent_pos(), radius=tol)
            self.target_reached = False

        else:
            if self.subepisode_sampling:
                if self.n_elapsed_episodes % self.n_subepisodes == 0 or np.allclose(self.get_agent_pos(), self.target_location, atol=self.tolerance_target_reached):
                    self.target_location = self.find_legal_point_target_close(self.get_agent_pos(), radius=tol)
                    self.n_elapsed_episodes = 0
            else:
                self.target_location = self.find_legal_point_target_close(self.get_agent_pos(), radius=tol)
        return self._get_obs()

    def _get_obs(self):
        img = self._get_image()

        agent_pos = self.get_agent_pos()/self.img_size
        target_pos = self.target_location/self.img_size

        return {'image': img,
                'agent_position': agent_pos.astype(np.float32),
                'target_position': target_pos.astype(np.float32),
                }

    def _get_image(self):
        self._render_frame()
        radius = self.get_radius()
        
        obs = np.full((self.img_size, self.img_size, 3), 255, dtype=np.uint8)
        agent_loc = tuple(int(self.agent_location[i]) for i in range(2))
        target_loc = tuple(int(self.target_location[i]) for i in range(2))
        cv2.circle(obs, target_loc, int(self.upscaled_microbubble_radius), (0, 0, 255), -1)
        cv2.circle(obs, agent_loc, int(radius), (255, 0, 0), -1)
        obs = np.where(self._obstacles_mask, self.obstacles_cv2, obs)

        image =  cv2.resize(obs, (self.config['Layout_settings']['IMG_DOWNSIZED_SIZE'], self.config['Layout_settings']['IMG_DOWNSIZED_SIZE']), interpolation=cv2.INTER_AREA)
        if self.render_mode == "human":
            cv2.imshow("image", self.upscale_img(image))
            cv2.waitKey(1)

        return image

    def _render_frame(self):
        pass
        # if self.window is None and self.render_mode == "human":
        #     print("Creating window")
        #     self._multi_window_init()
            
        # if self.clock is None and self.render_mode == "human" and self.render_fps > 0:
        #     self.clock = pygame.time.Clock()

        # if self.render_mode == "human":
        #     # First we create a blank canvas to draw on
        #     canvas = pygame.Surface((self.img_size, self.img_size))
        #     canvas.fill((255, 255, 255))
        #     # Draw the environment and player
        #     canvas.blit(self.obstacles, (0, 0))
        #     pygame.draw.circle(canvas, (0, 0, 255), self.agent_location, self.radius)
        #     pygame.draw.circle(canvas, (255, 0, 0), self.target_location, self.upscaled_microbubble_radius)
        #     self.window.blit(canvas, self.number_to_coordinate(self.name))
        #     if self.render_fps > 0:
        #         self.clock.tick(self.render_fps)
        #     pygame.event.pump()
        #     pygame.display.update()

    
    def _update_radius(self, value=None):
        if value is None:
            self.radius += np.random.normal(0, 0.3)*(self.max_bubble_radius - self.min_bubble_radius)/10
            self.radius = np.clip(self.radius, self.min_bubble_radius + 0.15, self.max_bubble_radius)
        elif value == 0:
            pass
        else:
            self.radius += value
            self.radius = np.clip(self.radius, self.min_bubble_radius, self.max_bubble_radius)

    def _reset_radius(self):
        self.radius = self.mean_bubble_radius
    
    def get_radius(self):
        return self.radius
    
    def _multi_window_init(self):
        pygame.init()
        pygame.display.init()
        self.rows = self.config["Multienv"].get("rows", 2)
        self.cols = self.config["Multienv"].get("cols", 3)
        self.window = pygame.display.set_mode(
                (self.img_size * self.cols, self.img_size * self.rows)
            )
        pygame.display.set_caption(f"Microbubble Emulator")

        # np.ceil(np.sqrt(self.max_envs))
        # self.cols = 3 # np.ceil(self.max_envs / self.rows)
    
    def number_to_coordinate(self, number):
        # assert 1 <= number <= self.max_envs, f"Number must be between 1 and {self.max_envs}"
        return (number[0]*self.img_size, number[1]*self.img_size)

    def get_agent_pos(self):
        return self.agent_location
    
    def _set_agent_pos(self, pos):
        self.agent_location = pos

    def move_agent(self, action, amplitude=1.0, fake=False):
        if fake:
            return super().move_agent(action, amplitude, True)
        new_pos = self.get_agent_pos() + action*amplitude
        self._set_agent_pos(new_pos)
        return self.agent_location
    
    def close(self):
        super().close()
        pygame.display.quit()
        pygame.quit()
        print("Closing window")
        self.window = None
        self.clock = None
    
    def _eval_pixel_collision(self, x, y, obstacles=None):
        if obstacles is None:
            obstacles = self._fast_obstacles
        mask = (x < 0) | (x >= self.img_size) | (y < 0) | (y >= self.img_size)
        if np.any(mask):
            return True
        pixel_colors = obstacles[x.flat, y.flat]
        if np.any(np.all(pixel_colors == (0, 0, 0), axis=1)):
            return True
        return False
     

class MicrorobotEnvGameRayWrappedCont(MicrorobotEnvContGame):
    
    def __init__(self, env_config):
        super().__init__(env_config["render_mode"], env_config["microbubble_radius"], env_config["image_string"], 
                         env_config["timeout"], env_config["subepisode_sampling"], env_config["subepisode_length"], env_config["reward_function"],
                         env_config["reward_target_reached"], env_config["reward_collision"], env_config["reward_step"], env_config["const"])
