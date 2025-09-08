import gymnasium as gym
from gymnasium import spaces
from matplotlib.pyplot import step
import numpy as np
import yaml
from utils.image_postprocessing import find_legal_point_target_close
import cv2
from abc import ABC, abstractmethod


class BaseMicrorobotEnv(gym.Env, ABC):

    def __init__(self, config):
        
        with open(config, 'r') as yaml_file:
            self.config = yaml.safe_load(yaml_file)
        self.verbose = self.config['General_environment_settings']['VERBOSE']
        
        reward_function = self.config["Reward Settings"]["reward_function"]
        reward_target_reached = self.config["Reward Settings"]["reward_target_reached"]
        reward_collision = self.config["Reward Settings"]["reward_collision"]
        reward_step = self.config["Reward Settings"]["reward_step"]
        const = self.config["Reward Settings"]["const"]

        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(self.config['Layout_settings']['IMG_SIZE'], self.config['Layout_settings']['IMG_SIZE'], 3), dtype=np.uint8,),
            "agent_position": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "target_position": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            })
        
        self.action_space = spaces.Discrete(self.config['Action_space_settings']['TOTAL_ACTIONS'])

        if reward_function == 'linear':
            self.reward_function = lambda x: reward_step*x + const
        elif reward_function == 'quadratic':
            self.reward_function = lambda x: reward_step*x**2 + const
        elif reward_function == 'log':
            self.reward_function = lambda x: reward_step*np.log(x) + const
        elif reward_function == 'inverse':
            assert reward_step > 0, "Reward step needs to be positive for inverse reward function"
            self.reward_function = lambda x: reward_step*(1/(x+0.1)) + const
        elif reward_function == 'inverse_squared':
            assert reward_step > 0, "Reward step needs to be positive for inverse reward function"
            self.reward_function = lambda x: reward_step*(1/(x**2+0.1)) + const
        else:
            raise Exception("Reward function not implemented")

        self.reward_target_reached = reward_target_reached
        self.reward_collision = reward_collision
        self.reward_termination = self.config["Reward Settings"]["reward_termination"] if "reward_termination" in self.config["Reward Settings"] else 0

        self.segmented = None

        self.tolerance_target_reached = self.config['General_environment_settings']['TARGET_REACHED_TOLERANCE'] # This decides how close the agent needs to be to the target to be considered reached
        self.tolerance_collision = self.config['General_environment_settings']['COLLISION_TOLERANCE'] # How close do we count it as a collision
        self.tolerance_collision_reset = self.config['General_environment_settings']['COLLISION_RESET_TOLERANCE'] # How close do we count it as a collision
        self.random_move_probability = self.config['General_environment_settings']['RANDOM_MOVE_PROBABILITY']
        self.n_safe_steps = self.config['Action_space_settings']['SAFE_STEPS']
        self.subepisode_sampling = (int(self.config['General_environment_settings']['SUBEPISODE_LENGTH']) > 0)
        self.n_subepisodes = int(self.config['General_environment_settings']['SUBEPISODE_LENGTH'])
        
        self.elapsed_steps = 0
        self.last_piezo = 0
        self.collision = False
        self.cumulative_reward = 0
        self.last_reward = 0
        self.state = 0
        self.tot_steps = 0
        self.n_elapsed_episodes = -1
        self.terminated = False
        self.truncated = False
        self.first_obs = True
        self.collision_reset_steps = False
        self.target_reached = False

    @abstractmethod
    def _get_obs(self):
        pass

    @abstractmethod
    def _eval_pixel_collision(self, x, y):
        pass

    @abstractmethod
    def get_agent_pos(self):
        pass
    
    @abstractmethod
    def _set_agent_pos(self, pos):
        pass

    @abstractmethod
    def move_agent(self, action, amplitude=1, fake=False):
        if fake:
            return self.get_agent_pos() + action*amplitude
        raise NotImplementedError("move_agent not implemented")
    
    def get_target_pos(self):
        return self.target_location
    
    def _set_target_pos(self, pos):
        self.target_location = pos

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self.get_agent_pos() - self.get_target_pos(), ord=2
                ),
            "target_reached": self.target_reached,
    }
    
    def find_legal_point(self, radius=None):
        location = self.np_random.integers(0, self.img_size, size=2, dtype=int)
        if not self.check_collision(location[0], location[1], radius):
            return np.array([location[0], location[1]], dtype=int) 
        else:
            return self.find_legal_point(radius=radius)

    def check_collision(self, player_x, player_y, radius=None):
        if radius is None:
            radius = self.tolerance_collision
        radius_u = int(np.ceil(radius))
        radius_l = int(np.floor(radius))
        if abs(player_x) + radius_u >= self.img_size or abs(player_y) + radius_u >= self.img_size:
            return True
        elif abs(player_x) <= radius_l or abs(player_y) <= radius_l:
            return True
        player_x = int(player_x)
        player_y = int(player_y)
        x = np.arange(player_x - radius_l, player_x + radius_l)
        y = np.arange(player_y - radius_l, player_y + radius_l)
        xx, yy = np.meshgrid(x, y)
        return np.any(self._eval_pixel_collision(xx, yy))
        # for x in range(player_x - radius_l, player_x + radius_l):
        #     for y in range(player_y - radius_l, player_y + radius_l):
        #         if self._eval_pixel_collision(x, y):
        #             return True
        # return False
    
    def find_legal_point_target_close(self, start_point: np.ndarray=None, radius=None):
        its = 0
        dist = self.config['General_environment_settings']['MAX_DISTANCE_TARGET_POINT']
        min_dist = self.config['General_environment_settings']['MIN_DISTANCE_TO_NEW_TARGET']
        radius = self.tolerance_collision_reset if radius is None else radius

        def inner(start_point):
            if any(start_point[i] > 1 for i in range(2)):
                start_point = start_point/self.img_size
            min_x = start_point[1] - dist
            max_x = start_point[1] + dist
            min_y = start_point[0] - dist
            max_y = start_point[0] + dist
            min_x = np.clip(min_x, 0, 1)
            max_x = np.clip(max_x, 0, 1)
            min_y = np.clip(min_y, 0, 1)
            max_y = np.clip(max_y, 0, 1)
            location_x = np.random.uniform(min_x, max_x)
            location_y = np.random.uniform(min_y, max_y)
            location = np.array([location_y*self.img_size, location_x*self.img_size], dtype=int)
            return location

        location = inner(start_point)

        while np.allclose(start_point*self.img_size, location, atol=min_dist*self.img_size) or self.check_collision(*location, radius):
            location = inner(start_point)
            its += 1
            if its > 1e3:
                print(f"Could not find a legal point for the target, augmenting the radius to: {dist*1.5}")
                dist = dist*1.5
                if dist > 10:
                    min_dist /= 2
                    print(f"Could not find a legal point for the target, diminishing the min_dist to: {min_dist}")
        return location

    def check_collisions(self, player_x, player_y, radius=None):
        if radius is None:
            radius = self.tolerance_collision
        assert isinstance(player_x, int), "player_x is not an integer: {}".format(player_x)
        assert isinstance(player_y, int), "player_y is not an integer: {}".format(player_y)
        radius_u = int(np.ceil(radius))
        
        x = np.arange(player_x - radius_u, player_x + radius_u)
        y = np.arange(player_y - radius_u, player_y + radius_u)
        xx, yy = np.meshgrid(x, y)
        collisions = self._eval_pixel_collision(xx, yy)
        
        right_mask = xx > player_x
        left_mask = xx < player_x
        down_mask = yy > player_y
        up_mask = yy < player_y
        
        directions = []
        if np.any(collisions & right_mask):
            directions.append(PIEZO_DIRECTIONS.RIGHT)
        if np.any(collisions & left_mask):
            directions.append(PIEZO_DIRECTIONS.LEFT)
        if np.any(collisions & down_mask):
            directions.append(PIEZO_DIRECTIONS.DOWN)
        if np.any(collisions & up_mask):
            directions.append(PIEZO_DIRECTIONS.UP)
        
        return directions
    
    def _get_safe_direction_from_img(self, cluster_center, radius=None):
        directions = self.check_collisions(cluster_center[0], cluster_center[1], radius)
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

        if self.verbose > 1:
            print('\033[91mSafe direction: {}\033[0m'.format(PIEZO_DIRECTIONS.convert(safe_direction)))
        if self.verbose > 2:
            print('Safe direction hor: ', PIEZO_DIRECTIONS.convert(safe_direction_hor))
            print('Safe direction ver: ', PIEZO_DIRECTIONS.convert(safe_direction_ver))
            print('right_tot: ', right_tot)
            print('left_tot: ', left_tot)
            print('up_tot: ', up_tot)
            print('down_tot: ', down_tot)
            print()
        return safe_direction
    
    def _sample_safe_action(self, radius=None):
        action = self._randomly_sample_action()  # --> [0, 1] or [1, 0]
        i = 0
        while not self.is_valid([act*2 for act in action], radius=radius):
            action = self._randomly_sample_action()
            i += 1
            if i > 1e3:
                print("Could not find a safe action")
                action = [0, 0]
                break
        return action
    
    def is_valid(self, action, amplitude=1, radius=None):
        agent_location = self.move_agent(action, amplitude, fake=True)
        return not self.check_collision(*agent_location, radius)
    
    def stable_softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def _randomly_sample_action(self):
        pos = self.get_agent_pos()/np.array([self.width_up, self.length_up])
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
        return out
    
    def _get_piezo_direction(self, direction):
        if direction == [1, 0]:
            return PIEZO_DIRECTIONS.RIGHT
        elif direction == [-1, 0]:
            return PIEZO_DIRECTIONS.LEFT
        elif direction == [0, 1]:
            return PIEZO_DIRECTIONS.UP
        elif direction == [0, -1]:
            return PIEZO_DIRECTIONS.DOWN    
        else:
            return PIEZO_DIRECTIONS.OFF   

    def downscale_img(self, img):
        return cv2.resize(img, (self.config['Layout_settings']['IMG_DOWNSIZED_SIZE'], self.config['Layout_settings']['IMG_DOWNSIZED_SIZE']), interpolation=cv2.INTER_AREA)
    
    def rescale_img(self, img):
        return cv2.resize(img, (self.config['Layout_settings']['IMG_SIZE'], self.config['Layout_settings']['IMG_SIZE']), interpolation=cv2.INTER_AREA)
    
    def upscale_img(self, img):
        return cv2.resize(img, (self.config['Layout_settings']['IMG_UPSCALED_SIZE'], self.config['Layout_settings']['IMG_UPSCALED_SIZE']), interpolation=cv2.INTER_NEAREST)
    
    def new_target(self, agent_location_normalized):
        target_location = self.find_legal_point_target_close(agent_location_normalized)
        while self.check_collision(*target_location, self.tolerance_collision):
            target_location = self.find_legal_point_target_close(agent_location_normalized)
        return target_location
    
    @abstractmethod
    def _get_norm_dist(self, agent_location, target_location):
        pass
    
class PIEZO_DIRECTIONS:
    OFF = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3
    UP = 4
    DOWN_RIGHT = 5
    DOWN_LEFT = 6
    UP_RIGHT = 7
    UP_LEFT = 8

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
        elif direction == PIEZO_DIRECTIONS.DOWN_RIGHT:
            return "DOWN_RIGHT"
        elif direction == PIEZO_DIRECTIONS.DOWN_LEFT:
            return "DOWN_LEFT"
        elif direction == PIEZO_DIRECTIONS.UP_RIGHT:
            return "UP_RIGHT"
        elif direction == PIEZO_DIRECTIONS.UP_LEFT:
            return "UP_LEFT"
        return f"ERROR, {direction}"
