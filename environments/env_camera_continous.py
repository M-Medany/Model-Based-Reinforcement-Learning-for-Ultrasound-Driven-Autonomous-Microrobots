from gymnasium import spaces
import numpy as np
import os
from utils.image_postprocessing import create_binary_bitmap, plot_cluster_on_image_blue, resize_and_crop_frame, find_legal_point_target_close, detect_largest_cluster
from utils import ImageSegmentation
from .ARSL_env_camera_dreamer import MicrorobotEnv as DiscreteMicrorobotEnv
from .microrobot_env import PIEZO_DIRECTIONS
from utils.path_planning_v2 import RRTStar
import cv2
import time
import csv
import copy
from .costum_wrappers.OneHot import onehot
from gymnasium import spaces
import yaml
import warnings



class MicrorobotEnvContinous(DiscreteMicrorobotEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100, "reward_types": ["sparse", "dense"]}  

    def __init__(self, fake=False, *args, **kwargs):
        if fake:
            with open(kwargs["config"], 'r') as yaml_file:
                self.config = yaml.safe_load(yaml_file)
            freq_action_space = spaces.Box(low=self.config['Action_space_settings']['MIN_FREQUENCY'], high=self.config['Action_space_settings']['MAX_FREQUENCY'], shape=(8,), dtype=np.float32)
            amplitude_action_space = spaces.Box(low=self.config['Action_space_settings']['MIN_AMPLITUDE'], high=self.config['Action_space_settings']['MAX_AMPLITUDE'], shape=(8,), dtype=np.float32)
            act_space_lows = np.array([[self.config['Action_space_settings']['MIN_FREQUENCY']]*8, [self.config['Action_space_settings']['MIN_AMPLITUDE']]*8])
            act_space_highs = np.array([[self.config['Action_space_settings']['MAX_FREQUENCY']]*8, [self.config['Action_space_settings']['MAX_AMPLITUDE']]*8])
            self.action_space = spaces.Box(low=act_space_lows, high=act_space_highs, dtype=np.float32)
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
        
        min_freq = self.config['Action_space_settings']['MIN_FREQUENCY']
        max_freq = self.config['Action_space_settings']['MAX_FREQUENCY']
        self.substep_reward = self.config["Reward Settings"]['substep_reward']
        self.distance_threshold = self.config['General_environment_settings']['DISTANCE_TO_TARGET_TOLERANCE']

        # self.action_space = spaces.Box(low=min_freq, high=max_freq, shape=(self.config['Action_space_settings']['NUMBER_PIEZOS'],), dtype=np.float32)
        # freq_action_space = spaces.Box(low=self.config['Action_space_settings']['MIN_FREQUENCY'], high=self.config['Action_space_settings']['MAX_FREQUENCY'], shape=(8,), dtype=np.float32)
        # amplitude_action_space = spaces.Box(low=self.config['Action_space_settings']['MIN_AMPLITUDE'], high=self.config['Action_space_settings']['MAX_AMPLITUDE'], shape=(8,), dtype=np.float32)
        # self.action_space = spaces.Dict({"freq": freq_action_space,
        #                                 "amplitude": amplitude_action_space})
        act_space_lows = np.array([[self.config['Action_space_settings']['MIN_FREQUENCY']]*8, [self.config['Action_space_settings']['MIN_AMPLITUDE']]*8])
        act_space_highs = np.array([[self.config['Action_space_settings']['MAX_FREQUENCY']]*8, [self.config['Action_space_settings']['MAX_AMPLITUDE']]*8])
        self.action_space = spaces.Box(low=act_space_lows, high=act_space_highs, dtype=np.float32)
        self.observation_space["piezo"] = spaces.Box(low=0, high=1, shape=(self.config['Action_space_settings']['NUMBER_PIEZOS'],), dtype=np.float32)
        self.observation_space['log_distance_reward'] = spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)
        self.observation_space['log_substep_reward'] = spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)
        self.observation_space['log_reached_reward'] = spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)
        self.observation_space['log_collision_reward'] = spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)
        
        assert self.random_moves == 0, "Random moves must be set to 0 for the continous environment"
        assert hasattr(self, 'planner'), "You must set the Path Planner in the continous environment"
        self.reset_path()

    def step(self, action):
        # start = time.perf_counter()        
        # area = self.tracker.get_bubble_area()
        # vpp = self._vpp_from_area(area)
        # self.function_generator.set_vpp(vpp)

        out = self.path(self.get_agent_pos(), self.target_reached)
        self.target_reached = False
        self.target_location = out["next_waypoint"]
        
        # print("raw action: ", action)
        
        freq = action[0]
        freq = np.clip(freq, self.config['Action_space_settings']['MIN_FREQUENCY'], self.config['Action_space_settings']['MAX_FREQUENCY'])
        
        amplitude = action[1]
        amplitude = np.clip(amplitude, self.config['Action_space_settings']['MIN_AMPLITUDE'], self.config['Action_space_settings']['MAX_AMPLITUDE'])

        # action = np.clip(action, self.config['Action_space_settings']['MIN_FREQUENCY'], self.config['Action_space_settings']['MAX_FREQUENCY'])
        freq = freq[out["piezo"]-1]  # Piezos are shift by 1 (they are 1-indexed)
        amplitude = amplitude[out["piezo"]-1]  # Piezos are shift by 1 (they are 1-indexed)

        self.function_generator.set_frequency(freq)
        self.function_generator.set_vpp(amplitude)
        self.arduino.set_piezo_by_number(out["piezo"])

        # time.sleep(self.config['Action_space_settings']['STEP_DURATION'])
        # print(f"Action: {action}, Piezo: {piezo}, Frequency: {action}, Substep amount: {substep_amount}")
        return self._post_step(vpp=amplitude, freq=freq, **out)
    
    def reset(self, *args, seed=None, options=None):    
        with open(f"{self.save_path_experiment}/episode_plots.csv", 'a', newline="") as file:
            writer = csv.writer(file)
            writer.writerow([self.elapsed_steps, self.cumulative_reward, self.collision, self.target_reached, self.truncated])
    
        replan = (self.is_target_reached() or self.collision)
        obs = super().reset(*args, seed=seed, options=options)
        if replan:
            self.reset_path()
        obs["piezo"] = self.path(self.get_agent_pos(), False, True)["next_piezo"]
        obs["log_distance_reward"] = np.array([0])
        obs["log_substep_reward"] = np.array([0])
        obs["log_reached_reward"] = np.array([0])
        obs["log_collision_reward"] = np.array([0])
        
        return obs

    def reset_path(self):
        self._get_obs_during_reset()
        its = 0
        while self.planner.inside_obstacle(self.get_agent_pos()):
            self.collision_reset(radius=self.config['Path Planning']['safety_threshold']+5)
            its += 1
            if its % 20 == 0:
                print("Resetting tracker")
                self.arduino.set_piezo_by_number(0)
                self.frame_green_box, success = self._reinitialize_tracker_autonomously(self.cleaned_image, location=self.agent_location)
        
        target = self.new_target(self.get_agent_pos()/self.img_size)
        while self.planner.inside_obstacle(target):
            target = self.new_target(self.get_agent_pos()/self.img_size)

        self.planner.set_start_end(self.get_agent_pos(), target)
        path = self.planner.plan()
        while path is None or len(path) < self.config['Path Planning']['depth']+1:
            print("No path found. Trying again...")
            if path is not None:
                print(f"Path length: {len(path)}")
            path = self.planner.plan()
        self.path = PathFollower(path, self.img_size, 
                                 tolerance=self.config['Path Planning']['tol'], 
                                 depth=self.config['Path Planning']['depth'],
                                 max_dist=self.config['Path Planning']['max_dist'])
        self.target_location = self.path(self.get_agent_pos(), True)["next_waypoint"]
    
    def is_target_reached(self):
        return self.path._is_target_reached()
    
    def _post_step(self, piezo, vpp, freq, rew, **kwargs):
        self.elapsed_steps += 1
        self.terminated = False
        self.truncated = False
        substep_reward = rew*self.substep_reward
        collision_reward = 0
        reward_target_reached = 0
        distance_reward = 0
        self.state += 1
        w, h = self.tracker.get_bbox_width_and_height()
        blue_area = self.tracker.get_bubble_area()
        self.agent_location = self.get_agent_pos()

        observation = self._get_obs()

        info = self._get_info()
        distance = self._get_norm_dist(self.agent_location, self.target_location)
        
        if distance > self.distance_threshold or kwargs['failed']:
            collision_reward += self.reward_collision
            self.terminated = True 
            self.collision = True
        elif np.allclose(self.agent_location, self.target_location, atol=self.tolerance_target_reached):
            reward_target_reached += self.reward_target_reached
            self.terminated = True
            self.target_reached = True       
        else:
            distance_reward += self.reward_function(distance)
        
        reward = distance_reward + collision_reward + reward_target_reached + substep_reward

        self.cumulative_reward += reward
        self.last_reward = reward
        self.last_piezo = piezo

        with open(f'{self.save_path_experiment}/experiment_data.csv', 'a', newline="") as file:
            writer = csv.writer(file)
            writer.writerow([self.elapsed_steps, self.state, self.agent_location[0], self.agent_location[1], self.target_location[0], self.target_location[1], piezo, vpp, freq, self.last_reward, self.cumulative_reward, self.terminated, self.truncated, False, w, h, self.bound_x, self.bound_y, self.bound_width, self.bound_height, blue_area])
        
        if self.verbose == 4:
            print('\033[94mBlue area step: ', self.current_bubble_area, 'Initial blue area: ', self.initial_bubble_area*self.size_threshold, '\033[0m')

        while self.current_bubble_area < self.initial_bubble_area*self.size_threshold:
            print("\033[91mTracker lost the bubble! Reinitializing in step..\033[0m")
            self.arduino.set_piezo_by_number(0)
            self.frame_green_box, success = self._reinitialize_tracker_autonomously(self.cleaned_image, location=self.agent_location)
            observation = self._get_obs()

        done = self.terminated
        if self.is_target_reached():
            info["TimeLimit.truncated"] = True
            done = True
            next_piezo = np.zeros(8, dtype=np.float32)
        else:
            next_piezo = self.path(self.get_agent_pos(), self.target_reached, True)["next_piezo"]
        
        observation["log_distance_reward"] = np.array([distance_reward])
        observation["log_substep_reward"] = np.array([substep_reward])
        observation["log_reached_reward"] = np.array([reward_target_reached])
        observation["log_collision_reward"] = np.array([collision_reward])
        observation["piezo"] = next_piezo
        
        with open(f'{self.save_path_experiment}/plots.csv', 'a', newline="") as file:
            writer = csv.writer(file)
            writer.writerow([self.elapsed_steps, reward, distance_reward, substep_reward, reward_target_reached, collision_reward])
        # end = time.perf_counter()
        # diff = end - kwargs['start']
        # print(f"Step time: {diff}, fps: {1/diff}")
            
        return observation, reward, done, info
    
    def _get_obs(self, path=None):
        ret, frame = self.video_stream.read()
        if not ret:
            raise Exception("Could not read from camera on episode, step: " + str(self.n_elapsed_episodes), str(self.elapsed_steps))
        
        _, cropped = resize_and_crop_frame(frame, self.roi_x, self.roi_y, self.roi_width, self.roi_height)
        # cv2.imwrite(f'{self.save_path_original_data}/original_frame_episode_{self.n_elapsed_episodes}_state_{self.state}.jpeg', cropped, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])

        self.cleaned_image = plot_cluster_on_image_blue(self.segmented, cropped, self.threshold)
        # cv2.imwrite(f'{self.save_path_blue_data}/blue_frame_episode_{self.n_elapsed_episodes}_state_{self.state}.png', self.cleaned_image)
        cleaned_image_copy = np.copy(self.cleaned_image)

        # Update tracker and agent position
        self.frame_green_box, success = self.tracker.track(self.cleaned_image)
        while not success:
            self.frame_green_box, success = self._reinitialize_tracker_autonomously(self.cleaned_image, location=self.agent_location)
        self.current_bubble_area = self.tracker.get_bubble_area()

        # Plot the target point onto the image -> This is what dreamer sees
        cv2.circle(cleaned_image_copy, tuple(self.target_location), self.bubble_size, (0, 0, 255), -1)
        # cv2.imwrite(f'{self.save_path_data_with_target_point}/clean_frame_episode_{self.n_elapsed_episodes}_state_{self.state}.png', cleaned_image_copy)

        # This is what we see externally which includes the tracking box
        cv2.circle(self.frame_green_box, tuple(self.target_location), self.bubble_size, (0, 0, 255), -1) 
        # cv2.imwrite(f'{self.save_path_tracking_data}/annotated_frame_episode_{self.n_elapsed_episodes}_state_{self.state}.png', self.frame_green_box)

        cleaned_image_copy = self.downscale_img(cleaned_image_copy)
            
        # cv2.imshow("Annotated Image", self.upscale_img(cleaned_image_copy))
        
        if hasattr(self, 'path'):
            path = self.path.return_path()
        if path is not None:
            for i in range(len(path) - 1):
                cv2.line(self.frame_green_box, tuple(path[i]), tuple(path[i + 1]), (0, 0, 255), 1)
                cv2.circle(self.frame_green_box, tuple(path[i]), 2, (0, 255, 255))
        self.tot_steps += 1
        if self.tot_steps % 1 == 0:
            cv2.imwrite(f'{self.save_path_downsized_data}/{self.tot_steps}_downsized_frame_episode_{self.n_elapsed_episodes}_state_{self.state}.png', cleaned_image_copy)
            cv2.imwrite(f'{self.save_path_RRT}/{self.tot_steps}_rrt_frame_episode_{self.n_elapsed_episodes}_state_{self.state}.png', self.frame_green_box)
            cv2.imwrite(f'{self.save_path_data_with_target_point}/{self.tot_steps}_clean_frame_episode_{self.n_elapsed_episodes}_state_{self.state}.png', cleaned_image_copy)
            cv2.imwrite(f'{self.save_path_original_data}/{self.tot_steps}_original_frame_episode_{self.n_elapsed_episodes}_state_{self.state}.jpeg', cropped, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            cv2.imwrite(f'{self.save_path_blue_data}/{self.tot_steps}_blue_frame_episode_{self.n_elapsed_episodes}_state_{self.state}.png', self.cleaned_image)

        cv2.imshow("Annotated frame", self.upscale_img(self.frame_green_box))
        cv2.waitKey(1)

        agent_position_normalized = self.get_agent_pos() / self.img_size
        target_position_normalized = self.target_location / self.img_size

        return {"image": cleaned_image_copy.astype(np.uint8), 
                "agent_position": agent_position_normalized.astype(np.float32), 
                "target_position": target_position_normalized.astype(np.float32), 
                }


class PathFollower():
    def __init__(self, path, img_size, max_dist=0.3, tolerance=0.01, depth=6):
        self.path = path
        self.og_path = copy.deepcopy(path)
        self.target_reached = False
        self.i = 0
        self.next_depth = depth
        self.tolerance = tolerance
        self.max_dist = max_dist
        self.img_size = img_size
        self.next_waypoint = np.array(self.path.pop(0))
        self.next_next_waypoint = np.array(self.og_path[self.next_depth])
        self.target_reached_count = 0

    def __call__(self, pos_unormalized, target_reached=False, next_step=False):
        if self.target_reached:
            warnings.warn("Target reached, this should not happen!", RuntimeWarning)
            return {"piezo": 0, "next_waypoint": self.next_next_waypoint, "rew": 0, "next_piezo": np.zeros(8, dtype=np.float32), "failed": False}
        
        # Normalize position
        pos = pos_unormalized/self.img_size
        rew = 0
        self.i += 1
        
        out = {"piezo": 0, "next_waypoint": self.next_next_waypoint, "rew": 0, "next_piezo": np.zeros(8, dtype=np.float32), "failed": False}

        # Check for reached waypoints
        furthest_seen = -1
        # distances = []
        for idx, wayp in enumerate(self.path[:self.next_depth*2]):
            if np.linalg.norm(pos - np.array(wayp)/self.img_size) < self.tolerance:
                rew += 1
                furthest_seen = max(idx, furthest_seen)
                # distances.append(np.linalg.norm(pos_unormalized - wayp))
                # print(f"Next waypoint: {np.array(self.path[idx]) / self.img_size}")
        out['rew'] = rew

        # If next_step is True, calculate the next action and return
        if next_step:
            self.i -= 1
            if len(self.path) <= furthest_seen+1:
                self.target_reached = True
                return out
            piezo = self.calc_action(pos, np.array(self.path[furthest_seen+1]) / self.img_size)
            out['next_piezo'] = onehot(8, piezo-1)
            return out

        # Check if maximum steps reached
        if self.i == 3000:
            print("3000 steps reached")
            self.target_reached = True
            return out

        # If any waypoints were reached, update the path and next waypoint
        if furthest_seen >= 0:
            self.path = self.path[furthest_seen+1:]
            if not self.path:
                print("Path empty, target reached")
                self.target_reached = True
                return out
            self.next_waypoint = np.array(self.path[0])
        
        if np.linalg.norm(pos - self.next_waypoint/self.img_size) > self.max_dist:
            self.target_reached = True
            out['failed'] = True
            return out

        # If target_reached is True, update the target_reached_count and next_next_waypoint
        if target_reached:
            self.target_reached_count += 1
            depth = self.next_depth*self.target_reached_count
            if len(self.og_path) > depth:
                self.next_next_waypoint = np.array(self.og_path[depth])
            else:
                self.target_reached = True
                return out
            print(f"Target reached, next waypoint: {self.next_next_waypoint/self.img_size}")

        piezo = self.calc_action(pos, self.next_waypoint / self.img_size)
        out['piezo'] = piezo
        out['next_waypoint'] = self.next_next_waypoint

        return out
    
    def calc_action(self, pos, target):
        """
        Calculate optimal piezo to actuate
        :param pos:    position
        :param offset:  Offset to target in pixel difference np.array([x, y])
        :return:        integer in [1, 2, 3, 4] with len() = number of piezos
        """
        x = target[0] - pos[0]  # x = target - pos
        y = -target[1] + pos[1]  # y = 1-target - (-1-pos)
        angle = np.arctan2(y, x)
        angle = np.degrees(angle)
        angle_partition = (angle + 382.5) % 360 // 45
        if angle_partition == 0:
            return PIEZO_DIRECTIONS.RIGHT
        elif angle_partition == 1:
            return PIEZO_DIRECTIONS.UP_RIGHT
        elif angle_partition == 2:
            return PIEZO_DIRECTIONS.UP
        elif angle_partition == 3:
            return PIEZO_DIRECTIONS.UP_LEFT
        elif angle_partition == 4:
            return PIEZO_DIRECTIONS.LEFT
        elif angle_partition == 5:
            return PIEZO_DIRECTIONS.DOWN_LEFT
        elif angle_partition == 6:
            return PIEZO_DIRECTIONS.DOWN
        else:
            return PIEZO_DIRECTIONS.DOWN_RIGHT
    
    def _is_target_reached(self):
        return self.target_reached
    
    def return_path(self):
        return self.path
    
    def __len__(self):
        return len(self.path)