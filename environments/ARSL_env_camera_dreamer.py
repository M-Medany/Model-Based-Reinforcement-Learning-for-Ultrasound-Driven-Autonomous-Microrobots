from gymnasium import spaces
import numpy as np
import os
from utils.actuator import Arduino, FunctionGenerator_1
from utils.tracking_CSRT import CSRT_tracker
from utils.image_postprocessing import create_binary_bitmap, plot_cluster_on_image_blue, resize_and_crop_frame, find_legal_point_target_close, detect_largest_cluster
from utils import ImageSegmentation
from .microrobot_env import BaseMicrorobotEnv, PIEZO_DIRECTIONS
from utils.path_planning_v2 import RRTStar
import cv2
import time
import csv
import pygame


class MicrorobotEnv(BaseMicrorobotEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100, "reward_types": ["sparse", "dense"]}  

    def __init__(self, config, save_path_experiment: str=None, parameters: str='rgb_on_rest_default', 
                 run: int=0, default_image_path=None, roi=None, 
                 default_mask=None, **kwargs):
        
        super().__init__(config)
        print("MicrorobotEnv initialized successfully")

        self.parameters = parameters
        self._make_folders(save_path_experiment, run)

        self.arduino=Arduino(self.config['Arduino_settings']['SERIAL_PORT_UBUNUTU'], baudrate=self.config['Arduino_settings']['BAUDRATE_ARDUINO'], config=self.config)
        self.arduino.set_piezo_by_number(0)
        print("Arduino initialized successfully")

        self.function_generator = FunctionGenerator_1(config=self.config, instrument_descriptor=self.config['Tektronix_settings']['INSTR_DESCRIPTOR'])
        print("Function generator initialized successfully")

        if roi is not None:
            self.roi_x, self.roi_y, self.roi_width, self.roi_height = roi  # Vascular Channel
        # self.roi_x, self.roi_y, self.roi_width, self.roi_height = (686, 269, 550, 550)  # Square Channel
            
        self.video_stream = cv2.VideoCapture(0)
        while True:
            _, frame = self.video_stream.read()
            
            if default_image_path is not None:
                self._overlay_img(frame, default_mask, roi)
            else:
                cv2.imshow('ROI', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if default_image_path is None and roi is None:
            _, frame = self.video_stream.read()
            self.roi_x, self.roi_y, self.roi_width, self.roi_height = cv2.selectROI("Draw ROI of whole image", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyAllWindows()
            
        ret, frame = self.video_stream.read()
        if not ret:
            raise ConnectionError("Could not read from camera")

        cv2.imwrite(f'{self.save_path_original_data}/original_frame_episode_0_state_0.png', frame)
        print("Camera initialized successfully")
        print('State 0 saved')

        initial_frame_resized, initial_frame_cropped = resize_and_crop_frame(frame, self.roi_x, self.roi_y, self.roi_width, self.roi_height)
        length_down, width_down, _ = initial_frame_resized.shape
        length_up, width_up, _ = initial_frame_cropped.shape
        assert length_up == width_up, f"Cropped image is not square, but {length_up}x{width_up}!"
        assert length_down == width_down, f"Resized image is not square, but {length_down}x{width_down}!"
        self.ratio = length_up/length_down
        self.img_size = length_up
        
        self.threshold = self.config['CSRT_Tracker_settings']['THRESHOLD']

        if default_mask is None:
            thr = self._naive_thresholding(initial_frame_cropped)
            segmented = self._seg_anything(thr, initial_frame_cropped)
            
        else:
            segmented = (np.asarray(cv2.imread(default_mask, cv2.IMREAD_GRAYSCALE), dtype=np.uint8)/255).astype(np.uint8)
            self.obstacles = pygame.image.load(default_mask) # The image to use as the obstacle environment
            self.obstacles = pygame.transform.scale(self.obstacles, (self.img_size, self.img_size))
            self._fast_obstacles = pygame.surfarray.pixels3d(self.obstacles)
        
        print("Segmentation done")
        self.segmented = segmented
        frame_cleaned = plot_cluster_on_image_blue(segmented, initial_frame_cropped, self.threshold)
        cv2.imwrite(f'/home/m4/git/DQN_for_Microrobot_control/binary_images/segmented_vascular.png', frame_cleaned)
        
        self.bound_x, self.bound_y, self.bound_width, self.bound_height = 0, 0, 0, 0 

        cv2.imwrite(f'{self.save_path_blue_data}/blue_frame_episode_0_state_0.png', frame_cleaned)
        print('Processed state 0 saved')

        self.tracker = CSRT_tracker(initial_image=frame_cleaned, params=parameters)
        self.threshold_area = self.tracker.get_bubble_area()*self.config['General_environment_settings']['MIN_SIZE_BEFORE_ABORT']

        print("Tracker initialized successfully")
        print("Agent location: ", self.get_agent_pos())
        self.current_bubble_area = self.tracker.get_bubble_area()
        print("Current Bubble area in init: ", self.current_bubble_area)
        self.initial_bubble_area = self.tracker.get_bubble_area()
        print("Initial Bubble area in init: ", self.initial_bubble_area)

        self.action_space = spaces.Discrete(self.config['Action_space_settings']['TOTAL_ACTIONS'])

        self.random_moves = self.config['General_environment_settings']['RANDOM_MOVES']
        self.size_threshold = self.config['General_environment_settings']['SIZE_THRESHOLD']
        self.tolerance_collision = self.tolerance_collision*self.ratio
        self.tolerance_target_reached = self.tolerance_target_reached*self.ratio
        self.size_threshold = self.size_threshold
        self.bubble_size = int(self.config['General_environment_settings']['BUBBLE_SIZE']*self.ratio)
        self.bbox_initial_width, self.bbox_initial_height = self.tracker.get_bbox_width_and_height()
        self.jpeg_quality = self.config['General_environment_settings']["JPEG_QUALITY"]
        
        if "Path Planning" in self.config:
            self.planner = RRTStar(segmented, self.save_path_RRT, self.config['Path Planning'])
            self.pathplanning = True

        agent_location_normalized = self.get_agent_pos() / self.img_size
        self.target_location = self.new_target(agent_location_normalized)
        print("Target location: ", self.target_location)

        # Save state 0 with target point plotted onto it
        initial_frame_with_target = np.copy(frame_cleaned)
        cv2.circle(initial_frame_with_target, tuple(self.target_location), 2, (0, 0, 255), -1)
        cv2.imwrite(f'{self.save_path_data_with_target_point}/clean_frame_episode_0_state_0.png', initial_frame_with_target)
        
        self._vpp_from_area = lambda area: np.clip(np.sqrt(area) + float(self.config['Action_space_settings']['VPP_OFFSET']), 
                                                   float(self.config['Action_space_settings']['MIN_AMPLITUDE']), 
                                                   float(self.config['Action_space_settings']['MAX_AMPLITUDE']))

    def _get_obs(self):
        ret, frame = self.video_stream.read()
        # frame = cv2.rotate(frame, cv2.ROTATE_180)
        if not ret:
            raise Exception("Could not read from camera on episode, step: " + str(self.n_elapsed_episodes), str(self.elapsed_steps))

        _, cropped = resize_and_crop_frame(frame, self.roi_x, self.roi_y, self.roi_width, self.roi_height)

        self.cleaned_image = plot_cluster_on_image_blue(self.segmented, cropped, self.threshold)
        cleaned_image_copy = np.copy(self.cleaned_image)

        # Update tracker and agent position
        self.frame_green_box, success = self.tracker.track(self.cleaned_image)
        while not success:
            self.frame_green_box, success = self._reinitialize_tracker_autonomously(self.cleaned_image)
        self.current_bubble_area = self.tracker.get_bubble_area()

        # Plot the target point onto the image -> This is what dreamer sees
        cv2.circle(cleaned_image_copy, tuple(self.target_location), self.bubble_size, (0, 0, 255), -1)

        # This is what we see externally which includes the tracking box
        cv2.circle(self.frame_green_box, tuple(self.target_location), self.bubble_size, (0, 0, 255), -1) 

        cleaned_image_copy = self.downscale_img(cleaned_image_copy)
        cv2.imshow("Annotated frame", self.upscale_img(cleaned_image_copy))
        cv2.imshow("tracking", self.upscale_img(self.frame_green_box))
        
        self.tot_steps += 1
        cv2.imwrite(f'{self.save_path_blue_data}/{self.tot_steps}_blue_frame_episode_{self.n_elapsed_episodes}_state_{self.state}.png', self.cleaned_image)
        cv2.imwrite(f'{self.save_path_tracking_data}/{self.tot_steps}_annotated_frame_episode_{self.n_elapsed_episodes}_state_{self.state}.png', self.frame_green_box)
        cv2.imwrite(f'{self.save_path_data_with_target_point}/{self.tot_steps}_clean_frame_episode_{self.n_elapsed_episodes}_state_{self.state}.png', cleaned_image_copy)
        cv2.imwrite(f'{self.save_path_original_data}/{self.tot_steps}_original_frame_episode_{self.n_elapsed_episodes}_state_{self.state}.jpeg', cropped, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        cv2.imwrite(f'{self.save_path_downsized_data}/{self.tot_steps}_downsized_frame_episode_{self.n_elapsed_episodes}_state_{self.state}.png', cleaned_image_copy)
        
        cv2.waitKey(1)
        agent_position_normalized = self.get_agent_pos() / self.img_size
        target_position_normalized = self.target_location / self.img_size

        return {"image": cleaned_image_copy.astype(np.uint8), 
                "agent_position": agent_position_normalized.astype(np.float32), 
                "target_position": target_position_normalized.astype(np.float32), 
                }
                
    # This only keeps the tracker on the cluster with default of saving these steps as False
    def _get_obs_during_reset(self, counter=0, path=None):
        ret, frame = self.video_stream.read()
        # frame = cv2.rotate(frame, cv2.ROTATE_180)
        if not ret:
            raise Exception("Could not read from camera on step: " + str(self.elapsed_steps))
        if ret:
            _, cropped = resize_and_crop_frame(frame, self.roi_x, self.roi_y, self.roi_width, self.roi_height)           
            cv2.imwrite(f'{self.save_path_original_data}/original_frame_episode_{self.n_elapsed_episodes}_step_{self.state}.jpeg', cropped, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])

        self.cleaned_image = plot_cluster_on_image_blue(self.segmented, cropped, self.threshold)
        cv2.imwrite(f'{self.save_path_blue_data}/blue_frame_episode_{self.n_elapsed_episodes}_step_{counter}_reset.png', self.cleaned_image)
        cleaned_image_copy = np.copy(self.cleaned_image)

        # Update tracker and agent position
        self.frame_green_box, success = self.tracker.track(self.cleaned_image)
        if not success:
            cv2.imshow("Annotated frame", self.upscale_img(self.frame_green_box))
            cv2.waitKey(1)
            self.frame_green_box, success = self._reinitialize_tracker_autonomously(self.cleaned_image)
            return self.cleaned_image
        # self.current_bubble_area = self.tracker.get_bubble_area()
        # self.agent_location = self.get_agent_pos()

        # During resetting this is still the old target point of the episode in which dreamer crashed
        cv2.circle(cleaned_image_copy, tuple(self.target_location), self.bubble_size, (0, 0, 255), -1)

        # This is what we see externally which includes the tracking box
        cv2.circle(self.frame_green_box, tuple(self.target_location), self.bubble_size, (0, 0, 255), -1)
        if hasattr(self, 'path'):
            path = self.path.return_path()
        if path is not None:
            for i in range(len(path) - 1):
                cv2.line(self.frame_green_box, tuple(path[i]), tuple(path[i + 1]), (0, 0, 255), 1)
            for point in path:
                cv2.circle(self.frame_green_box, tuple(point), 2, (0, 255, 255))

        cv2.imshow("Annotated frame", self.upscale_img(self.frame_green_box))
        
        self.tot_steps += 1
        cv2.imwrite(f'{self.save_path_data_with_target_point}/{self.tot_steps}_clean_frame_episode_{self.n_elapsed_episodes}_step_{counter}_reset.png', cleaned_image_copy)
        cv2.imwrite(f'{self.save_path_tracking_data}/{self.tot_steps}_annotated_frame_episode_{self.n_elapsed_episodes}_step_{counter}_reset.png', self.frame_green_box)
        cv2.waitKey(1)
        return self.cleaned_image

    def step(self, action):

        area = self.tracker.get_bubble_area()
        vpp = self._vpp_from_area(area)
        self.function_generator.set_vpp(vpp)
        # freq = self.function_generator.set_frequency_from_action(action)
        freq = np.random.uniform(2.3, 2.5)
        freq = self.function_generator.set_frequency(freq)
        piezo = self.arduino.set_piezo_from_action(action)
        if np.random.random() > 0.99:
            freq = self.function_generator.set_frequency_from_action(np.random.randint(0,16))
            print("random freq: ", freq)

        time.sleep(self.config['Action_space_settings']['STEP_DURATION'])
        return self._post_step(piezo, vpp, freq)

    def _post_step(self, piezo, vpp, freq):
        self.elapsed_steps += 1
        w, h = self.tracker.get_bbox_width_and_height()
        blue_area = self.tracker.get_bubble_area()
        self.agent_location = self.get_agent_pos()

        with open(f'{self.save_path_experiment}/experiment_data.csv', 'a', newline="") as file:
            writer = csv.writer(file)
            writer.writerow([self.elapsed_steps, self.state, self.agent_location[0], self.agent_location[1], self.target_location[0], self.target_location[1], piezo, vpp, freq, self.last_reward, self.cumulative_reward, self.terminated, self.truncated, False, w, h, self.bound_x, self.bound_y, self.bound_width, self.bound_height, blue_area])

        # Set done for state s* to False
        self.terminated = False
        self.truncated = False
        reward = 0
        self.elapsed_steps += 1
        self.state += 1 # move to s*
        observation = self._get_obs() # get state s* after taking action a

        info = self._get_info()
        self.agent_location = self.get_agent_pos() # get agent location after taking action a (at state s*)
        
        if self.check_collision(self.agent_location[0], self.agent_location[1], self.tolerance_collision):
            reward = self.reward_collision
            self.terminated = True 
            self.collision = True
            print("\033[91mCollision detected\033[0m")
        elif np.allclose(self.agent_location, self.target_location, atol=self.tolerance_target_reached): # check if goal is reached by agent.
            reward = self.reward_target_reached
            self.terminated = True
            self.target_reached = True
            print("\033[92mTarget reached\033[0m")
        else:
            distance = self._get_norm_dist(self.agent_location, self.target_location)
            reward = self.reward_function(distance)

        self.cumulative_reward += reward
        self.last_reward = reward
        self.last_piezo = piezo
        
        if self.verbose == 4:
            print('\033[94mBlue area step: ', self.current_bubble_area, 'Initial blue area: ', self.initial_bubble_area*self.size_threshold, '\033[0m')

        # while self.current_bubble_area < self.initial_bubble_area*self.size_threshold:
        #     print("\033[91mTracker lost the bubble! Reinitializing in step..\033[0m")
        #     self.arduino.set_piezo_by_number(0)
        #     #self.frame_green_box = self._reinizialize_tracker(self.cleaned_image)
        #     self.frame_green_box, success = self._reinitialize_tracker_autonomously(self.cleaned_image)
        #     observation = self._get_obs()

        done = self.terminated
        return observation, reward, done, info
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        print("Resetting environment for next episode")
        if self.verbose == 2:
            print("Number of episodes completed: ", self.n_elapsed_episodes)
            print("Elapsed steps in last episode: ", self.elapsed_steps) # This is states-1 TODO: Do we need this in our data?
            print("Cumulative_reward: ", self.cumulative_reward)
            print("Resetting environment for next episode")
        if self.verbose == 4:
            print('\033[94mBlue area reset: ', self.current_bubble_area, 'Initial blue area: ', self.initial_bubble_area, 'Initial Blue Area * threshold:', self.initial_bubble_area*self.size_threshold, '\033[0m')
        self.arduino.set_piezo_by_number(0)
        self.elapsed_steps = 0
        self.state = 0
        self.cumulative_reward = 0
        return self._post_reset()
    
    def _post_reset(self):
        c = 0
        if self.collision:
           c = self.collision_reset()
        self.collision = False

        if (self.random_moves > 0 and np.random.random() < self.random_move_probability):
            self.random_movent(c)

        agent_location_normalized = self.get_agent_pos()/ self.img_size
        if self.target_reached:
            self.target_location = self.new_target(agent_location_normalized)
            self.target_reached = False
        else:
            if not self.subepisode_sampling:
                self.target_location = self.new_target(agent_location_normalized)
            elif self.n_elapsed_episodes % self.n_subepisodes == 0:
                self.initial_bubble_area = self.current_bubble_area
                self.target_location = self.new_target(agent_location_normalized)

        observation = self._get_obs()
        self.n_elapsed_episodes += 1

        if self.verbose == 4:
            print('current bubble area:', self.current_bubble_area)
            print('initial bubble area:', self.initial_bubble_area)
            print('inital bubble area*self.size_thres:', self.initial_bubble_area*self.size_threshold)
 
        return observation

    def close(self):
        self.video_stream.release()
        cv2.destroyAllWindows()
        self.arduino.close()
        self.function_generator.turn_off()  

    def get_agent_pos(self):
        return self.tracker.get_agent_location()
    
    def _set_agent_pos(self, pos):
        raise NotImplementedError("You can't set the agent position in the real environment")

    def move_agent(self, action, fake=False):
        if fake:
            super().move_agent(action, True)
        else:
            raise NotImplementedError("You cant move the agent in the real environment")

    def _eval_pixel_collision(self, x, y):
        mask = (x < 0) | (x >= self.img_size) | (y < 0) | (y >= self.img_size)
        if np.any(mask):
            return True
        pixel_colors = self._fast_obstacles[x.flat, y.flat]
        if np.any(np.all(pixel_colors == (0, 0, 0), axis=1)):
            return True
        return False
        # valid_indices = (x >= 0) & (x < self.img_size) & (y >= 0) & (y < self.img_size)
        # return np.logical_or(~valid_indices, self.segmented[y[valid_indices], x[valid_indices]] == 0)
    
    def new_target_far(self):
        target_location = self.find_legal_point() # (self.segmented)
        while self.check_collision(*target_location, radius=4): # self.tolerance_collision 
            target_location = self.find_legal_point()
        return target_location
    
    def random_movent(self, counter=0):
        if self.pathplanning:
            self.planner.reset()
            self.planner.set_start(self.get_agent_pos())
            self.planner.set_end(self.new_target_far())
            path = self.planner.plan()
            if path is not None:
                self._follow_path(path)
            self.target_reached = True
                
        else:
            for counter_2 in range(self.random_moves):
                direction = self._sample_safe_action()
                piezo = self._get_piezo_direction(direction)
                print('\033[92mRandom move in progress, direction: ', PIEZO_DIRECTIONS.convert(piezo), '\033[0m')
                
                action = np.random.randint(1, self.config['Action_space_settings']['TOTAL_ACTIONS'])
                # vpp = self.function_generator.set_vpp_from_action(action)
                vpp = self._vpp_from_area(self.tracker.get_bubble_area())
                self.function_generator.set_vpp(vpp)
                freq = self.function_generator.set_frequency_from_action(action)
                self.arduino.set_piezo_after_collision(piezo)
                time.sleep(self.config['Action_space_settings']['RESET_STEP_DURATION'])
                self.arduino.set_piezo_by_number(0)
                self._get_obs_during_reset(counter=counter_2+counter)
                with open(f'{self.save_path_experiment}/experiment_data.csv', 'a', newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([counter_2, self.state, self.get_agent_pos()[0], self.get_agent_pos()[1], self.target_location[0], self.target_location[1], piezo, vpp, freq, None, None, self.terminated, self.truncated, True, None, None, self.bound_x, self.bound_y, self.bound_width, self.bound_height, self.current_bubble_area])

    def _reinizialize_tracker(self, observation):
        print("\n\033[91mTracker lost the bubble!\033[0m, reinitializing tracker")
        self.tracker = CSRT_tracker(initial_image=observation, params=self.parameters)
        self.current_bubble_area = self.tracker.get_bubble_area()
        print('Tracker reinitialized successfully')
        return self.tracker.track(observation)
    
    def _follow_path(self, rrt_path):
        i = 0
        next_waypoint = np.array([rrt_path[0][0], rrt_path[0][1]])
        next_waypoint_normalized = next_waypoint / self.img_size
        target_reached = False
        print(f'Next waypoint: {next_waypoint}')
        print(f'Next waypoint normalized: {next_waypoint_normalized}')
        self.target_location = np.array(rrt_path[-1])

        while not target_reached:
            self._get_obs_during_reset(counter=f'path_{i}', path=rrt_path)
            while self.current_bubble_area < self.initial_bubble_area*self.size_threshold:
                print("\033[91mTracker lost the bubble! Reinitializing in step..\033[0m")
                self.arduino.set_piezo_by_number(0)
                self.frame_green_box, success = self._reinitialize_tracker_autonomously(self.cleaned_image)

            pos_unormalized = self.tracker.get_agent_location()
            pos = pos_unormalized/self.img_size
            

            offset = next_waypoint_normalized - pos
            piezo = self.calc_action(pos, offset)
            vpp = self._vpp_from_area(self.tracker.get_bubble_area())
            frequency = np.random.uniform(1.9, 2.1)
            print(f'Calculated piezo: {piezo}, Random frequency: {frequency}, Random amplitude: {vpp}')

            self.function_generator.set_frequency(frequency)
            self.function_generator.set_vpp(vpp)
            self.arduino.set_piezo_by_number(piezo)

            time.sleep(self.config['Action_space_settings']['RESET_STEP_DURATION'])
            self.arduino.set_piezo_by_number(0)        

            if np.allclose(pos_unormalized, next_waypoint, atol=self.tolerance_target_reached):
                print("Waypoint reached")
                rrt_path.pop(0)
                if len(rrt_path) == 0:
                    print("No more waypoints")
                    print("Target reached")
                    target_reached = True
                else:
                    next_waypoint = np.array([rrt_path[0][0], rrt_path[0][1]])
                    next_waypoint_normalized = next_waypoint / self.img_size
                    print(f'Next waypoint: {next_waypoint_normalized}')
            elif i > 200:
                print("Could not reach waypoint")
                target_reached = True
                break
            i += 1
    
    def _overlay_img(self, frame, img, pos):
        height, width = frame.shape[:2]

        # Calculate the coordinates of the "+" sign
        start_vertical = (750 + 642 // 2, 0)
        end_vertical = (750 + 642 // 2, height)
        start_horizontal = (0, height // 2)
        end_horizontal = (width, height // 2)

        # Draw the vertical line
        # cv2.line(frame, start_vertical, end_vertical, (0, 255, 0), thickness=2)

        # # Draw the horizontal line
        # cv2.line(frame, start_horizontal, end_horizontal, (0, 255, 0), thickness=2)
        
        _, frame = resize_and_crop_frame(frame, *pos)

        # Load the overlay image
        overlay_image = cv2.imread(img)

        # Make sure the overlay image is the same size as the main image
        overlay_image = cv2.resize(overlay_image, (frame.shape[1], frame.shape[0]))

        # # Calculate the weighted sum of the images
        alpha = 0.2 # Transparency factor.
        frame = cv2.addWeighted(overlay_image, alpha, frame, 1 - alpha, 0)

        cv2.imshow('buffer', frame)

    def calc_action(self, pos, offset):
        """
        Calculate optimal piezo to actuate
        :param pos0:    Swarm position
        :param offset:  Offset to target in pixel difference np.array([x, y])
        :return:        integer in [1, 2, 3, 4] with len() = number of piezos
        """
        
        if abs(offset[0]) >= abs(offset[1]):
            if offset[0] > 0:
                return PIEZO_DIRECTIONS.RIGHT
            else:
                return PIEZO_DIRECTIONS.LEFT
        else:
            if offset[1] > 0:
                return PIEZO_DIRECTIONS.DOWN
            else:
                return PIEZO_DIRECTIONS.UP
    
    def _reinitialize_tracker_autonomously(self, image, location=None):
        print("\n\033[91mTracker lost the bubble!\033[0m, reinitializing tracker autonomously")
        counter = 1
        while True:
            # self.arduino.set_piezo_by_number(PIEZO_DIRECTIONS.OFF)
            x, y, w, h = detect_largest_cluster(image)
            if counter%10 == 0 or (x == 0 and y == 0 and w == 0 and h == 0):
                counter += 1
                val = print(f"Tracker lost the bubble! Current area: {self.current_bubble_area}, Initial area: {self.initial_bubble_area}, Threshold area: {self.threshold_area}. Continue? [y|n] ")
                self.arduino.set_piezo_by_number(PIEZO_DIRECTIONS.RIGHT)
                # location = self.agent_location
                # print(f"Agent location: {location}")
                # if location is not None and location[0] > self.img_size//2:
                #     self.arduino.set_piezo_by_number(PIEZO_DIRECTIONS.LEFT)
                #     print("Going left")
                # elif location is not None and location[0] < self.img_size//2:
                #     self.arduino.set_piezo_by_number(PIEZO_DIRECTIONS.RIGHT)
                #     print("Going right")
                # else:
                #     continue
                # self.function_generator.set_vpp(9)
                # freq = np.random.uniform(2.4, 2.9)
                # self.function_generator.set_frequency(freq)
                # print("Using frequency: ", freq)
                # time.sleep(self.config['Action_space_settings']['RESET_STEP_DURATION']*2)
            else:
                self.tracker = CSRT_tracker(initial_image=image, params=self.parameters, autonomous=True, x=x, y=y, w=w, h=h, box_padding=self.config['CSRT_Tracker_settings']['BOX_PADDING'], img_size=self.img_size)
                self.current_bubble_area = self.tracker.get_bubble_area()
                if self.current_bubble_area < self.threshold_area: # Do this to avoid continuing even though the bubble is extremely small or gone
                    counter += 1
                else:
                    print('Tracker reinitialized successfully! \n')
                    return self.tracker.track(image)
            image = self._get_obs_during_reset(counter=f'reinitialization_{counter}')
        # raise ValueError("Bubble area is too small, aborting")
    
    def _get_norm_dist(self, agent_location, target_location):
        aget_loc_norm = agent_location / self.img_size
        target_loc_norm = target_location / self.img_size
        return np.linalg.norm(aget_loc_norm - target_loc_norm, ord=2)
    
    def collision_reset(self, radius=None):
        if self.verbose == 4:
            print('\033[94mBlue area reset: ', self.current_bubble_area, 'Initial blue area: ', self.initial_bubble_area*self.size_threshold, '\033[0m')
        for counter in range(self.config['Action_space_settings']['SAFE_STEPS']):
            while self.current_bubble_area < self.initial_bubble_area*self.size_threshold:
                print("\033[91mTracker lost the bubble! Reinitializing in reset..\033[0m")
                #self.arduino.set_piezo_by_number(0) # Turn off piezo before reinitializing the tracker
                #self.frame_green_box = self._reinizialize_tracker(self.cleaned_image)
                self.frame_green_box, success = self._reinitialize_tracker_autonomously(self.cleaned_image)
                self._get_obs_during_reset(counter=counter, path=self.planner.path)
                if self.verbose == 2:
                    print("Current bubble area: ", self.current_bubble_area)
                    print("Initial bubble area: ", self.initial_bubble_area)
            self.arduino.set_piezo_by_number(0)
            loc = (int(self.get_agent_pos()[0]), int(self.get_agent_pos()[1]))
            safe_piezo = self._get_safe_direction_from_img(loc, radius)
            
            # vpp = self.function_generator.set_vpp_from_action(action)
            vpp = self._vpp_from_area(self.tracker.get_bubble_area()) + self.config['Action_space_settings']['VPP_OFFSET_RESET']
            self.function_generator.set_vpp(vpp)
            freq = np.random.uniform(self.config['Action_space_settings']['MIN_FREQUENCY'], self.config['Action_space_settings']['MAX_FREQUENCY'])
            freq = self.function_generator.set_frequency(freq)
            self.arduino.set_piezo_after_collision(safe_piezo)
            time.sleep(self.config['Action_space_settings']['RESET_STEP_DURATION'])
            self.arduino.set_piezo_by_number(0)
            self._get_obs_during_reset(counter=counter)
            with open(f'{self.save_path_experiment}/experiment_data.csv', 'a', newline="") as file:
                writer = csv.writer(file)
                self.agent_location = self.get_agent_pos()
                writer.writerow([counter, self.state, self.agent_location[0], self.agent_location[1], self.target_location[0], self.target_location[1], safe_piezo, vpp, freq, None, None, self.terminated, self.truncated, True, None, None, self.bound_x, self.bound_y, self.bound_width, self.bound_height, self.current_bubble_area])
            if self.verbose == 2:
                print('Safe step: ', counter+1) # account for range() indexing and better readability
                print("Vpp: ", vpp)
                print("Frequency: ", freq)
            print(f"Going direction: {PIEZO_DIRECTIONS.convert(safe_piezo)}")
            # agent_location_during_reset = self.get_agent_pos()
            # with open(f'{self.save_path_experiment}/experiment_data.csv', 'a', newline="") as file:
            #     writer = csv.writer(file)
            #     writer.writerow([self.elapsed_steps, self.state, agent_location_during_reset[0], agent_location_during_reset[1], None, None, safe_piezo, vpp, freq, 0, 0, True, False, True])
        print('Collision reset done')
        return counter
    
    def _make_folders(self, save_path_experiment, run):

        current_time = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(save_path_experiment, exist_ok=True)
        self.save_path_experiment = f'{save_path_experiment}/experiment_piezo_fixed_{current_time}_run_{run}'
        self.save_path_original_data = f'{self.save_path_experiment}/original_data'
        self.save_path_blue_data = f'{self.save_path_experiment}/blue_data'
        self.save_path_data_with_target_point = f'{self.save_path_experiment}/data_with_target_point'
        self.save_path_tracking_data = f'{self.save_path_experiment}/tracking_data'
        self.save_path_RRT = f'{self.save_path_experiment}/RRT'
        self.save_path_downsized_data = f'{self.save_path_experiment}/downsized_data'
        os.mkdir(self.save_path_experiment)
        os.mkdir(self.save_path_original_data)
        os.mkdir(self.save_path_blue_data)
        os.mkdir(self.save_path_data_with_target_point)
        os.mkdir(self.save_path_tracking_data)
        os.mkdir(self.save_path_RRT)
        os.mkdir(self.save_path_downsized_data)
        # create csv files for saving data
        with open(f'{self.save_path_experiment}/experiment_data.csv', 'w', newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["elapsed_steps", "state", "agent_location_x", "agent_location_y", "target_location_x", "target_location_y", "piezo", "vpp", "frequency", "reward", "cumulative_reward", "terminated", "truncated", "collision_reset_step", 'bbox_width', 'bbox_height', 'x_bound', 'y_bound', 'width_bound', 'height_bound', 'blue_area', 'time_since_start'])
        with open(f"{self.save_path_experiment}/plots.csv", 'w', newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["step", "reward", "distance_reward", "substep_reward", "reward_target_reached", "collision_rewar"])
        with open(f"{self.save_path_experiment}/episode_plots.csv", 'w', newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["step", "reward", "collision", "target_reached", "truncated"])
        
        
    def _naive_thresholding(self, initial_frame_cropped):
        from utils.preprocessing import ThresholdMaskGenerator
        thr = ThresholdMaskGenerator(initial_frame_cropped)
        cv2.destroyAllWindows()
        threshold = 2
        foreground = [(57, 328), (61, 371), (138, 420), (179, 438), (246, 465), (284, 484), (307, 486), (340, 482), (367, 481), (399, 460), (440, 433), (471, 415), (505, 398), (542, 382), (568, 350), (591, 308), (578, 275), (518, 276), (467, 275), (249, 261), (287, 253), (334, 246), (363, 247), (168, 285), (142, 315), (160, 331), (174, 340), (217, 358), (257, 368), (292, 371), (336, 372), (367, 366), (577, 248), (532, 210), (473, 193), (415, 169), (333, 138), (295, 129), (239, 146), (196, 187), (159, 218), (97, 231), (65, 274)]
        background = [(264, 328), (325, 321), (326, 307), (387, 317), (330, 195), (296, 198), (266, 210), (119, 264), (500, 245), (499, 350), (347, 415), (318, 428), (286, 430), (132, 377), (60, 488), (135, 533), (232, 558), (403, 564), (482, 547), (554, 498), (583, 457), (568, 159), (485, 102), (390, 63), (264, 45), (155, 62), (87, 119)]
        good_enough = False
        while not good_enough:
            img_thres = thr.threshold(threshold)
            cv2.imshow("Thresholded Image", img_thres)
            cv2.waitKey(100)
            threshold = input("New threshold: (leave empty to continue)  ")
            if threshold == "q" or threshold == "":
                break
            threshold = int(threshold)
        cv2.destroyAllWindows()
        
        print("Draw contours")
        thr.draw_black_lines()
        
        print("Select the background poitns")
        thr.set_backround_points(*background)
        
        print("Select the foreground points")
        thr.set_foreground_points(*foreground)
        
        thr.color_rectangles("w")
        cv2.destroyAllWindows()
        return thr

    def _seg_anything(self, thr, initial_frame_cropped):
        try:
            mask_in = np.load('/home/m4/git/DQN_for_Microrobot_control/models/mask_in.npy')
        except(FileNotFoundError):
            mask_in = None
    
        segmentation = ImageSegmentation(image=thr.get_img_mask(), sam_checkpoint=self.config["sam_config"]["sam_checkpoint"])
        good_enough = False
        while not good_enough:
            segmented, mask_in = create_binary_bitmap(thr.get_img_mask(), segmentation,
                                                        foreground_points=thr.get_foreground_points(), 
                                                        background_points=thr.get_background_points(),
                                                        mask_in=mask_in, kernel_size=(3,3))
            # segmented = cv2.invert(segmented)[1]
            frame_cleaned = plot_cluster_on_image_blue(segmented, initial_frame_cropped, self.threshold)
            cv2.imshow("Blue Image", frame_cleaned)
            cv2.waitKey(1)
            answer = input("Like it? [y|(N)] ")
            if answer.lower() == "y":
                good_enough = True
            thr.reset_points()
        np.save(r'/home/m4/git/DQN_for_Microrobot_control/models/mask_in.npy', mask_in)