import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
import os
from utils.actuator import Arduino, FunctionGenerator_1
from utils.tracking_CSRT import CSRT_tracker
from utils.image_postprocessing import create_binary_bitmap, plot_cluster_on_image_blue, resize_and_crop_frame, find_legal_point_target_close
import cv2
import time
import os
import csv
import json
from environments.costum_wrappers.NormalizeObs import NormalizedObs
from microrobot_env import BaseMicrorobotEnv

# TODO: Tracker initial image
class MicrorobotEnv(BaseMicrorobotEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100, "reward_types": ["sparse", "dense"]}  

    def __init__(self, timeout=100, save_path_experiment: str=None, threshold=90, parameters: str='rgb_on_rest_default', run: int=0, subepisode_sampling: bool=False): # TODO: Lets make a decision on the threshold. before it was 80 and maybe too conservative

        # Read yaml 
        with open(f'/home/m4/git/DQN_for_Microrobot_control/scripts/config.yaml', 'r') as yaml_file:
            self.config = yaml.safe_load(yaml_file)

        # TODO: Include this CSRT tracker configuration into the config.yml file
        with open(f'/home/m4/Documents/Tracker_Robustness_Data_Acquisition/CSRT_parameters/{parameters}.json', 'r') as f:
            params = json.load(f)
        print(params)

        current_time = time.strftime("%Y%m%d-%H%M%S")        
        self.threshold = threshold
        self.parameters = parameters
        self.verbose = self.config['General_environment_settings']['VERBOSE']

        self.save_path_experiment = f'{save_path_experiment}/experiment_piezo_fixed_{current_time}_run_{run}'
        os.mkdir(self.save_path_experiment)
        self.save_path_original_data = f'{self.save_path_experiment}/original_data'
        os.mkdir(self.save_path_original_data)
        self.save_path_blue_data = f'{self.save_path_experiment}/blue_data'
        os.mkdir(self.save_path_blue_data)
        self.save_path_data_with_target_point = f'{self.save_path_experiment}/data_with_target_point'
        os.mkdir(self.save_path_data_with_target_point)
        self.save_path_tracking_data = f'{self.save_path_experiment}/tracking_data'
        os.mkdir(self.save_path_tracking_data)
        self.save_path_RRT = f'{self.save_path_experiment}/RRT'
        os.mkdir(self.save_path_RRT)
        self.save_path_downsized_data = f'{self.save_path_experiment}/downsized_data'
        os.mkdir(self.save_path_downsized_data)
        # create csv files for saving data
        with open(f'{self.save_path_experiment}/experiment_data.csv', 'w', newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["elapsed_steps", "state", "agent_location_x", "agent_location_y", "target_location_x", "target_location_y", "piezo", "vpp", "frequency", "reward", "cumulative_reward", "terminated", "truncated", "collision_reset_step", 'bbox_width', 'bbox_height', 'x_bound', 'y_bound', 'width_bound', 'height_bound', 'blue_area'])

        self.arduino=Arduino(self.config['Arduino_settings']['SERIAL_PORT_UBUNUTU'], baudrate=self.config['Arduino_settings']['BAUDRATE_ARDUINO'], config=self.config)
        self.arduino.set_piezo_by_number(0)
        print("Arduino initialized successfully")

        self.function_generator = FunctionGenerator_1(config=self.config, instrument_descriptor=self.config['Tektronix_settings']['INSTR_DESCRIPTOR'])
        print("Function generator initialized successfully")

        # Initialize the camera 
        self.video_stream = cv2.VideoCapture(0)

        while True:
            tre, frame = self.video_stream.read()
            cv2.imshow('buffer', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Read the ROI image and define the ROI if neccessary 
        tre, frame = self.video_stream.read()
        if tre:
            self.roi_x, self.roi_y, self.roi_width, self.roi_height = cv2.selectROI("Draw ROI of whole image", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyAllWindows()
        
        # Read initial frame
        ret, frame = self.video_stream.read()
        if not ret:
            raise ConnectionError("Could not read from camera")
        if ret:
            cv2.imwrite(f'{self.save_path_original_data}/original_frame_episode_0_state_0.png', frame)
            print("Camera initialized successfully")
            print('State 0 saved')

        initial_frame, _ = resize_and_crop_frame(frame, self.roi_x, self.roi_y, self.roi_width, self.roi_height)
        self.initial_image = initial_frame
        self.flength, self.fwidth, _ = initial_frame.shape
        
        good_enough = False
        mask_in = np.load('/home/m4/git/DQN_for_Microrobot_control/models/mask_in.npy')
        while not good_enough:
            processed, segmented, mask_in = create_binary_bitmap(self.initial_image, manual=True, mask_in=mask_in)
            frame_cleaned = plot_cluster_on_image_blue(segmented, self.initial_image, self.threshold)
            cv2.imshow("Blue Image", frame_cleaned)
            cv2.waitKey(0)
            answer = input("Good Enough???? [y|(N)] ")
            if answer.lower() == "y":
                good_enough = True
        print("Segmentation done")
        
        self.bound_x, self.bound_y, self.bound_width, self.bound_height = cv2.selectROI("Fit rectangle to boundaries of channel", initial_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        

        np.save(r'/home/m4/git/DQN_for_Microrobot_control/models/mask_in.npy', mask_in)

        # Plot the cluster on the segmented image
        self.cluster_image_initial = frame_cleaned
        self.channel_segmented_image = segmented
        cv2.imshow("Cluster image", self.cluster_image_initial)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        # Save the processed image
        cv2.imwrite(f'{self.save_path_blue_data}/blue_frame_episode_0_state_0.png', frame_cleaned)
        print('Processed state 0 saved')

        # TODO: We do not save the initial image with the tracking window on it
        self.tracker = CSRT_tracker(initial_image=frame_cleaned, params=parameters)
        self.agent_location = self.tracker.get_agent_location()
        self.agent_area = 0
        print("Tracker initialized successfully")
        print("Agent location: ", self.agent_location)
        #self.initial_bubble_area = self.tracker.get_bubble_area()
        self.current_bubble_area = 0
        self.initial_bubble_area = 0

        agent_location_normalized = self.agent_location / np.array([self.fwidth, self.flength])
        self.target_location = find_legal_point_target_close(self.channel_segmented_image, agent_location_normalized) #needs a bitmap
        print("Target location: ", self.target_location)

        # Save state 0 with target point plotted onto it
        initial_frame_with_target = np.copy(self.cluster_image_initial)
        cv2.circle(initial_frame_with_target, tuple(self.target_location), 5, (0, 0, 255), -1)
        cv2.imwrite(f'{self.save_path_data_with_target_point}/clean_frame_episode_0_state_0.png', initial_frame_with_target)
        print('State 0 with target point saved')

        # TODO: Do we want to save the image with the final result
        # Initialize the RRT* planner TODO: Account for the coordinate inversion in bitmap -> RGB image
        # self.rrt_star = RRTStar(self.channel_segmented_image, self.agent_location, self.target_location, self.save_path_RRT, self.config)
        # self.rrt_star.plan() # The path is saved in RRTStar.path
        # # TODO: Decide how this is represented in the environment
        # # TODO: Decide how we want to see the training live -> render mode or something else?
        # print("RRT* planner initialized successfully")
        # if self.rrt_star.path is not None:
        #     print("Path: ", self.rrt_star.path)
        # if self.rrt_star.path is None:
        #     print("No path found")
        #     raise Exception("Error in RRT* planner")

        # Initialize observation space -> this is a RGB image TODO: Does dreamer need RGB or BGR?
        #self.observation_space = spaces.Box(low=0.0, high=255.0, shape=(self.config['Layout_settings']['IMG_DOWNSIZED_SIZE'], self.config['Layout_settings']['IMG_DOWNSIZED_SIZE'], 3), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.config['Layout_settings']['IMG_DOWNSIZED_SIZE'], self.config['Layout_settings']['IMG_DOWNSIZED_SIZE'], 3), dtype=np.float32)
        # Prune this action space to help convergence
        self.action_space = spaces.Discrete(self.config['Action_space_settings']['TOTAL_ACTIONS'])

        self.tolerance_target_reached = self.config['General_environment_settings']['TARGET_REACHED_TOLERANCE'] # This decides how close the agent needs to be to the target to be considered reached
        self.tolerance_collision = self.config['General_environment_settings']['COLLISION_TOLERANCE'] # How close do we count it as a collision
        self.size_threshold = self.config['General_environment_settings']['SIZE_THRESHOLD'] # This decides how much the bubble can shrink before we consider it lost
        self.elapsed_steps = 0
        self.last_piezo = 0
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
        self.n_subepisodes = self.config['General_environment_settings']['N_SUBEPISODES']
        self.target_reached = False

        self.reward_target_reached = self.config['General_environment_settings']["Reward_Shape"]['REWARD_TARGET_REACHED']
        self.reward_collision = self.config['General_environment_settings']["Reward_Shape"]['REWARD_COLLISION']
        self.reward_step = self.config['General_environment_settings']["Reward_Shape"]['REWARD_STEP']
        self.reward_function = self.config['General_environment_settings']["Reward_Shape"]['REWARD_FUNCTION']
        if self.reward_function == 'linerar':
            self.reward_function = lambda x: self.reward_step*x
        elif self.reward_function == 'quadratic':
            self.reward_function = lambda x: self.reward_step*x**2
        elif self.reward_function == 'log':
            self.reward_function = lambda x: self.reward_step*np.log(x)
        elif self.reward_function == 'inverse':
            self.reward_function = lambda x: self.reward_step*(1/(x+0.1))
        elif self.reward_function == 'inverse_squared':
            self.reward_function = lambda x: self.reward_step*(1/(x**2+0.1))
        else:
            raise Exception("Reward function not implemented")

    def _get_obs(self):
        ret, frame = self.video_stream.read()
        if not ret:
            raise Exception("Could not read from camera on episode, step: " + str(self.n_elapsed_episodes), str(self.elapsed_steps))
        if ret:
            cv2.imwrite(f'{self.save_path_original_data}/original_frame_episode_{self.n_elapsed_episodes}_state_{self.state}.png', frame)
            resized_frame, _ = resize_and_crop_frame(frame, self.roi_x, self.roi_y, self.roi_width, self.roi_height)
            self.current_image = resized_frame

        self.cleaned_image = plot_cluster_on_image_blue(self.channel_segmented_image, self.current_image, self.threshold)
        cv2.imwrite(f'{self.save_path_blue_data}/blue_frame_episode_{self.n_elapsed_episodes}_state_{self.state}.png', self.cleaned_image)
        cleaned_image_copy = np.copy(self.cleaned_image)

        # Update tracker and agent position
        self.frame_green_box = self.tracker.track(self.cleaned_image)
        self.current_bubble_area = self.tracker.get_bubble_area()

        # Plot the target point onto the image -> This is what dreamer sees
        cv2.circle(cleaned_image_copy, tuple(self.target_location), 5, (0, 0, 255), -1)
        cv2.imwrite(f'{self.save_path_data_with_target_point}/clean_frame_episode_{self.n_elapsed_episodes}_state_{self.state}.png', cleaned_image_copy)

        # This is what we see externally which includes the tracking box
        cv2.circle(self.frame_green_box, tuple(self.target_location), 5, (0, 0, 255), -1) 
        cv2.imwrite(f'{self.save_path_tracking_data}/annotated_frame_episode_{self.n_elapsed_episodes}_state_{self.state}.png', self.frame_green_box)
        cv2.imshow("Annotated frame", self.upscale_img(self.frame_green_box))
        cv2.waitKey(10)

        cleaned_image_copy = self.downscale_img(cleaned_image_copy)
        cv2.imwrite(f'{self.save_path_downsized_data}/downsized_frame_episode_{self.n_elapsed_episodes}_state_{self.state}.png', cleaned_image_copy)
        #return cleaned_image_copy.astype(np.float32)
        img2 = (cleaned_image_copy.astype(np.float32) / 128.0) - 1.0
        return {"image": img2}

    def _get_info(self):
        return {"distance": np.linalg.norm(self.agent_location - self.target_location, ord=2),
                "agent_position": "TODO",}
                
    # This only keeps the tracker on the cluster with default of saving these steps as False
    def _get_obs_during_reset(self, counter=0):
        ret, frame = self.video_stream.read()
        if not ret:
            raise Exception("Could not read from camera on step: " + str(self.elapsed_steps))
        if ret:
            cv2.imwrite(f'{self.save_path_original_data}/original_frame_reset_{self.n_elapsed_episodes}_step_{counter}.png', frame)
            resized_frame, _ = resize_and_crop_frame(frame, self.roi_x, self.roi_y, self.roi_width, self.roi_height)
            self.current_image = resized_frame

        self.cleaned_image = plot_cluster_on_image_blue(self.channel_segmented_image, self.current_image, self.threshold)
        cv2.imwrite(f'{self.save_path_blue_data}/blue_frame_reset_{self.n_elapsed_episodes}_step_{counter}.png', self.cleaned_image)
        cleaned_image_copy = np.copy(self.cleaned_image)

        # Update tracker and agent position
        self.frame_green_box = self.tracker.track(self.cleaned_image)
        self.current_bubble_area = self.tracker.get_bubble_area()
        self.agent_location = self.tracker.get_agent_location()

        # During resetting this is still the old target point of the episode in which dreamer crashed
        cv2.circle(cleaned_image_copy, tuple(self.target_location), 5, (0, 0, 255), -1)
        cv2.imwrite(f'{self.save_path_data_with_target_point}/clean_frame_reset_{self.n_elapsed_episodes}_step_{counter}.png', cleaned_image_copy)

        # This is what we see externally which includes the tracking box
        cv2.circle(self.frame_green_box, tuple(self.target_location), 5, (0, 0, 255), -1) 
        cv2.imshow("Annotated frame", self.upscale_img(self.frame_green_box))
        cv2.imwrite(f'{self.save_path_tracking_data}/annotated_frame_reset_episode_{self.n_elapsed_episodes}_step_{counter}.png', self.frame_green_box)
        cv2.waitKey(10)

    def check_collision(self, player_location_x: int, player_location_y: int, bitmap: np.ndarray):
        # Check around the player location for a black pixel within a range of 5 pixels
        for i in range(player_location_x - self.tolerance_collision, player_location_x + self.tolerance_collision):
            for j in range(player_location_y - self.tolerance_collision, player_location_y + self.tolerance_collision):
                if bitmap[j, i] == 0:  # Assuming black pixels have a value of 0
                    return True
        return False
    
    def check_collisions(self, player_location_x: int, player_location_y: int, bitmap: np.ndarray):
        collisions = []
        for i in range(player_location_x - self.tolerance_collision, player_location_x + self.tolerance_collision):
            for j in range(player_location_y - self.tolerance_collision, player_location_y + self.tolerance_collision):
                if bitmap[j, i] == 0:  # Assuming black pixels have a value of 0
                    if i > player_location_x:
                        collisions.append(PIEZO_DIRECTIONS.RIGHT)
                    elif i < player_location_x:
                        collisions.append(PIEZO_DIRECTIONS.LEFT)
                    if j > player_location_y:
                        collisions.append(PIEZO_DIRECTIONS.DOWN)
                    elif j < player_location_y:
                        collisions.append(PIEZO_DIRECTIONS.UP)
        return collisions 

    # TODO: Check the reward structure which made dreamer converge before and implement it here
    def step(self, action):
        # Turn off piezo before taking another action
        self.arduino.set_piezo_by_number(0)

        # Apply action to the function generator and arduino
        vpp = self.function_generator.set_vpp_from_action(action)
        #print("Vpp: ", vpp)
        freq = self.function_generator.set_frequency_from_action(action)
        #print("Frequency: ", freq)
        piezo = self.arduino.set_piezo_from_action(action)

        # make the agent take a step and stop the movement afterwards
        time.sleep(self.config['Action_space_settings']['STEP_DURATION']) 
        self.arduino.set_piezo_by_number(0)

        w, h = self.tracker.get_bbox_width_and_height()
        blue_area = self.tracker.get_bubble_area()

        # save state s and action a
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
        while self.current_bubble_area < self.initial_bubble_area*self.size_threshold:
            self.frame_green_box = self._reinizialize_tracker(self.cleaned_image)
            observation = self._get_obs()

        info = self._get_info()
        self.agent_location = self.tracker.get_agent_location() # get agent location after taking action a (at state s*)
        
        if self.check_collision(self.agent_location[0], self.agent_location[1], self.channel_segmented_image):
            reward = self.reward_collision
            self.terminated = True 
            self.collision = True
        elif np.allclose(self.agent_location, self.target_location, atol=self.tolerance_target_reached): # check if goal is reached by agent.
            reward = self.reward_target_reached
            self.terminated = True
            self.target_reached = True       
        else:
            distance = self._get_norm_dist(self.agent_location, self.target_location)
            reward = self.reward_function(distance)

        self.cumulative_reward += reward
        self.last_reward = reward
        self.last_piezo = piezo

        if self.current_bubble_area < self.initial_bubble_area*self.size_threshold:
            self.terminated = True
            print("\033[91mTracker lost the bubble!\033[0m")

        #done = self.truncated or self.terminated

        return observation, reward, self.terminated, self.truncated, info

    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        if self.verbose == 2:
            print("Number of episodes completed: ", self.n_elapsed_episodes)
            print("Elapsed steps in last episode: ", self.elapsed_steps) # This is states-1 TODO: Do we need this in our data?
            print("Cumulative_reward: ", self.cumulative_reward)
            print("Resetting environment for next episode")
        self.n_elapsed_episodes += 1
        self.arduino.set_piezo_by_number(0)
        self.elapsed_steps = 0 # TODO: We could keep this as a counter for the number of episodes, total steps, etc.
        self.state = 0
        self.cumulative_reward = 0

        # Find the opposite direction of the last piezo direction
        if self.collision:
            #self.collision_reset_steps = True
            safe_piezo = self._get_safe_direction(self.last_piezo)
            if self.verbose == 2:
                print('\nCollision reset in progress')
                print('Last piezo: ', self.last_piezo)
                print('Safe piezo: ', safe_piezo)
            for counter in range(self.config['Action_space_settings']['SAFE_STEPS']):
                while self.current_bubble_area < self.initial_bubble_area*self.size_threshold:
                    self.frame_green_box = self._reinizialize_tracker(self.cleaned_image)
                    self._get_obs_during_reset(counter=counter)
                    if self.verbose == 2:
                        print("Current bubble area: ", self.current_bubble_area)
                        print("Initial bubble area: ", self.initial_bubble_area)
                self.arduino.set_piezo_by_number(0)
                safe_piezo = self._get_safe_direction_from_img(self.channel_segmented_image, self.agent_location)
                # choose action as random number between 1 and all actions in config
                action = np.random.randint(1, self.config['Action_space_settings']['TOTAL_ACTIONS'])
                
                vpp = self.function_generator.set_vpp_from_action(action)
                freq = self.function_generator.set_frequency_from_action(action)
                self.arduino.set_piezo_after_collision(safe_piezo)
                time.sleep(self.config['Action_space_settings']['STEP_DURATION'])
                self.arduino.set_piezo_by_number(0)
                self._get_obs_during_reset(counter=counter)
                if self.verbose == 2:
                    print('Safe step: ', counter+1) # account for range() indexing and better readability
                    print("Vpp: ", vpp)
                    print("Frequency: ", freq)
                # agent_location_during_reset = self.tracker.get_agent_location()
                # with open(f'{self.save_path_experiment}/experiment_data.csv', 'a', newline="") as file:
                #     writer = csv.writer(file)
                #     writer.writerow([self.elapsed_steps, self.state, agent_location_during_reset[0], agent_location_during_reset[1], None, None, safe_piezo, vpp, freq, 0, 0, True, False, True])
            print('Collision reset done')

        self.collision = False

        # Set target location to the RRT* location
        agent_location_normalized = self.agent_location / np.array([self.fwidth, self.flength])
        if self.target_reached:
            self.target_location = find_legal_point_target_close(self.channel_segmented_image, agent_location_normalized)
            self.target_reached = False
        else:
            if self.subepisode_sampling:
                if self.n_elapsed_episodes % self.n_subepisodes == 0:
                    self.target_location = find_legal_point_target_close(self.channel_segmented_image, agent_location_normalized)
            else:
                self.target_location = find_legal_point_target_close(self.channel_segmented_image, agent_location_normalized)

        observation = self._get_obs()
        info = self._get_info()

        if self.first_obs:
            self.initial_bubble_area = self.current_bubble_area
            self.first_obs = False

        if self.verbose == 2:
            print('current bubble area:', self.current_bubble_area)
            print('initial bubble area:', self.initial_bubble_area)
            print('inital bubble area*self.size_thres:', self.initial_bubble_area*self.size_threshold)
 
        return observation, info


    def close(self):
        self.video_stream.release()
        cv2.destroyAllWindows()
        self.arduino.close()
        self.function_generator.turn_off()
    

    def _get_safe_direction(self, last_piezo):
        safe_direction = 0
        if last_piezo == 1:
            safe_direction = 4
        elif last_piezo == 2:
            safe_direction = 3
        elif last_piezo == 3:
            safe_direction = 2
        elif last_piezo == 4:
            safe_direction = 1
        return safe_direction
    
    def _get_safe_direction_from_img(self, cleaned_img, cluster_center):
        directions = self.check_collisions(cluster_center[0], cluster_center[1], cleaned_img)
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

        print('\033[91mSafe direction: {}\033[0m'.format(PIEZO_DIRECTIONS.convert(safe_direction)))
        if self.verbose == 2:
            print('\nSafe direction hor: ', PIEZO_DIRECTIONS.convert(safe_direction_hor))
            print('Safe direction ver: ', PIEZO_DIRECTIONS.convert(safe_direction_ver))
            print('right_tot: ', right_tot)
            print('left_tot: ', left_tot)
            print('up_tot: ', up_tot)
            print('down_tot: ', down_tot)
        return safe_direction

    def downscale_img(self, img):
        return cv2.resize(img, (self.config['Layout_settings']['IMG_DOWNSIZED_SIZE'], self.config['Layout_settings']['IMG_DOWNSIZED_SIZE']), interpolation=cv2.INTER_AREA)
    
    def rescale_img(self, img):
        return cv2.resize(img, (self.config['Layout_settings']['IMG_SIZE'], self.config['Layout_settings']['IMG_SIZE']), interpolation=cv2.INTER_AREA)
    
    def upscale_img(self, img):
        return cv2.resize(img, (self.config['Layout_settings']['IMG_UPSCALED_SIZE'], self.config['Layout_settings']['IMG_UPSCALED_SIZE']), interpolation=cv2.INTER_AREA)
    
    def _reinizialize_tracker(self, observation):
        print("\n\033[91mTracker lost the bubble!\033[0m, reinitializing tracker")
        self.tracker = CSRT_tracker(initial_image=observation, params=self.parameters)
        print('Tracker reinitialized successfully')
        return self.tracker.track(observation)
    
    def _get_norm_dist(self, agent_location, target_location):
        aget_loc_norm = agent_location / np.array([self.fwidth, self.flength])
        target_loc_norm = target_location / np.array([self.fwidth, self.flength])
        return np.linalg.norm(aget_loc_norm - target_loc_norm, ord=2)
    
class PIEZO_DIRECTIONS:
    RIGHT = 1
    UP = 2
    DOWN = 3
    LEFT = 4
    OFF = 0
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
        

class MicrorobotEnvRay(MicrorobotEnv):
    def __init__(self, env_config):
        super().__init__(env_config['timeout'], env_config['save_path_experiment'], env_config['threshold'], env_config['parameters'], env_config['run'], env_config['subepisode_sampling'])



class MicrorobotEnvRayWrapped(MicrorobotEnv):

    # UserWarning: WARN: It seems a Box observation space is an image but the `dtype` is not `np.uint8`, actual type: float32. 
    # If the Box observation space is not an image, we recommend flattening the observation to have only a 1D vector.
    # UserWarning: WARN: It seems a Box observation space is an image but the lower and upper bounds are not [0, 255]. 
    # Actual lower bound: -1.0, upper bound: 1.0. Generally, CNN policies assume observations are within that range, so you may encounter an issue if the observation values are not. 

    def __init__(self, env_config):
        self.env = MicrorobotEnv(env_config['timeout'], env_config['save_path_experiment'], env_config['threshold'], env_config['parameters'], env_config['run'], env_config['subepisode_sampling'])
        self.env = NormalizedObs(self.env) # This makes the observation space 64x64x3 RGB with values between -1 and 1, float32
        #self.env = GrayScaleObs(self.env) # This makes the observation space 64x64 grayscale with values between 0 and 255, uint8
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed, options)
    
    def step(self, action):
        return self.env.step(action)
    
    def close(self):
        return self.env.close()