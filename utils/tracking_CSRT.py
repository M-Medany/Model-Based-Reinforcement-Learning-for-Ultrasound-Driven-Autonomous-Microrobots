import cv2
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from utils.image_postprocessing import detect_largest_cluster

# Takes an RGB image
class CSRT_tracker():

    def __init__(self, online=False, path=None, initial_image=None, params=None, autonomous=False, x=None, y=None, w=None, h=None, box_padding=0.5, **kwargs):
        self.online = online
        self.path = path
        self.counter = 0
        
        if params is None: #check if this works
            self._tracker = cv2.TrackerCSRT_create(params)
        else:
            self._tracker = cv2.TrackerCSRT_create()
            
        if autonomous:
            width = (w)*box_padding
            height = (h)*box_padding
            initial_box_coords = (x-width, y-height, w+width*2, h+height*2)

            self.initial_box_coords = np.clip(tuple(int(val) for val in initial_box_coords), 0, kwargs['img_size'])
            print("Initial box coords: ", self.initial_box_coords)
        else:
            self.initial_box_coords = self.draw_bounding_box(initial_image)
        self._tracker.init(initial_image, self.initial_box_coords)
        self.agent_location = self.bbox_to_center(self.initial_box_coords)
        self.bbox_width = w
        self.bbox_height = h
        self.blue_area = self.get_initial_blue_area(self.initial_box_coords, initial_image)

        if self.online:
            self.counter = len(os.listdir(self.path))

    def draw_bounding_box(self, image):
        bbox = cv2.selectROI("Draw Bounding Box", image, fromCenter=True, showCrosshair=True) # false if you wannaa chooose from top left corner
        cv2.destroyAllWindows()
        a, b, c, d = bbox
        return a, b, c, d
    
    def get_initial_blue_area(self, bbox, frame):
        x, y, w, h = [int(i) for i in bbox]
        # Calculate the area within the bounding box that is blue
        blue_pixels = (frame[y:y+h, x:x+w] == [255, 0, 0]).all(axis=2)
        total_blue_pixels = np.sum(blue_pixels)
        if total_blue_pixels > 0:
            return total_blue_pixels
        else:
            print("No blue pixels in initial bounding box")
            return 0

    def bbox_to_center(self, bbox):
        x, y, w, h = [int(i) for i in bbox]
        center_x = x + w // 2
        center_y = y + h // 2
        return np.array([center_x, center_y], dtype=int)

    def track(self, frame, verbose=1, return_frame=True):
        success, bbox = self._tracker.update(frame)

        if success:
            x, y, w, h = [int(i) for i in bbox]
            
            # Calculate center coordinates
            center_x = x + w // 2
            center_y = y + h // 2
            self.bbox_width = w
            self.bbox_height = h

            # Calculate the area within the bounding box that is blue
            blue_pixels = (frame[y:y+h, x:x+w] == [255, 0, 0]).all(axis=2)
            blue_pixels_coordinates = blue_pixels.nonzero()                 
            center_x1 = np.mean(blue_pixels_coordinates[1]) + x
            center_y1 = np.mean(blue_pixels_coordinates[0]) + y
            total_blue_pixels = np.sum(blue_pixels)

            if total_blue_pixels > 0:
                total_pixels = w * h
                self.blue_area = total_blue_pixels

                if verbose==2:
                    print("Center coordinates:", center_x, center_y)
                    print("Center coordinates from blue bubble:", center_x1, center_y1)
                    print("Bubble area:", self.blue_area)
                    print('blue pixels:', total_blue_pixels)
                    print('total pixels:', total_pixels)

                self.bbox = bbox
                self.center = (center_x1, center_y1)
                self.agent_location = np.array([center_x1, center_y1], dtype=int)

                annotated_frame = cv2.rectangle(np.copy(frame), (x, y), (x + w, y + h), (0, 255, 0), 2)
                if return_frame:
                    return annotated_frame, True
            
        self.blue_area = 0
        annotated_frame = frame

        if return_frame:
            return annotated_frame, False
        
    def get_agent_location(self):
        return self.agent_location

    def get_bubble_area(self):
        return self.blue_area
    
    def get_bbox_width_and_height(self):
        return self.bbox_width, self.bbox_height


class TrackingException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

if __name__ == '__main__':
    test = 2
    if test == 1:
        frame0 = cv2.imread('/home/m4/git/DQN_for_Microrobot_control/test_imgs/tracking/blue_frame_episode_2_state_61.png')
        frame1 = cv2.imread('/home/m4/git/DQN_for_Microrobot_control/test_imgs/tracking/blue_frame_episode_2_state_80.png')
    test = 2
    if test == 1:
        frame0 = cv2.imread('/home/m4/git/DQN_for_Microrobot_control/test_imgs/tracking/blue_frame_episode_2_state_61.png')
        frame1 = cv2.imread('/home/m4/git/DQN_for_Microrobot_control/test_imgs/tracking/blue_frame_episode_2_state_80.png')

        length, width, channels = frame0.shape
        print(frame0.shape)

        tracker = CSRT_tracker(initial_image=frame0)
        im_out = tracker.track(frame1, verbose=2)
        center = tracker.get_agent_location()
        target = np.array([170, 180])
        target_2 = np.array([0, 0])
        target_3 = np.array([128, 142])
        target_4 = np.array([143, 157])

        center_norm = center / np.array([width, length])
        target_norm = target / np.array([width, length])
        target_norm_2 = target_2 / np.array([width, length])
        target_norm_3 = target_3 / np.array([width, length])
        target_norm_4 = target_4 / np.array([width, length])
        target_norm_5 = center_norm    
        print("Center norm: ", center_norm)
        print("Target norm: ", target_norm)
        print(np.linalg.norm(center - target, ord=2))
        norm_dist = np.linalg.norm(center_norm - target_norm, ord=2)
        norm_dist_2 = np.linalg.norm(center_norm - target_norm_2, ord=2)
        norm_dist_3 = np.linalg.norm(center_norm - target_norm_3, ord=2)
        norm_dist_4 = np.linalg.norm(center_norm - target_norm_4, ord=2)
        norm_dist_5 = np.linalg.norm(center_norm - target_norm_5, ord=2)
        print("Normalized distance: ", norm_dist)
        print("Normalized distance: ", norm_dist_2)
        print("Normalized distance: ", norm_dist_3)

        function = lambda x: 0.03*(1/(x+0.1))
        print("\nNormalized distance rewards: ")
        print(function(norm_dist))
        print(function(norm_dist_2))
        print(function(norm_dist_3))
        print(function(norm_dist_4))
        print(function(norm_dist_5))
        # Plot a circle at the center point
        center = tracker.get_agent_location()
        radius = 10  # Adjust the radius as needed
        color = (255, 0, 0)  # Green color
        thickness = 2  # Adjust the thickness as needed
        cv2.circle(im_out, center, 5, (0, 0, 255), -1)
        cv2.circle(im_out, target, 5, (0, 255, 0), -1)
        cv2.imshow('test', im_out)

        cv2.waitKey(0)

    if test == 2:
        path = '/Users/liamachenbach/Library/Mobile Documents/com~apple~CloudDocs/BachelorThesis/Images/blue_frame_episode_7_state_25.png'
        img = cv2.imread(path)
        tracker = CSRT_tracker(initial_image=img)
        tracked_img = tracker.track(img, verbose=2)
        cv2.imshow('test', tracked_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imwrite('/Users/liamachenbach/Library/Mobile Documents/com~apple~CloudDocs/BachelorThesis/Images/blue_frame_episode_7_state_25_tracked.png', tracked_img)