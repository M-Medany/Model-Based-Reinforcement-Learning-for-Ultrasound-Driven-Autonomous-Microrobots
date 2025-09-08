import cv2
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import yaml
import numpy as np
from typing import Union, Tuple
from utils.segmentation import ImageSegmentation
import time
from operator import itemgetter
import os
from pathlib import Path


path = Path(os.path.dirname(os.path.abspath(__file__))).parent
with open(Path(path, 'scripts/config.yaml'), 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)
    
def select_ROI(image):
    ROIs = []
    while True:
        ROI = cv2.selectROI("Draw ROI", image, fromCenter=False, showCrosshair=True)
        if ROI[2] != 0 and ROI[3] != 0:
            ROIs.append(ROI)
            cv2.rectangle(image, (ROI[0], ROI[1]), (ROI[0]+ROI[2], ROI[1]+ROI[3]), (0, 255, 0), 2)
        else:
            break
    cv2.destroyAllWindows()
    return ROIs

def select_image_points(image):
    points = []

    # Mouse callback function
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            points.append((x, y))
            print(points)
    # Create a window and set the mouse callback function
    cv2.namedWindow('Select Image Points')
    cv2.setMouseCallback('Select Image Points', mouse_callback)

    while True:
        # Display the image
        
        cv2.imshow('Select Image Points', image)

        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF

        # If the 'q' key is pressed, break from the loop
        if key == ord('q'):
            break

    # Convert the list of points to a numpy array
    points_array = np.array(points)
    cv2.destroyAllWindows()

    return points_array

def select_image_point(image):
    points = []

    # Mouse callback function
    def mouse_callback(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            points.append((x, y))
            print(points)
        if event == cv2.EVENT_RBUTTONDOWN:
            exiting = True
            print("Exiting")
            points = [None]
            cv2.destroyAllWindows()
            return

    # Create a window and set the mouse callback function
    cv2.namedWindow('Select Image Points')
    cv2.setMouseCallback('Select Image Points', mouse_callback)

    while True:
        cv2.imshow('Select Image Points', image)
        key = cv2.waitKey(1) & 0xFF
        if len(points) == 1:
            break
        if key == ord('q'):
            points = [None]
 
    cv2.destroyAllWindows()
    return points[0]


def create_binary_bitmap(image, segmentation: ImageSegmentation, mask_in=None, foreground_points=None, background_points=None, kernel_size=(5,5), kernel_iteration=3, loops=25):
    """
    Returns:
        - closed image (from morphology application)
        - low_res_mask
    """

    if foreground_points is None :
        print("Select input points:")
        foreground_points = select_image_points(image)
    
    if background_points is None:
        print("Select input background:")
        background_points = select_image_points(image)
    
    input_labels = np.array([1]*len(foreground_points) + [0]*len(background_points))
    print(input_labels)
    print(foreground_points)
    print(background_points)
    foreground_points = np.array(foreground_points)
    background_points = np.array(background_points)
    input_points = np.concatenate((foreground_points, background_points), axis=0)

    segmentation.add_input_points(input_points=input_points, input_labels=input_labels)
    
    for _ in range(loops):
        masks, _, mask_out = segmentation.predict(mask_in)
        mask_in = mask_out

    # apply mask to image
    binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    binary_mask[masks[0]] = 1
    kernel = np.ones(kernel_size, np.uint8)
    closed_image = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=kernel_iteration)
    
    output = np.ones(image.shape[:2], dtype=np.uint8)
    output[closed_image == 1] = 0

    return closed_image, mask_out


# Takes a binary image as a bitmap and returns a RGB image
def plot_cluster_on_image(channel_mask, image):
    threshold = 80
    thresholded_image = np.zeros(image.shape, dtype=np.uint8)
    thresholded_image[np.logical_and(image[:, :, 0] > threshold, image[:, :, 1] > threshold, image[:, :, 2] > threshold)] = 255
    
    thresholded_image = cv2.cvtColor(thresholded_image, cv2.COLOR_BGR2GRAY)
    channel_mask_copy = np.copy(channel_mask)

    channel_mask_copy[thresholded_image != 255] = 0
    channel_mask_copy = channel_mask_copy * 255
    # make channel mask an rgb image
    channel_mask_copy = cv2.cvtColor(channel_mask_copy, cv2.COLOR_GRAY2RGB)

    return channel_mask_copy


# Takes a binary image as a bitmap and returns a RGB image
def plot_cluster_on_image_blue(channel_mask, image, threshold):
    
    thresholded_image = np.zeros(image.shape, dtype=np.uint8)
    thresholded_image[np.logical_and(image[:, :, 0] < threshold, image[:, :, 1] < threshold, image[:, :, 2] < threshold)] = [255, 0, 0]

    channel_mask_copy = np.copy(channel_mask)
    channel_mask_copy *= 255
    channel_mask_copy = cv2.cvtColor(channel_mask_copy, cv2.COLOR_GRAY2RGB)

    thresholded_image[channel_mask_copy[:, :, 2] != 255] = [0, 0, 0]
    
    channel_mask_copy[thresholded_image[:, :, 0] == 255] = [255, 0, 0] # blue: [255, 0, 0]

    return channel_mask_copy

def overlay_segmented_image(channel_mask, image):

    channel_mask_copy = np.copy(channel_mask)
    channel_mask_copy *= 255
    channel_mask_copy = cv2.cvtColor(channel_mask_copy, cv2.COLOR_GRAY2RGB)

    image[channel_mask_copy[:, :, 2] != 255] = [0, 0, 0]
    return image


def segment_image(obstacles, image):
    thr = np.ones(image.shape, dtype=np.uint8)
    thr[obstacles[:, :] != 0 ] = 0
    image[thr > 0] = 0
    return image, thr

# This works with the binary bitmap
def find_legal_point_target_close(binary_image: np.ndarray, start_point: np.ndarray=None):
    min_x = start_point[1] - config['General_environment_settings']['MAX_DISTANCE_TARGET_POINT']
    max_x = start_point[1] + config['General_environment_settings']['MAX_DISTANCE_TARGET_POINT']
    min_y = start_point[0] - config['General_environment_settings']['MAX_DISTANCE_TARGET_POINT']
    max_y = start_point[0] + config['General_environment_settings']['MAX_DISTANCE_TARGET_POINT']
    min_x = np.clip(min_x, 0, 1)
    max_x = np.clip(max_x, 0, 1)
    min_y = np.clip(min_y, 0, 1)
    max_y = np.clip(max_y, 0, 1)
    location_x = np.random.uniform(min_x, max_x)
    location_y = np.random.uniform(min_y, max_y)
    location = np.array([location_x*binary_image.shape[1], location_y*binary_image.shape[0]], dtype=int)
    return location[::-1] # This is needed to make coordinates match the image  
    # if binary_image[location[1], location[0]] == 1:
    # return find_legal_point_target_close(binary_image, start_point)
    
def find_legal_point(binary_image: np.ndarray):
    location_x = np.random.uniform(0, 1)
    location_y = np.random.uniform(0, 1)
    location = np.array([location_x*binary_image.shape[1], location_y*binary_image.shape[0]], dtype=int)
    return location[::-1]

def resize_and_crop_frame(frame, roi_x, roi_y, roi_width, roi_height, *img_size: Union[None, Tuple[float, float]]):
    cropped = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
    if img_size:
        resized = cv2.resize(cropped, img_size, img_size)
    else:
        resized = cv2.resize(cropped, (config['Layout_settings']['IMG_SIZE'], config['Layout_settings']['IMG_SIZE']))
    return resized, cropped


# Find largest cluster and return box
def detect_largest_cluster(image):
    
    blue_mask = (image[:,:,0] == 255) & (image[:,:,1] == 0) & (image[:,:,2] == 0) # This is a boolean mask assuming a BGR image 

    # Convert to uint8
    binary_mask = np.uint8(blue_mask) * 255

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ensure at least one contour was found
    if contours:
        # Find the largest contour based on area
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Calculate the moments of the largest contour to find its center
        # M = cv2.moments(largest_contour)
        # if M["m00"] != 0:
        #     cX = int(M["m10"] / M["m00"])
        #     cY = int(M["m01"] / M["m00"])
        # else:
        #     cX, cY = 0, 0
        #     print("No center found in detect_largest_cluster, returning 0, 0, M['m00'] = 0")

        print("Box of largest cluster: ", x, y, w, h)
        return x, y, w, h
    else:
        print("No contours found in detect_largest_cluster, returning 0, 0, 0, 0")
        return 0, 0, 0, 0


def detect_largest_cluster_center(image):
    
    blue_mask = (image[:,:,0] == 255) & (image[:,:,1] == 0) & (image[:,:,2] == 0) # This is a boolean mask assuming a BGR image 

    # Convert to uint8
    binary_mask = np.uint8(blue_mask) * 255

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ensure at least one contour was found
    if contours:
        # Find the largest contour based on area
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate the moments of the largest contour to find its center
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
            print("No center found in detect_largest_cluster, returning 0, 0, M['m00'] = 0")
        return cX, cY
    else:
        print("No contours found in detect_largest_cluster, returning 0, 0")
        return 0, 0

    


if __name__ == '__main__':

    test = "find_legal_point" #TODO: Change this

    if test == "binary_bitmap":
        source = '/home/m4/git/DQN_for_Microrobot_control/example_images/img_original_1.png'
        image = cv2.imread(source)
        threshold = 80

        
        # _, channel_mask = create_binary_bitmap(image, manual=True)

        
        # test = plot_cluster_on_image_blue_old(channel_mask, image, threshold=80)


        # cv2.imshow('test', test)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows

    


        
        # # Testing
        _, test, _ = create_binary_bitmap(image, manual=True)
        # print(test.shape)q
        # cv2.imshow('og', image)
        # cv2.imshow('Test Image', test*255)
        # image_3, _ = segment_image(test, image)
        # cv2.imshow('Test Image_2q', image_3)
        # cv2.waitKey(0)
        # image_2 = plot_cluster_on_image_blue_old(test, image, 80)
        # cv2.imshow('Test Image_2q', image_2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # # Histogram of the test image
        
    if test == 'find_legal_point':
        img = cv2.imread('/home/m4/git/DQN_for_Microrobot_control/binary_images/closing4_edited.png')
        imgcopy = img.copy()
        _, segmented, _ = create_binary_bitmap(imgcopy, manual=True)
        for i in range(1000):
            
            # find a legal point
            legal_point = find_legal_point(segmented)
            # plot onto the image
            cv2.circle(img, (legal_point[1], legal_point[0]), 5, (0, 255, 0), -1)
        cv2.imshow('Test Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            
    
    if test == "find_target_point":

        test_points = []
        cv2.imshow('Test Image', test*255)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        original_point = np.array([0.7, 0.7])
        test_2 = cv2.cvtColor(test*255, cv2.COLOR_GRAY2RGB)
        
        for i in range(10):
            original_point = np.random.uniform(0, 1, 2)
            cv2.circle(test_2, (int(original_point[0]*test.shape[1]), int(original_point[1]*test.shape[0])), 5, (255, 0, 0), -1)
            for _ in range(1000):
                try:
                    legal_point = find_legal_point_target_close(test, start_point=original_point)
                    test_points.append(legal_point)
                    cv2.circle(test_2, (legal_point[0], legal_point[1]), 5, (0, 255, 0), -1)
                    print(legal_point)
                except:
                    pass
            cv2.imshow('Test Image', test_2)
            cv2.waitKey(0)

    

        # for legal_point in test_points:
        
        # print(test.shape)

        cv2.imshow('Test Image', test_2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if test == "detection":

        source = '/Users/liamachenbach/Library/Mobile Documents/com~apple~CloudDocs/BachelorThesis/Images/blue_frame_episode_7_state_25.png'
        image = cv2.imread(source)
        cX, cY = detect_largest_cluster(image)

        cv2.circle(image, (cX, cY), 5, (0, 255, 0), -1)

        # Show the result
        cv2.imshow('Center of Largest Cluster', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imwrite('/Users/liamachenbach/Library/Mobile Documents/com~apple~CloudDocs/BachelorThesis/Images/blue_frame_episode_7_state_25_with_circle.png', image)

    if test == 'detection_run':
        source = '/Volumes/m8/experiment_piezo_fixed_20240130-130228_run_0/blue_data/'
        for i in range(50):
            for j in range(50):
                image = cv2.imread(source + f'blue_frame_episode_{i}_state_{j}.png')
                if image is not None:
                    cX, cY = detect_largest_cluster(image)

                    cv2.circle(image, (cX, cY), 5, (0, 255, 0), -1)

                    # Show the result
                    cv2.imshow('Center of Largest Cluster', image)
                    cv2.waitKey(1000)
                    cv2.destroyAllWindows()



    if test == "display":
        source = '/Volumes/m8/experiment_piezo_fixed_20240130-130228_run_0/blue_data/blue_frame_episode_1_state_0.png'
        image = cv2.imread(source)
        cv2.imshow('Center of Largest Cluster', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
