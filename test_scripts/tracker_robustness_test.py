from utils.tracking_CSRT import CSRT_tracker
import cv2
import time
from utils.image_postprocessing import create_binary_bitmap, plot_cluster_on_image_blue, resize_and_crop_frame
import json
import numpy as np


def tracker_robustness_test(parameters: str='default', threshold = 80):
    
    counter = 1
    print(threshold)
    path_to_directory = '/home/m4/Documents/Tracker_Robustness_Data_Acquisition/images_20240108-161516'

    init_image = cv2.imread(path_to_directory + '/frame_1.png')
    roi_x, roi_y, roi_width, roi_height= cv2.selectROI("Draw ROI of whole image", init_image, fromCenter=False, showCrosshair=True)
    cropped_frame, _ = resize_and_crop_frame(init_image, roi_x, roi_y, roi_width, roi_height)

    good_enough = False
    mask_in = None
    while not good_enough:
        processed, segmented, mask_in = create_binary_bitmap(cropped_frame, manual=True, mask_in=mask_in)
        frame_cleaned = plot_cluster_on_image_blue(segmented, cropped_frame, threshold=threshold)
        cv2.imshow("Blue Image", frame_cleaned)
        cv2.waitKey(0)
        answer = input("Good Enough???? [y|n] ")
        if answer.lower() == "y":
            good_enough = True

    cv2.destroyAllWindows()
    # read params from json file
    with open(f'/home/m4/Documents/Tracker_Robustness_Data_Acquisition/CSRT_parameters/{parameters}.json', 'r') as f:
        params = json.load(f)
    print(params)

    param_handler = cv2.TrackerCSRT_Params()
    for key, value in params.items():
        param_handler.__setattr__(key, value)

    tracker = CSRT_tracker(initial_image=frame_cleaned, params=param_handler)
    
    done = False

    while not done:
        # Read frame from folder
        frame = cv2.imread(path_to_directory + '/frame_' + f'{counter}' + '.png')
        frame_for_copy = np.copy(frame)
        resized_frame, _ = resize_and_crop_frame(frame_for_copy, roi_x, roi_y, roi_width, roi_height)
        frame_cleaned = plot_cluster_on_image_blue(segmented, resized_frame, threshold=threshold)

        annotated_frame = tracker.track(frame_cleaned)
        
    
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        #cv2.imshow("CSRT Tracking", annotated_frame)
        cv2.imshow("CSRT Tracking", annotated_frame)

        if counter == 2100:
            done = True
        # Increment the counter
        # Reset background image to plot on
        counter += 1

        # Delay for approximately 0.33 seconds (3 Hz)
        #time.sleep(0.1)


cv2.destroyAllWindows()



if __name__ == '__main__':
    tracker_robustness_test('rgb_wlr_4', threshold=100)