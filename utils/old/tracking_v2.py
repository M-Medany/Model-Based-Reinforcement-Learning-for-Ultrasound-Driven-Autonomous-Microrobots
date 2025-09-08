# Imports
import cv2
from image_postprocessing import create_binary_bitmap, plot_cluster_on_image, plot_cluster_on_image_blue
import time
import matplotlib.pyplot as plt
import numpy as np

# call the background segmentation once

path_to_directory = '/Users/liamachenbach/Desktop/BT_data/Experiment_manual_control_22_11_2023/exp1700647222.7612143/img'

start_time = time.time()
channel_segmented_image = cv2.imread(path_to_directory + '/img_1071.png')
CHANNEL_SEGMENTED = create_binary_bitmap(channel_segmented_image)
end_time = time.time()
print(f"create_binary_bitmap execution time: {end_time - start_time} seconds")

done = False
counter = 100

# Open CV CSRT tracking

def draw_bounding_box(image):
    bbox = cv2.selectROI("Draw Bounding Box", image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    a, b, c, d = bbox
    return a, b, c, d

# Usage example:
frame = cv2.imread(path_to_directory + '/img_' + f'{counter}' + '.png')
frame_cleaned = plot_cluster_on_image_blue(CHANNEL_SEGMENTED, frame)
a, b, c, d = draw_bounding_box(frame_cleaned)
tracker = cv2.TrackerCSRT_create()
tracker.init(frame_cleaned, (a, b, c, d))
counter += 1

while not done:
    # Read frame from folder
    frame = cv2.imread(path_to_directory + '/img_' + f'{counter}' + '.png')
    frame_cleaned = plot_cluster_on_image_blue(CHANNEL_SEGMENTED, frame)
    
    success, bbox = tracker.update(frame_cleaned)
    annotated_frame = np.copy(frame_cleaned)

    if success:
        # Tracking successful, update the bounding box
        x, y, w, h = [int(i) for i in bbox]
        annotated_frame = cv2.rectangle(frame_cleaned, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Calculate center coordinates
        center_x = x + w // 2
        center_y = y + h // 2
        print("Center coordinates:", center_x, center_y)
        
        # Calculate the area within the bounding box that is black
        black_pixels = np.sum(frame_cleaned[y:y+h, x:x+w] == 0)
        total_pixels = w * h
        black_area = black_pixels / total_pixels
        print("Black area:", black_area)
    else:
        # Tracking failed, you may want to handle this case
        raise Exception("Tracking failed at frame: " + str(counter))

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    #cv2.imshow("CSRT Tracking", annotated_frame)
    cv2.imshow("CSRT Tracking", annotated_frame)

    if counter == 1000:
        done = True
    # Increment the counter
    # Reset background image to plot on
    counter += 1

    # Delay for approximately 0.33 seconds (3 Hz)
    time.sleep(0.1)


cv2.destroyAllWindows()







# frame = cv2.imread(path_to_directory + '/img_' + f'{counter}' + '.png')
# frame_cleaned = plot_cluster_on_image(channel_segmented, frame)

# # display the image and the cleaned image next to each otehr

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
# ax1.imshow(frame)
# ax2.imshow(frame_cleaned)
# ax3.imshow(channel_segmented, cmap= 'gray')
# plt.show()


# Yolo tracking

# while not done:
#     # Read frame from folder
#     frame = cv2.imread(path_to_directory + '/img_' + f'{counter}' + '.png')
#     frame_cleaned = plot_cluster_on_image(CHANNEL_SEGMENTED, frame)
#     results = model.track(frame_cleaned, persist=True)

#     annotated_frame = results[0].plot()

#     # Display the annotated frame
#     cv2.imshow("YOLOv8 Tracking", annotated_frame)

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

#     if counter == 400:
#         done = True
#     # Increment the counter
#     # Reset background image to plot on
#     counter += 1

#     # Delay for approximately 0.33 seconds (3 Hz)
#     time.sleep(0.33)
    

# cv2.destroyAllWindows()

