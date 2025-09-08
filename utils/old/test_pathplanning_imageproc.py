import cv2

from ultralytics import YOLO
import cv2
from image_postprocessing import create_binary_bitmap, plot_cluster_on_image, plot_cluster_on_image_blue
import time
import matplotlib.pyplot as plt
import numpy as np
from path_planning_v2 import RRTStar, find_legal_point_target
import numpy as np
import yaml

# call the background segmentation once

path_to_directory = '/Users/liamachenbach/Desktop/BT_data/Experiment_manual_control_22_11_2023/exp1700647222.7612143/img'
save_path = '/Users/liamachenbach/Desktop/BT_data/'

with open(f'scripts/config.yaml', 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

start_time = time.time()
channel_segmented_image = cv2.imread(path_to_directory + '/img_1071.png')
CHANNEL_SEGMENTED = create_binary_bitmap(channel_segmented_image)
end_time = time.time()
print(f"create_binary_bitmap execution time: {end_time - start_time} seconds")
print(CHANNEL_SEGMENTED.shape)
print(CHANNEL_SEGMENTED.dtype)
print(CHANNEL_SEGMENTED)

# Print how much black and white in channel segmented
print(np.sum(CHANNEL_SEGMENTED == 0))
print(np.sum(CHANNEL_SEGMENTED == 1))


legal_point = find_legal_point_target(CHANNEL_SEGMENTED, (100, 100))
print('legal_point: ', CHANNEL_SEGMENTED[legal_point[0], legal_point[1]])

test_point = find_legal_point_target(CHANNEL_SEGMENTED, (100, 100))
print('test_point', CHANNEL_SEGMENTED[test_point[0], test_point[1]])

print(legal_point)
print(legal_point.dtype)
print(legal_point.shape)


# perform a numpy array comparison with legal_point and test_point

print(np.array_equal(legal_point, test_point))

plt.imshow(CHANNEL_SEGMENTED)


for _ in range(100):
    test_point = find_legal_point_target(CHANNEL_SEGMENTED, (100, 100))
    print('test_point', CHANNEL_SEGMENTED[test_point[0], test_point[1]])
    # plot the point on the image
    plt.scatter(test_point[0], test_point[1], c='b')
    

plt.show()

# start_time_2 = time.time()
# rrt = RRTStar(CHANNEL_SEGMENTED, test_point, legal_point, save_path=save_path, config=config)
# rrt.plan()
# end_time_2 = time.time()
# print(rrt.path)
# print(f"RRTStar execution time: {end_time_2 - start_time_2} seconds")

# # Plot the path on the image and the start point in blue and the end point in red and the path in green
# plt.imshow(CHANNEL_SEGMENTED)
# plt.scatter(test_point[0], test_point[1], c='b')
# plt.scatter(legal_point[0], legal_point[1], c='r')

# # Convert the path to a numpy array
# path_array = np.array(rrt.path)

# # Check if the path is not empty before plotting
# if path_array.size > 0:
#     # Plot the path using the x and y coordinates
#     plt.plot(path_array[:, 0], path_array[:, 1], c='g')

# plt.show()





