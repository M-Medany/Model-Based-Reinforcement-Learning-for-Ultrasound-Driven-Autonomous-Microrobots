import pymmcore
import os.path
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from utils.image_processing import *
import serial

# Find the indices of the top n values from a list or array quickly
def find_top_n_indices(data, top):
    indexed = enumerate(data)  # create pairs [(0, v1), (1, v2)...]
    sorted_data = sorted(indexed,
                         key=itemgetter(1),   # sort pairs by value
                         reverse=True)       # in reversed order
    return [d[0] for d in sorted_data[:top]]  # take first N indices



# Hammamatsu settings and time between images
size = (IMG_SIZE, IMG_SIZE)

resized_image = cv2.imread('/Users/liamachenbach/Desktop/img_original_1.png')
print(resized_image.shape)

gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
print(gray_image.shape)
_, thresh_binary = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

kernel = np.ones((6, 6), np.uint8)

eroded = cv2.erode(thresh_binary, kernel)

canny = cv2.Canny(eroded, threshold1=0, threshold2=0)

contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Exception handling
if not contours:
    raise ValueError('No contours detected')


# Define the number of longest contours you want to keep
num_longest_contours = 16  # Change this value as needed

# Sort contours by their lengths in descending order
sorted_contours = sorted(contours, key=lambda x: cv2.arcLength(x, closed=True), reverse=True)

# Keep only the top N longest contours
filtered_contours = sorted_contours[:num_longest_contours]

# Draw the filtered contours on a blank canvas
contour_image = np.zeros_like(resized_image)
cv2.drawContours(contour_image, filtered_contours, -1, (255, 255, 255), 2)




#cleaned = largest_connected_component(eroded, kernel)

#extracted = extract_largest_two_components(closed_image)

# Display the images with appropriate titles
plt.subplot(141), plt.imshow(thresh_binary, cmap='gray')
plt.title('Threshold Binary')
plt.subplot(142), plt.imshow(eroded, cmap='gray')
plt.title('eroded')
plt.subplot(143), plt.imshow(canny, cmap='gray')
plt.title('canny')
plt.subplot(144), plt.imshow(contour_image, cmap='gray')
plt.title('cleaned')

# Create a new figure for the next set of images
plt.show()

