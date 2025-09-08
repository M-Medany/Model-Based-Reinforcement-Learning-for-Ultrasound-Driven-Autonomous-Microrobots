import cv2
import numpy as np
import matplotlib.pyplot as plt
#read an image with opencv and plot a list of points onto it and then show it
def plot_points_on_image(image, points):
    """
    Plot a list of points on an image
    :param image:   Image to plot points on
    :param points:  List of points to plot
    :return:        None
    """
    # Plot points on image
    for point in points:
        cv2.circle(image, (point[0], point[1]), 5, (0, 0, 255), -1)
    # Show image
    plt.imshow(image)
    plt.show()


img = cv2.imread('example_images/img_original_2.png')
points = np.array([[100, 450], [300, 450], [800, 400], [750, 650], [500, 300],[500, 650], [0,0], [500, 500]])
labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
plot_points_on_image(img, points)
