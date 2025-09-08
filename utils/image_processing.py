import cv2
from operator import itemgetter
import matplotlib.pyplot as plt


def image_cleanup(image, threshold_value):
    if not image.any():
        raise ValueError('Empty image received')

    # Check if image is grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur image and threshold TODO:check threshold value    
    thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)[1], (2, 2)
    cleared_image = cv2.blur(thresholded_image) 

    #TODO:Maybe include an opening operation here to remove small objects or make channels more clean

    # Separate clusters from background and convert background to black
    canny = cv2.Canny(cleared_image, threshold1=0, threshold2=0)
    
    return canny


# Find n largest clusters using thresholding, canny edge detection and contour finding from OpenCV
def find_largest_clusters(image, threshold_value, amount_of_clusters):
    """
    Detect clusters based on blur, thresholding and canny edge detection
    :param image:               Working image
    :param amount_of_clusters:  Number of clusters to detect (algorithm detects the #amount_of_clusters biggest ones)
    :return:                    Centroids
    """
    canny = image_cleanup(image, threshold_value)

    # Find contours
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Exception handling
    if not contours:
        raise ValueError('No contours detected')

    # Locate n biggest contours
    biggest_contours = find_top_n_indices([cv2.contourArea(con) for con in contours],
                                          top=amount_of_clusters)

    
     # Calculate centroid moment and area
    M = cv2.moments(contours[biggest_contours[0]])
    cX = int(M["m10"] / (M["m00"] + 1e-8))
    cY = int(M["m01"] / (M["m00"] + 1e-8))
    return (cX, cY)


# Find the indices of the top n values from a list or array quickly
def find_top_n_indices(data, top):
    indexed = enumerate(data)  # create pairs [(0, v1), (1, v2)...]
    sorted_data = sorted(indexed,
                         key=itemgetter(1),   # sort pairs by value
                         reverse=True)       # in reversed order
    return [d[0] for d in sorted_data[:top]]  # take first N indices