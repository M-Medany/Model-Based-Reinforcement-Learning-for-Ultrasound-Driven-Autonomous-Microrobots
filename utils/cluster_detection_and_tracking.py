import cv2
from operator import itemgetter
import numpy as np
from settings import *
import matplotlib.pyplot as plt


""" DETECTION AND TRACKING """


# Find the indices of the top n values from a list or array quickly
def find_top_n_indices(data, top):
    indexed = enumerate(data)  # create pairs [(0, v1), (1, v2)...]
    sorted_data = sorted(indexed,
                         key=itemgetter(1),   # sort pairs by value
                         reverse=True)       # in reversed order
    return [d[0] for d in sorted_data[:top]]  # take first N indices


# Find n largest clusters using thresholding, canny edge detection and contour finding from OpenCV
def find_clusters(image, amount_of_clusters, verbose=True):
    """
    Detect clusters based on blur, thresholding and canny edge detection
    :param image:               Working image
    :param amount_of_clusters:  Number of clusters to detect (algorithm detects the #amount_of_clusters biggest ones)
    :param verbose:             Plotting True or False
    :return:                    Centroids, areas, bboxes of clusters
    """
    # Exception handling
    if not image.any():
        raise ValueError('Empty image received')

    # Check if image is grayscale
    # assert len(image.shape) == 2, "Image must be grayscale"

    # Using cv2.blur() method
    cleared_image = cv2.blur(cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)[1], (2, 2))  # TODO --> Automatic threshold settings

    # Separate clusters from background and convert background to black
    canny = cv2.Canny(cleared_image, threshold1=0, threshold2=0)

    # Find contours
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Exception handling
    if not contours:
        raise ValueError('No contours detected')

    # Locate n biggest contours
    biggest_contours = find_top_n_indices([cv2.contourArea(con) for con in contours],
                                          top=amount_of_clusters)

    # Locate the centroid of each contour
    centroids = []
    areas = []
    bboxes = []

    # Find the features of contours
    for n in biggest_contours:

        # Calculate centroid moment and area
        M = cv2.moments(contours[n])
        cX = int(M["m10"] / (M["m00"] + 1e-8))
        cY = int(M["m01"] / (M["m00"] + 1e-8))
        area = cv2.contourArea(contours[n])

        if area <= 1:
            continue

        # Add features to respective lists
        centroids.append((cX, cY))
        areas.append(area)
        squared_area = np.sqrt(area)
        bboxes.append([int(cX-squared_area),
                       int(cY-squared_area),
                       int(2*squared_area),
                       int(2*squared_area)])
    # Draw results
    if verbose:

        img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        for n in range(len(centroids)):
            cv2.drawContours(img, contours, n, (0, 0, 255), 1)
            cv2.circle(img, centroids[n], 0, (255, 0, 0), 5)
            cv2.rectangle(img,
                          pt1=(int(bboxes[n][0]), int(bboxes[n][1])),
                          pt2=(int(bboxes[n][0] + bboxes[n][2]), int(bboxes[n][1] + bboxes[n][3])),
                          color=(255, 255, 0))

        # Display result
        cv2.imshow("Tracking", img)
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            return

    return centroids, areas, bboxes

class TrackClusters:

    def __init__(self, bbox=None):
        self.bbox = bbox

    def reset(self, img):
        """
        Initialize tracker based on first image and initial swarm coordinate
        :param img: Working image
        :return:    Center and size of cluster
        """
        # Check if we specified a bounding box to start with, otherwise select largest cluster
        if not self.bbox:
            _, _, self.bbox = find_clusters(image=img, amount_of_clusters=1, verbose=False)
        if not self.bbox:
            self.bbox = (0, 0, IMG_SIZE, IMG_SIZE)

        # Define tracker and initialise
        self.tracker = cv2.TrackerCSRT_create()  # Very accurate, dynamic sizing, not the fastest, still okay
        # self.tracker = cv2.legacy_TrackerMedianFlow.create()  # Very fast, dynamic sizing, medium accuracy

        self.ok = self.tracker.init(img, self.bbox)

        # Calculate center of bounding box
        self.center = [int(self.bbox[0] + 0.5 * self.bbox[2]), int(self.bbox[1] + 0.5 * self.bbox[3])]

        return self.center, np.mean((self.bbox[2], self.bbox[3]))

    def update(self, img, action, target, current_time, step, path_test, verbose: bool=False):
        """
        Track cluster based on previous and current position
        :param img:     Working image
        :param target:  Target point (for verbose purposes)
        :param action:  Piezo actuation (for verbose purposes)
        :param verbose: Plotting
        :return:        Center and size of cluster
        """
        # Perform tracker update and calculate new center
        try:
            self.ok, self.bbox = self.tracker.update(img)
        except:
            raise ValueError('Swarm could not be tracked')
        self.bbox = list(self.bbox)
        self.center = [int(self.bbox[0] + 0.5 * self.bbox[2]), int(self.bbox[1] + 0.5 * self.bbox[3])]


        # Draw results
        if verbose:

            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # Tracking success
            p1 = (int(self.bbox[0]), int(self.bbox[1]))
            p2 = (int(self.bbox[0] + self.bbox[2]), int(self.bbox[1] + self.bbox[3]))
            cv2.rectangle(img, p1, p2, (255, 102, 102))
            cv2.circle(img, target, 0, (178, 255, 102), 5)


            # Draw green line on the side which the piezo was actuated
            if action == 0: # right
                cv2.line(img, (IMG_SIZE - 2, IMG_SIZE), (298, 0), (153, 153, 255), 4)
            elif action == 2: # up
                cv2.line(img, (2, IMG_SIZE), (2, 0), (153, 153, 255), 4)
            elif action == 1: # down
                cv2.line(img, (0, IMG_SIZE - 2), (IMG_SIZE, 298), (153, 153, 255), 4)
            elif action == 3: # left
                cv2.line(img, (0, 2), (IMG_SIZE, 2), (153, 153, 255), 4)

            cv2.putText(img, "Time=", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_8)
            cv2.putText(img, (str(current_time)[:5]), (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_8)
            cv2.putText(img, "Step=", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_8)
            cv2.putText(img, str(step), (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_8)

            for i in range(len(path_test) - 1):
                cv2.line(img, path_test[i], path_test[i + 1], (0, 0, 255), 2)


            # Display image
            cv2.imshow("Tracking", img)

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                return

        return self.center, np.mean((self.bbox[2], self.bbox[3]))