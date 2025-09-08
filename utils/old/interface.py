import time
import numpy as np
import binascii
import pymmcore
import yaml
from pathlib import Path
import serial
import cv2
from ..image_processing import image_cleanup, find_largest_clusters
from ..segmentation import ImageSegmentation


class CameraHammamatsu(object):

    def __init__(self, config):

        # Initialize core object
        self.core = pymmcore.CMMCore()
        self.config = config

        # Find camera self.config --> get camera
        self.core.setDeviceAdapterSearchPaths(["C:/Program Files/Micro-Manager-2.0gamma"])
        self.core.loadSystemConfiguration('C:/Program Files/Micro-Manager-2.0gamma/mmHamamatsu.cfg')
        self.label = self.core.getCameraDevice()

        # Set exposure time
        self.core.setExposure(self.config['EXPOSURE_TIME'])

        # Prepare acquisition
        self.core.prepareSequenceAcquisition(self.label)
        self.core.startContinuousSequenceAcquisition(0.025)
        self.core.initializeCircularBuffer()
        time.sleep(1)
        print('Camera initialized')
    
    #TODO: Check framerate, do image processing
    #TODO: Maybe split this up into two functions to make it computationally more efficient
    def get_image_and_bubble_coordinates(self, segment_image=False):
        self.core.snapImage()
        img = self.core.getImage()
        img = (cv2.resize(img, (self.config['IMG_SIZE'], self.config['IMG_SIZE'])) / 256).astype("uint8")
        #Get cleaned image
        cleaned_img = image_cleanup(img, self.config['THRESHOLD_VALUE']) #TODO: check threshold value
        segmented_img = self.generate_bitmap(img, self.config['sam_config'])

        #Get coordinates
        coordinates = find_largest_clusters(cleaned_img, self.config['THRESHOLD_VALUE'], 1) #TODO: How many clusters do we want to find? (I guess one)

        cv2.destroyAllWindows()
        self.core.stopSequenceAcquisition()
        self.core.reset()

        return cleaned_img, segmented_img, coordinates
    
    def generate_bitmap(self, img, sam_config):
        """
        Generates a binary image from the input image.

        Args:
            img (numpy.ndarray): The input image.
            sam_config (dict): The configuration dictionary for the image segmentation.

        Returns:
            numpy.ndarray: The binary image.

        """
        sam = ImageSegmentation(sam_config)
        masks, _, _ = sam.predict(img)

        return masks[0]
    

    # def get_cleaned_image(self):
    #     self.core.snapImage()
    #     img = self.core.getImage()
    #     img = (cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 256).astype("uint8")
    #     cv2.destroyAllWindows()
    #     return image_cleanup(img, THRESHOLD_VALUE) #TODO: check threshold value


    # def get_bubble_coordinates(self, cleaned_img):
    #     #Get coordinates
    #     coordinates = find_largest_clusters(cleaned_img, THRESHOLD_VALUE, 1) #TODO: How many clusters do we want to find? (I guess one)

    #     cv2.destroyAllWindows()
        
    #     return coordinates


class VideoStreamHammamatsu:

    def __init__(self, config):

        # Initialiye core object
        self.core = pymmcore.CMMCore()

        # Find camera config --> get camera
        self.core.setDeviceAdapterSearchPaths(["C:/Program Files/Micro-Manager-2.0gamma"])
        self.core.loadSystemConfiguration('C:/Program Files/Micro-Manager-2.0gamma/mmHamamatsu.cfg')
        self.label = self.core.getCameraDevice()

        # Set exposure time
        self.core.setExposure("EXPOSURE_TIME")
        self.config = config

        # Prepare acquisition
        self.core.prepareSequenceAcquisition(self.label)
        self.core.startContinuousSequenceAcquisition(0.025)
        self.core.initializeCircularBuffer()
        time.sleep(1)

    def snap(self, f_name, size=(256, 256)):

        # Error handling (sometimes the video buffer is empty if we take super fast images)
        img = None
        while not np.any(img):
            try:
                # Get image
                img = self.core.getLastImage()
            except:
                pass

        # Resize image
        img = (cv2.resize(img, size) / 256).astype("uint8")

        # Save image
        cv2.imwrite(f_name, img)

        # Return image
        return img


class VideoStreamKronos:
    STREAM_URL = "rtsp://10.4.96.106"
    def __init__(self, url=STREAM_URL):
        import vlc

        # Define VLC instance
        instance = vlc.Instance()

        # Define VLC player
        self.player = instance.media_player_new()

        # Define VLC media
        self.media = instance.media_new(url)

        # Set player media
        self.player.set_media(self.media)
        self.player.play()
        time.sleep(2)

    def snap(self, f_name):

        # Snap an image of size (IMG_SIZE, IMG_SIZE)
        self.player.video_take_snapshot(0, f_name, 256, 256)
