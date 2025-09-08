import pymmcore
import os.path
import numpy as np
import cv2
# import tektronix_func_gen as tfg
import time
import matplotlib.pyplot as plt
# from environment_pipeline_S import ActuatorPiezos
from pathlib import Path
import yaml

# Hammamatsu settings
config_path = Path(__file__).parent / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
size = (config["Layout_settings"]["IMG_SIZE"], config["Layout_settings"]["IMG_SIZE"])
counter = 1
# Initialize core object
mmc = pymmcore.CMMCore()
i = 0
# Find camera config --> get camera
mmc.setDeviceAdapterSearchPaths(["C:/Program Files/Micro-Manager-2.0gamma"])
mmc.loadSystemConfiguration('C:/Program Files/Micro-Manager-2.0gamma/mmHamamatsu.cfg')
label = mmc.getCameraDevice()
mmc.setExposure(config["Hammamatsu_settings"]["EXPOSURE_TIME"])
mmc.getLoadedDevices()
mmc.snapImage()
img = mmc.getImage()  # img - it's just numpy array

# cv2.namedWindow('Video')

mmc.startContinuousSequenceAcquisition(1)
# actuator = ActuatorPiezos()
# actuator.close(5)

while True:
    # img = mmc.getImage()
    if mmc.getRemainingImageCount() > 0:
        frame = mmc.getLastImage()
        # img = mmc.getlastImage()
        # frame = mmc.popNextImage()
        frame = (cv2.resize(frame, size) / 256).astype("uint8")
        # cv2.resizeWindow('Video', 300, 300)
        cv2.imshow('Video', frame)
        # plt.imshow(frame, cmap='gray')
        # plt.show()
        # print(f"{size}")
        # cv2.imwrite('C:/Users/ARSL/Desktop/Mahmoud/writing/' + str(i) + '.tiff', frame)
        # i += 1
        # plt.imshow(img, cmap='gray')
        # plt.imshow(img)
        # plt.show()
        # plt.close(frame)
    if cv2.waitKey(20) >= 0:
        break

cv2.destroyAllWindows()
mmc.stopSequenceAcquisition()
mmc.reset()
    # or frame = mmc.popNextImage()
    # img = mmc.snap()
    # # cv2.imwrite('C:/Users/ARSL/Desktop/Mahmoud/experiment1.png', img)
    # # cv2.imshow("image", img)
    # imgplot = plt.imshow(img)

# from pymmcore_plus import CMMCorePlus
# core = CMMCorePlus.instance()
# core.loadSystemConfiguration() # loads demo config
# print(core.getLoadedDevices())

