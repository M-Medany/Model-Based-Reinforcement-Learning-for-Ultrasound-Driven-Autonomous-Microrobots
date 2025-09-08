import pymmcore
import os.path
import numpy as np
import time
import win32api
import cv2
from pyfirmata import Arduino

port=('com5')
pin9 = 10
pin1 = 9
pin2 = 10
pin3 = 11
pin4 =12
# pin2 = 8
# pin3 = 7
# pin4 = 6
pin5 = 5
pin6 = 4
pin7 = 3
pin8 = 2

board=Arduino(port)
i=2
j=240000
x1=0
x1=0
x2=0
x3=0
x4=0
x5=0
x6=0
x7=0
x8=0
x9=0
P1 = 0
P2 = 0
P3 = 0
P4 = 0
P5 = 0
P6 = 0
P7 = 0
P8 = 0
P9 = 0



# Hammamatsu settings
# EXPOSURE_TIME = 10 # Exposure time Hammamatsu
# IMG_SIZE = 500
EXPOSURE_TIME = 30 # Exposure time Hammamatsu
IMG_SIZE = 1000

size = (IMG_SIZE, IMG_SIZE)
counter = 1
# Initialize core object
mmc = pymmcore.CMMCore()
i = 0
# Find camera config --> get camera
mmc.setDeviceAdapterSearchPaths(["C:/Program Files/Micro-Manager-2.0gamma"])
mmc.loadSystemConfiguration('C:/Program Files/Micro-Manager-2.0gamma/mmHamamatsu.cfg')
label = mmc.getCameraDevice()
mmc.setExposure(EXPOSURE_TIME)
mmc.getLoadedDevices()
mmc.snapImage()
img = mmc.getImage()  # img - it's just numpy array

# cv2.namedWindow('Video')

mmc.startContinuousSequenceAcquisition(1)
start_time = time.time()

while True:
    # img = mmc.getImage()
    print("--- %s seconds ---" % (time.time() - start_time))
    if mmc.getRemainingImageCount() > 0:
        frame = mmc.getLastImage()
        # img = mmc.getlastImage()
        # frame = mmc.popNextImage()
        frame = (cv2.resize(frame, size) / 256).astype("uint8")
        # cv2.resizeWindow('Video', 300, 300)
        current_time = time.time() - start_time
        piezo_1 = win32api.GetKeyState(0x31)
        piezo_2 = win32api.GetKeyState(0x32)
        piezo_3 = win32api.GetKeyState(0x33)
        piezo_4 = win32api.GetKeyState(0x34)
        piezo_5 = win32api.GetKeyState(0x35)
        piezo_6 = win32api.GetKeyState(0x36)
        piezo_7 = win32api.GetKeyState(0x37)
        piezo_8 = win32api.GetKeyState(0x38)
        piezo_9 = win32api.GetKeyState(0x39)
        piezo_1_off = win32api.GetKeyState(0x61)
        piezo_2_off = win32api.GetKeyState(0x62)
        piezo_3_off = win32api.GetKeyState(0x63)
        piezo_4_off = win32api.GetKeyState(0x64)
        piezo_5_off = win32api.GetKeyState(0x65)
        piezo_6_off = win32api.GetKeyState(0x66)
        piezo_7_off = win32api.GetKeyState(0x67)
        piezo_8_off = win32api.GetKeyState(0x68)
        piezo_9_off = win32api.GetKeyState(0x69)

        if piezo_1 < 0:
            P1 = 1
            print("toggel1")
            board.digital[pin1].write(1)

        elif piezo_2 < 0:
            P2 = 1
            print("turn on 2")
            board.digital[pin2].write(1)

        elif piezo_3 < 0:
            P3 = 1
            print("turn on 3")
            board.digital[pin3].write(1)

        elif piezo_4 < 0:
            P4 = 1
            print("turn on 4")

            board.digital[pin4].write(1)
        elif piezo_5 < 0:
            P5 = 1
            print("turn on 5")
            board.digital[pin5].write(1)
        elif piezo_6 < 0:
            P6 = 1
            print("turn on 6")
            board.digital[pin6].write(1)
        elif piezo_7 < 0:
            P7 = 1
            print("turn on 7")
            board.digital[pin7].write(1)
        elif piezo_8 < 0:
            P8 = 1
            print("toggel 8")
            board.digital[pin8].write(1)
        elif piezo_9 < 0:
            P9 = 1
            print("toggel 9")
            board.digital[pin9].write(1)

        elif piezo_1_off < 0:
            P1 = 0
            print("toggel1")
            board.digital[pin1].write(0)
        elif piezo_2_off < 0:
            P2 = 0

            print("turn on 2")
            board.digital[pin2].write(0)
        elif piezo_3_off < 0:
            P3 = 0

            print("turn on 3")
            board.digital[pin3].write(0)
        elif piezo_4_off < 0:
            P4 = 0

            print("turn on 4")
            board.digital[pin4].write(0)
        elif piezo_5_off < 0:
            P5 = 0
            print("turn on 5")
            board.digital[pin5].write(0)
        elif piezo_6_off < 0:
            P6 = 0
            print("turn on 6")
            board.digital[pin6].write(0)
        elif piezo_7_off < 0:
            P7 = 0
            print("turn on 7")
            board.digital[pin7].write(0)
        elif piezo_8_off < 0:
            P8 = 0
            print("toggel 8")
            board.digital[pin8].write(0)
        elif piezo_9_off < 0:
            P9 = 0
            print("toggel 9")
            board.digital[pin9].write(0)
        # cv2.putText(frame, "Frequency=", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_8)
        # cv2.putText(frame, "Frequency=", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_8)
        # cv2.putText(frame, (str(freq)), (250, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_8)
        # cv2.putText(frame, "Amplitude=", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_8)
        # cv2.putText(frame, (str(amp)), (250, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_8)
        # cv2.putText(frame, "Time=", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_8)
        # cv2.putText(frame, (str(current_time)[:5]), (250, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3,cv2.LINE_8)
        # cv2.putText(frame, (str(P1)), (440, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_8)
        if P1 == 0:
            cv2.circle(frame, (440, 20), 10, (0, 0, 0), -1)
        elif P1 == 1:
            cv2.circle(frame, (440, 20), 10, (255, 255, 255), -1)
        if P2 == 0:
            cv2.circle(frame, (465, 20), 10, (0, 0, 0), -1)
        elif P2 == 1:
            cv2.circle(frame, (465, 20), 10, (255, 255, 255), -1)
        if P3 == 0:
            cv2.circle(frame, (490, 20), 10, (0, 0, 0), -1)
        elif P3 == 1:
            cv2.circle(frame, (490, 20), 10, (255, 255, 255), -1)
        if P4 == 0:
            cv2.circle(frame, (440, 45), 10, (0, 0, 0), -1)
        elif P4 == 1:
            cv2.circle(frame, (440, 45), 10, (255, 255, 255), -1)
        if P5 == 0:
            cv2.circle(frame, (465, 45), 10, (0, 0, 0), -1)
        elif P5 == 1:
            cv2.circle(frame, (465, 45), 10, (255, 255, 255), -1)
        if P6 == 0:
            cv2.circle(frame, (490, 45), 10, (0, 0, 0), -1)
        elif P6 == 1:
            cv2.circle(frame, (490, 45), 10, (255, 255, 255), -1)
        if P7 == 0:
            cv2.circle(frame, (440, 70), 10, (0, 0, 0), -1)
        elif P7 == 1:
            cv2.circle(frame, (440, 70), 10, (255, 255, 255), -1)
        if P8 == 0:
            cv2.circle(frame, (465, 70), 10, (0, 0, 0), -1)
        elif P8 == 1:
            cv2.circle(frame, (465, 70), 10, (255, 255, 255), -1)
        if P9 == 0:
            cv2.circle(frame, (490, 70), 10, (0, 0, 0), -1)
        elif P9 == 1:
            cv2.circle(frame, (490, 70), 10, (255, 255, 255), -1)
        # cv2.putText(frame, (str(P2)), (460, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_8)
        # cv2.putText(frame, (str(P3)), (480, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_8)
        # cv2.putText(frame, (str(P4)), (440, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_8)
        # cv2.putText(frame, (str(P5)), (460, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_8)
        # cv2.putText(frame, (str(P6)), (480, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_8)
        # cv2.putText(frame, (str(P7)), (440, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_8)
        # cv2.putText(frame, (str(P8)), (460, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_8)
        # cv2.putText(frame, (str(P9)), (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_8)
        cv2.imshow('Video', frame)
        time.sleep(0.03)
        # plt.imshow(frame, cmap='gray')
        # plt.show()
        # print(f"{size}")
        # cv2.imwrite(r'F:\Experiment_manual_control_8_11_2023\images' + str(i) + '.png', frame)
        i += 1
        # plt.imshow(img, cmap='gray')
        # plt.imshow(img)
        # plt.show()
        # plt.close(frame)

    if cv2.waitKey(1) == 27:
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



