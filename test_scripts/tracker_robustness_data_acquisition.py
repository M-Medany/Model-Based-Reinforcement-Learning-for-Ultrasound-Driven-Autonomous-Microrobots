from curtsies import Input
import time
import cv2
#from pyfirmata import Arduino
from utils.actuator import Arduino, FunctionGenerator_1
import yaml
from utils.image_postprocessing import create_binary_bitmap, plot_cluster_on_image_blue, resize_and_crop_frame
import threading
import os


# Read yaml 
with open(f'scripts/config.yaml', 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

current_time = time.strftime("%Y%m%d-%H%M%S")
save_path = f'/home/m4/Documents/Tracker_Robustness_Data_Acquisition/images_{current_time}'
os.mkdir(save_path)

board=Arduino(config['Arduino_settings']['SERIAL_PORT_UBUNUTU'], baudrate=config['Arduino_settings']['BAUDRATE_ARDUINO'], config=config)
print("Arduino initialized successfully")

function_generator = FunctionGenerator_1(config=config, instrument_descriptor=config['Tektronix_settings']['INSTR_DESCRIPTOR'])
print("Function generator initialized successfully")

x1, x2, x3, x4, P1, P2, P3, P4 = 0, 0, 0, 0, 0, 0, 0, 0
piezo_1 = "d"
piezo_2 = "w"
piezo_3 = "s"
piezo_4 = "a"

cap = cv2.VideoCapture(0)


while True:
    tre, frame = cap.read()
    cv2.imshow('ROI', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
tre, frame = cap.read()



size = (1080, 720)
counter = 1
i = 0
start_time = time.time()

while(True):
    tre, frame = cap.read()

    if tre:
        _, frame = resize_and_crop_frame(frame, 0, 0, frame.shape[1], frame.shape[0], *size)
        cv2.imwrite(f"{save_path}/frame_{i}.png", frame)
        

        
    with Input(keynames='curses') as input_generator:
        e = input_generator.send(0.05)
        if e == piezo_1:
            P1 = 1
            print("turn on 1")
            board.set_piezo_after_collision(1)

        elif e == piezo_2:
            P2 = 1
            print("turn on 2")
            board.set_piezo_after_collision(2)

        elif e == piezo_3:
            P3 = 1
            print("turn on 3")
            board.set_piezo_after_collision(3)

        elif e == piezo_4:
            P4 = 1
            print("turn on 4")
            board.set_piezo_after_collision(4)
        elif e == "q":
            print("turn off all")
            P1 = 0
            P2 = 0
            P3 = 0                
            P4 = 0
            board.set_piezo_after_collision(0)

        cv2.imshow('Video', frame)
        i += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()

