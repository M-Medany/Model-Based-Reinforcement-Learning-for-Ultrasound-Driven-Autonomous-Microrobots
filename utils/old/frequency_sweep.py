from curtsies import Input
import time
import cv2
#from pyfirmata import Arduino
from utils.actuator import Arduino, FunctionGenerator_1
import yaml
from utils.image_postprocessing import create_binary_bitmap, plot_cluster_on_image_blue, resize_and_crop_frame, detect_largest_cluster_center
from utils.tracking_CSRT import CSRT_tracker
import numpy as np
import csv

track = False

# Read yaml 
with open(f'scripts/config.yaml', 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

board=Arduino(config['Arduino_settings']['SERIAL_PORT_UBUNUTU'], baudrate=config['Arduino_settings']['BAUDRATE_ARDUINO'], config=config)
print("Arduino initialized successfully")

function_generator = FunctionGenerator_1(config=config, instrument_descriptor=config['Tektronix_settings']['INSTR_DESCRIPTOR'])
print("Function generator initialized successfully")

x1, x2, x3, x4, P1, P2, P3, P4 = 0, 0, 0, 0, 0, 0, 0, 0
piezo_1 = "s"
piezo_2 = "d"
piezo_3 = "a"
piezo_4 = "w"
increase_frequency = "+"
decrease_frequency = "-"
increase_amplitude = ">"
decrease_amplitude = "<"
start_frequency_sweep = "f"
start_vpp_sweep = "v"
start_frequency = 1.8
end_frequency = 2.2
step = 0.5
saving_frames = True

def generate_frequency_sweep(start_frequency, end_frequency, duration, step=0.1):
    current_frequency = start_frequency
    while current_frequency < end_frequency:
        function_generator.set_frequency(current_frequency)
        time.sleep(duration)
        current_frequency += step
        
        
def generate_vpp_sweep(start_vpp, end_vpp, duration, step=0.1):
    current_vpp = start_vpp
    while current_vpp < end_vpp:
        function_generator.set_vpp(current_vpp)
        time.sleep(duration)
        current_vpp += step        
        

# current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
# with open(f'/media/m4/Mahmoud_T7_Touch/Sweep/sweep_{current_time}.csv', 'w', newline="") as file:
#             writer = csv.writer(file)
#             writer.writerow(["Time", "Frequency", "Amplitude", "P1", "P2", "P3", "P4", "x1", "y1", "x2", "y2"])


cap = cv2.VideoCapture(0)

# if track:
#     while True:
#         tre, frame = cap.read()
#         cv2.imshow('ROI', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     tre, frame = cap.read()

#     good_enough = False
#     mask_in = None
#     while not good_enough:
#         processed, segmented, mask_in = create_binary_bitmap(frame, manual=True, mask_in=mask_in)
#         frame_cleaned = plot_cluster_on_image_blue(segmented, frame, 80)
#         cv2.imshow("Blue Image", frame_cleaned)
#         cv2.imshow("Processed Image 2", processed*255)
#         cv2.waitKey(0)
#         answer = input("Good Enough???? [y|N] ")
#         if answer.lower() == "y":
#             good_enough = True
#     tracker = CSRT_tracker(initial_image=frame_cleaned)


size = (1080, 720)
counter = 1
i = 0
start_time = time.time()
current_frequency = start_frequency

while(True):
    tre, frame = cap.read()
    # if start_time
    # function_generator.set_frequency(np.random.choice([1.8, 1.9, 2.0, 2.1]))

    if tre:
        #frame = (cv2.resize(frame, size)).astype("uint8")
        _, frame = resize_and_crop_frame(frame, 0, 0, frame.shape[1], frame.shape[0], *size)
        current_time = time.time() - start_time
        if saving_frames:
            cv2.imwrite(f'/media/m4/Mahmoud_T7_Touch/Sweep/frame_{current_time}_{function_generator.get_frequency()}.png', frame)
        
        # if track:
        #     frame_cleaned = plot_cluster_on_image_blue(segmented, frame, 80)
        #     frame = tracker.track(frame_cleaned)

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
        elif e == increase_frequency:
            function_generator.set_frequency(function_generator.get_frequency()/1000000 + 0.1)
            print(f"Frequency: {function_generator.get_frequency()}")
        elif e == decrease_frequency:
            function_generator.set_frequency(function_generator.get_frequency()/1000000 - 0.1)
            print(f"Frequency: {function_generator.get_frequency()}")
        elif e == increase_amplitude:
            function_generator.set_vpp(function_generator.get_vpp() + 1.0)
            print(f"Amplitude: {function_generator.get_vpp()}")
        elif e == decrease_amplitude:
            function_generator.set_vpp(function_generator.get_vpp() - 1.0)
            print(f"Amplitude: {function_generator.get_vpp()}")
        elif e == start_frequency_sweep:
            generate_frequency_sweep(start_frequency, end_frequency, 0.2, step)
        
        
        if P1 == 0:
            cv2.circle(frame, (int(size[0]/2), 50), 30, (0, 0, 0), -1)
        elif P1 == 1:
            cv2.circle(frame, (int(size[0]/2), 50), 30, (255, 255, 255), -1)
        if P2 == 0:
            cv2.circle(frame, (int(size[0]/2 + 50), 100), 30, (0, 0, 0), -1)
        elif P2 == 1:
            cv2.circle(frame, (int(size[0]/2 + 50), 100), 30, (255, 255, 255), -1)
        if P3 == 0:
            cv2.circle(frame, (int(size[0]/2 -50), 100), 30, (0, 0, 0), -1)
        elif P3 == 1:
            cv2.circle(frame, (int(size[0]/2 -50), 100), 30, (255, 255, 255), -1)
        if P4 == 0:
            cv2.circle(frame, (int(size[0]/2), 150), 30, (0, 0, 0), -1)
        elif P4 == 1:
            cv2.circle(frame, (int(size[0]/2), 150), 30, (255, 255, 255), -1)

        cv2.imshow('Video', frame)
        i += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()



