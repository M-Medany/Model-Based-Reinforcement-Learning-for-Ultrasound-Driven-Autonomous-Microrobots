from curtsies import Input
import time
import cv2
from utils.image_postprocessing import create_binary_bitmap, plot_cluster_on_image_blue, resize_and_crop_frame
from utils.tracking_CSRT import CSRT_tracker
from utils.segmentation import ImageSegmentation
from utils.actuator import Arduino, FunctionGenerator_1
from environments.game_env_8_actions import PIEZO_DIRECTIONS8
import yaml
import numpy as np
import os


save_dir = "Shapeforming"

# Ensure the directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def main(): 
    track = True
    closeness_factor = 1.5

    with open(f'scripts/config.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    board=Arduino(config['Arduino_settings']['SERIAL_PORT_UBUNUTU'], baudrate=config['Arduino_settings']['BAUDRATE_ARDUINO'], config=config)
    print("Arduino initialized successfully")

    fgen = FunctionGenerator_1(config=config, instrument_descriptor=config['Tektronix_settings']['INSTR_DESCRIPTOR'])
    print("Function generator initialized successfully")

    piezo = PIEZO_MANUAL(board, fgen)

    x1, x2, x3, x4, P1, P2, P3, P4 = 0, 0, 0, 0, 0, 0, 0, 0

    cap = cv2.VideoCapture(0)

    fgen.set_frequency(2.8)
    fgen.set_vpp(12.0)
    for _ in range(10):
        tre, frame = cap.read()


    if track:
        # while True:
        #     tre, frame = cap.read()
        #     cv2.imshow('ROI', frame)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        tre, frame = cap.read()
        size = (1080, 720)
        _, frame = resize_and_crop_frame(frame, 0, 0, frame.shape[1], frame.shape[0], *size)
        good_enough = False
        mask_in = None
        segmentation = ImageSegmentation(frame, **config['sam_config'])
        while not good_enough:
            segmented, mask_in = create_binary_bitmap(frame, segmentation=segmentation, mask_in=mask_in)
            segmented_copy = segmented.copy()
            frame_cleaned = plot_cluster_on_image_blue(segmented, frame, 80)
            cv2.imshow("Blue Image", frame_cleaned)
            cv2.imshow("Processed Image 2", segmented*255)
            frame_filename = os.path.join(save_dir, "Processed_Image_2.png")
            cv2.imwrite(frame_filename, frame_cleaned)
            cv2.waitKey(0)
            answer = input("Good Enough???? [y|N] ")
            if answer.lower() == "y":
                good_enough = True
        tracker = CSRT_tracker(initial_image=frame_cleaned)

    print("amount of white pixels: ", np.sum(segmented_copy))
    print("shape of segmented_copy: ", segmented_copy.shape)
    print("rows, cols with 1: ", np.where(segmented_copy == 1))
    print("rows, cols with 0: ", np.where(segmented_copy == 0))
    print("rows, cols with 255: ", np.where(segmented_copy == 255))

    bottleneck_center = find_bottleneck_center(frame, segmented_copy, plotting=True) # enable plotting if you want to do a visual check


    start_time = time.time()
    last_move = None
    while True:
        tre, frame = cap.read()
        print("frame shape: ", frame.shape)
        if tre:
            #frame = (cv2.resize(frame, size)).astype("uint8")
            _, frame = resize_and_crop_frame(frame, 0, 0, frame.shape[1], frame.shape[0], *size)
            # frame = cv2.rotate(frame, cv2.ROTATE_180)
            current_time = time.time() - start_time

            if track:
                frame_cleaned = plot_cluster_on_image_blue(segmented, frame, 80)
                frame, success = tracker.track(frame_cleaned)
            
            if track:
                bubble_location = tracker.get_agent_location()
                bubble_area = tracker.get_bubble_area()
                print("bubble location: ", bubble_location)
                print("botleneck center: ", bottleneck_center)
                # Case one: bubble is left to the bottleneck: Move right

                if bubble_location is not None:
                    if np.linalg.norm(bubble_location - bottleneck_center, ord=2) < np.sqrt(bubble_area/np.pi)*closeness_factor:
                        print("Bottle neck reached")
                        # turn off up and down piezo and take the last mocev
                        if last_move == "right":
                            fgen.set_vpp(14.0)
                            piezo(PIEZO_MANUAL.up)
                            piezo(PIEZO_MANUAL.down)
                            piezo(PIEZO_MANUAL.right)
                            fgen.set_vpp(12.0)
                        elif last_move == "left":
                            fgen.set_vpp(14.0)
                            piezo(PIEZO_MANUAL.up)
                            piezo(PIEZO_MANUAL.down)
                            piezo(PIEZO_MANUAL.left)
                            fgen.set_vpp(12.0)

                        # TODO: Add the correct piezo direction here
                        # TODO: FOR MAHMOUD
                    elif bubble_location[0] < bottleneck_center[0]:
                        print("Move right")
                        piezo(PIEZO_MANUAL.up_right)
                        last_move = "right"
                    elif bubble_location[0] > bottleneck_center[0]:
                        print("Move left")
                        piezo(PIEZO_MANUAL.up_left)
                        last_move = "left"
                else:
                    print("No bubble found")





        with Input(keynames='curses') as input_generator:
            e = input_generator.send(0.01)
            piezo(e)
            
            # if P1 == 0:
            #     cv2.circle(frame, (int(size[0]/2), 50), 30, (0, 0, 0), -1)
            # elif P1 == 1:
            #     cv2.circle(frame, (int(size[0]/2), 50), 30, (255, 255, 255), -1)
            # if P2 == 0:
            #     cv2.circle(frame, (int(size[0]/2 + 50), 100), 30, (0, 0, 0), -1)
            # elif P2 == 1:
            #     cv2.circle(frame, (int(size[0]/2 + 50), 100), 30, (255, 255, 255), -1)
            # if P3 == 0:
            #     cv2.circle(frame, (int(size[0]/2 -50), 100), 30, (0, 0, 0), -1)
            # elif P3 == 1:
            #     cv2.circle(frame, (int(size[0]/2 -50), 100), 30, (255, 255, 255), -1)
            # if P4 == 0:
            #     cv2.circle(frame, (int(size[0]/2), 150), 30, (0, 0, 0), -1)
            # elif P4 == 1:
            #     cv2.circle(frame, (int(size[0]/2), 150), 30, (255, 255, 255), -1)

            cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()


class PIEZO_MANUAL:
    off = "0"
    up = "8"
    down = "2"
    right = "6"
    left = "4"
    down_right = "3"
    down_left = "1"
    up_right = "9"
    up_left = "7"
    increase_frequency = "+"
    decrease_frequency = "-"
    increase_amplitude = "*"
    decrease_amplitude = "/"
    set_sweep_mode = "s"
    set_fixed_mode = "f"
    
    def __init__(self, board: Arduino, fgen: FunctionGenerator_1):
        self.board = board
        self.fgen = fgen
    
    def convert(self, key):
        if key == self.off:
            return PIEZO_DIRECTIONS8.OFF
        elif key == self.up:
            return PIEZO_DIRECTIONS8.UP
        elif key == self.down:
            return PIEZO_DIRECTIONS8.DOWN
        elif key == self.right:
            return PIEZO_DIRECTIONS8.RIGHT
        elif key == self.left:
            return PIEZO_DIRECTIONS8.LEFT
        elif key == self.down_right:
            return PIEZO_DIRECTIONS8.DOWN_RIGHT
        elif key == self.down_left:
            return PIEZO_DIRECTIONS8.DOWN_LEFT
        elif key == self.up_right:
            return PIEZO_DIRECTIONS8.UP_RIGHT
        elif key == self.up_left:
            return PIEZO_DIRECTIONS8.UP_LEFT
        else:
            return None
    
    def __call__(self, key):
        piezo = self.convert(key)
        if piezo is not None:
            self.board.set_piezo_after_collision(piezo)
        elif key == self.set_sweep_mode:
            self.fgen.set_sweep_mode()
            low = float(input("Low frequency: "))*1000000
            high = float(input("High frequency: "))*1000000
            self.fgen.set_sweep_limits(low, high, "Hz")
            print("Sweep mode")
        elif key == self.set_fixed_mode:
            self.fgen.set_fixed_mode()
            print("Fixed mode")
        elif key == self.increase_frequency:
            self.fgen.set_frequency(self.fgen.get_frequency()/1000000 + 0.1)
            print(f"Frequency: {self.fgen.get_frequency()}")
        elif key == self.decrease_frequency:
            self.fgen.set_frequency(self.fgen.get_frequency()/1000000 - 0.1)
            print(f"Frequency: {self.fgen.get_frequency()}")
        elif key == self.increase_amplitude:
            self.fgen.set_vpp(self.fgen.get_vpp() + 1.0)
            print(f"Amplitude: {self.fgen.get_vpp()}")
        elif key == self.decrease_amplitude:
            self.fgen.set_vpp(self.fgen.get_vpp() - 1.0)
            print(f"Amplitude: {self.fgen.get_vpp()}")
            

    
#     return bottleneck_center # return in x, y where x width and y height from top left corner
def find_bottleneck_center(frame, segmented_copy, plotting=False, padding_factor=0.2):
    group_size = 10  # Number of columns to group together
    height, width = segmented_copy.shape

    # Calculate the padded search region
    left_bound = int(width * padding_factor)
    right_bound = int(width * (1 - padding_factor))
    
    min_white_pixels = None
    bottleneck_position = None
    bottleneck_center = None

    # Iterate through columns in groups, but only within the padded region
    for col_start in range(left_bound, right_bound, group_size):
        col_end = min(col_start + group_size, right_bound)
        # Extract the group of columns
        cols = segmented_copy[:, col_start:col_end]
        # Count the number of white pixels in these columns
        white_pixels = np.sum(cols)
        
        # Only update if we find fewer white pixels than current minimum but more than a small threshold
        if (min_white_pixels is None or white_pixels < min_white_pixels) and white_pixels > 3:
            min_white_pixels = white_pixels
            bottleneck_position = (col_start + col_end) // 2
            # Compute the center of white pixels in these columns
            rows, cols_indices = np.where(cols == 1)
            if len(rows) > 0:
                bottleneck_center_row = np.mean(rows)
                bottleneck_center = (bottleneck_position, int(bottleneck_center_row))

    print(f"Bottleneck at column {bottleneck_position} with {min_white_pixels} white pixels")
    
    if plotting:
        print(f"Bottleneck center at pixel coordinates: {bottleneck_center}")
        # close all cv2 windows first
        cv2.destroyAllWindows()
        cv2.circle(frame, bottleneck_center, 30, (0, 0, 255), -1)  # Draw a red circle at the bottleneck
        cv2.imshow("Bottleneck", frame)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
    
    return bottleneck_center  # return in (x, y) where x is width and y is height from top-left corner

if __name__ == "__main__":
    main()