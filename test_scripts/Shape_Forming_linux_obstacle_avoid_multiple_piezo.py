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
import serial

ser = serial.Serial('/dev/ttyACM0', 115200)  # Replace with your port
time.sleep(2)  # Wait for 2 seconds to allow the Arduino to reset

save_dir = "Shapeforming"

# Ensure the directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def main(): 
    track = True
    closeness_factor = 1.5
    passing_factor = 0.1

    with open(f'scripts/config.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    board = Arduino(config['Arduino_settings']['SERIAL_PORT_UBUNUTU'], baudrate=config['Arduino_settings']['BAUDRATE_ARDUINO'], config=config)
    print("Arduino initialized successfully")

    fgen = FunctionGenerator_1(config=config, instrument_descriptor=config['Tektronix_settings']['INSTR_DESCRIPTOR'])
    print("Function generator initialized successfully")

    piezo = PIEZO_MANUAL(board, fgen)

    cap = cv2.VideoCapture(0)
    image_counter = 0

    fgen.set_frequency(2.8)
    fgen.set_vpp(12.0)
    start_time = time.time()

    # Skip the first 10 frames to allow stabilization
    print("Skipping the first 10 frames...")
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame during initialization. Exiting.")
            return

    # Allow user to select ROI
    print("Please select the ROI for the experiment.")
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture initial frame. Exiting.")
        return

    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")

    # Crop to ROI
    x, y, w, h = map(int, roi)
    print(f"Selected ROI: x={x}, y={y}, width={w}, height={h}")

    # FSM States
    NORMAL = 0
    APPROACH_BOTTLENECK = 1
    PASS_BOTTLENECK = 2
    CROSS_OBSTACLE = 3
    
    state = NORMAL
    last_move = None
    bottleneck_passed = False

    if track:
        frame = frame[y:y+h, x:x+w]
        size = (640, 480)  # Reduced frame size
        _, frame = resize_and_crop_frame(frame, 0, 0, frame.shape[1], frame.shape[0], *size)
        good_enough = False
        mask_in = None
        segmentation = ImageSegmentation(frame, **config['sam_config'])
        while not good_enough:
            segmented, mask_in = create_binary_bitmap(frame, segmentation=segmentation, mask_in=mask_in)
            segmented_copy = segmented.copy()
            frame_cleaned = plot_cluster_on_image_blue(segmented, frame, 80)
            cv2.imshow("Blue Image", frame_cleaned)
            cv2.imshow("Processed Image 2", segmented * 255)
            cv2.waitKey(0)
            answer = input("Good Enough???? [y|N] ")
            if answer.lower() == "y":
                good_enough = True
        tracker = CSRT_tracker(initial_image=frame_cleaned)

    bottleneck_center = find_bottleneck_center(frame, segmented_copy, plotting=True)

    while True:
        tre, frame = cap.read()
        if not tre:
            break

        frame = frame[y:y+h, x:x+w]  # Crop frame to ROI
        _, frame = resize_and_crop_frame(frame, 0, 0, frame.shape[1], frame.shape[0], *size)
        current_time = time.time() - start_time

        if track:
            frame_cleaned = plot_cluster_on_image_blue(segmented, frame, 80)
            frame, success = tracker.track(frame_cleaned)

        bubble_location = tracker.get_agent_location()
        bubble_area = tracker.get_bubble_area()

        if bubble_location is not None:
            distance_to_bottleneck = np.linalg.norm(bubble_location - bottleneck_center, ord=2)

            if state == NORMAL:
                if distance_to_bottleneck < np.sqrt(bubble_area / np.pi) * closeness_factor:
                    print("Approaching bottleneck")
                    state = APPROACH_BOTTLENECK

            elif state == APPROACH_BOTTLENECK:
                if distance_to_bottleneck < passing_factor * np.sqrt(bubble_area / np.pi):
                    print("Passing bottleneck")
                    state = PASS_BOTTLENECK

            elif state == PASS_BOTTLENECK:
                print("Crossing obstacle")
                state = CROSS_OBSTACLE
   
            elif state == CROSS_OBSTACLE:
                if last_move == "right":
                    ser.write(bytes([3]))  # Right piezo action
                elif last_move == "left":
                    ser.write(bytes([4]))  # Left piezo action

                # Continue tracking without delay
                if distance_to_bottleneck > passing_factor * np.sqrt(bubble_area / np.pi):
                    state = NORMAL  # Return to normal state

            # Movement logic
            if state != CROSS_OBSTACLE:
                if bubble_location[0] < bottleneck_center[0]:
                    print("Move right")
                    ser.write(bytes([2]))
                    last_move = "right"
                elif bubble_location[0] > bottleneck_center[0]:
                    print("Move left")
                    ser.write(bytes([1]))
                    last_move = "left"

        else:
            print("No bubble found")

        # Display and save frame with overlays
        frequency = fgen.get_frequency()
        amplitude = fgen.get_vpp()
        text = f"Freq: {frequency:.2f} Hz | Amp: {amplitude:.2f} V | Time: {current_time:.2f} s"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (0, 255, 0)
        thickness = 2
        position = (10, 30)
        cv2.putText(frame, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)

        frame_filename = os.path.join(save_dir, f"tracked_output_{image_counter:04d}.png")
        cv2.imwrite(frame_filename, frame)
        image_counter += 1

        # Display the video
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

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
            low = float(input("Low frequency: ")) * 1000000
            high = float(input("High frequency: ")) * 1000000
            self.fgen.set_sweep_limits(low, high, "Hz")
            print("Sweep mode")
        elif key == self.set_fixed_mode:
            self.fgen.set_fixed_mode()
            print("Fixed mode")
        elif key == self.increase_frequency:
            self.fgen.set_frequency(self.fgen.get_frequency() / 1000000 + 0.1)
            print(f"Frequency: {self.fgen.get_frequency()}")
        elif key == self.decrease_frequency:
            self.fgen.set_frequency(self.fgen.get_frequency() / 1000000 - 0.1)
        elif key == self.decrease_frequency:
            self.fgen.set_frequency(self.fgen.get_frequency() / 1000000 - 0.1)
            print(f"Frequency: {self.fgen.get_frequency()}")
        elif key == self.increase_amplitude:
            self.fgen.set_vpp(self.fgen.get_vpp() + 1.0)
            print(f"Amplitude: {self.fgen.get_vpp()}")
        elif key == self.decrease_amplitude:
            self.fgen.set_vpp(self.fgen.get_vpp() - 1.0)
            print(f"Amplitude: {self.fgen.get_vpp()}")

def find_bottleneck_center(frame, segmented_copy, plotting=False, padding_factor=0.1):
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
        cv2.destroyAllWindows()
        cv2.circle(frame, bottleneck_center, 30, (0, 0, 255), -1)  # Draw a red circle at the bottleneck
        cv2.imshow("Bottleneck", frame)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

    return bottleneck_center

if __name__ == "__main__":
    main()