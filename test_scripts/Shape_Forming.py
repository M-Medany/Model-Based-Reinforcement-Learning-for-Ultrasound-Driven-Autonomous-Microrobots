from curtsies import Input
import time
import cv2
from utils.image_postprocessing import create_binary_bitmap, plot_cluster_on_image_blue, resize_and_crop_frame
from utils.tracking_CSRT import CSRT_tracker
from utils.segmentation import ImageSegmentation
from utils.actuator import Arduino, FunctionGenerator_1
import yaml
import numpy as np
import os
import serial
import threading

# Serial communication setup
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
time.sleep(2)  # Wait for Arduino to initialize

save_dir = "Shapeforming"

# Ensure the directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def send_serial_command(command):
    """Send a command to the Arduino."""
    try:
        print(f"Sending command: {command}")
        ser.write(f"{command}\n".encode())  # Send ASCII-encoded command with newline
        time.sleep(0.2)  # Allow Arduino time to process

        if ser.in_waiting > 0:
            response = ser.readline().decode().strip()
            print(f"Arduino response: {response}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    track = True
    closeness_factor = 3.0
    passing_factor = 0.3

    # Load configuration
    with open(f'scripts/config.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Initialize function generator
    fgen = FunctionGenerator_1(config=config, instrument_descriptor=config['Tektronix_settings']['INSTR_DESCRIPTOR'])
    print("Function generator initialized successfully.")

    cap = cv2.VideoCapture(0)
    image_counter = 0

    fgen.set_frequency(2.8)
    fgen.set_vpp(12.0)
    start_time = time.time()

    # Skip the first 10 frames for stabilization
    print("Skipping the first 10 frames...")
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame during initialization. Exiting.")
            return

    # Select ROI
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
    CROSS_BOTTLENECK = 2
    EXIT_BOTTLENECK = 3
    
    state = NORMAL
    last_move = None

    if track:
        frame = frame[y:y+h, x:x+w]
        size = (640, 480)
        _, frame = resize_and_crop_frame(frame, 0, 0, frame.shape[1], frame.shape[0], *size)
        good_enough = False
        mask_in = None
        segmentation = ImageSegmentation(frame, **config['sam_config'])
        while not good_enough:
            segmented, mask_in = create_binary_bitmap(frame, segmentation=segmentation, mask_in=mask_in)
            segmented_copy = segmented.copy()
            frame_cleaned = plot_cluster_on_image_blue(segmented, frame, 80)
            cv2.imshow("Blue Image", frame_cleaned)
            cv2.imshow("Processed Image", segmented * 255)
            cv2.waitKey(0)
            answer = input("Good Enough? [y|N] ")
            if answer.lower() == "y":
                good_enough = True
        tracker = CSRT_tracker(initial_image=frame_cleaned)

    bottleneck_center = find_bottleneck_center(frame, segmented_copy, plotting=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = frame[y:y+h, x:x+w]
        _, frame = resize_and_crop_frame(frame, 0, 0, frame.shape[1], frame.shape[0], *size)
        current_time = time.time() - start_time

        if track:
            frame_cleaned = plot_cluster_on_image_blue(segmented, frame, 90)
            frame, success = tracker.track(frame_cleaned)

            if not success:
                print("Tracker lost! Reinitializing...")
                tracker = CSRT_tracker(initial_image=frame_cleaned)
                continue

            bubble_location = tracker.get_agent_location()
            bubble_area = tracker.get_bubble_area()

            if bubble_location is not None:
                distance_to_bottleneck = np.linalg.norm(bubble_location - bottleneck_center, ord=2)
                bubble_radius = np.sqrt(bubble_area / np.pi)

                # NORMAL State
                if state == NORMAL:
                    if distance_to_bottleneck < bubble_radius * closeness_factor:
                        print("Approaching bottleneck")
                        obstacle_side = "right" if bubble_location[0] < bottleneck_center[0] else "left"
                        state = APPROACH_BOTTLENECK

                # APPROACH_BOTTLENECK State
                elif state == APPROACH_BOTTLENECK:
                    if distance_to_bottleneck < bubble_radius * passing_factor:
                        print(f"Crossing bottleneck from {obstacle_side}")
                        state = CROSS_BOTTLENECK

                # CROSS_BOTTLENECK State
                elif state == CROSS_BOTTLENECK:
                    if obstacle_side == "right":
                        send_serial_command('4')  # Move left
                    elif obstacle_side == "left":
                        send_serial_command('3')  # Move right
                    if distance_to_bottleneck > bubble_radius * passing_factor * 20:
                        print("Exited bottleneck. Returning to NORMAL state.")
                        state = NORMAL

                # Movement Logic
                if state == NORMAL:
                    if distance_to_bottleneck > bubble_radius * passing_factor:
                        if bubble_location[0] < bottleneck_center[0]:
                            send_serial_command('2')  # Move right
                            last_move = "right"
                        elif bubble_location[0] > bottleneck_center[0]:
                            send_serial_command('1')  # Move left
                            last_move = "left"

            else:
                print("No bubble found")

            time.sleep(0.1)



         # Display and save frame
        frequency = fgen.get_frequency()
        amplitude = fgen.get_vpp()
        text_frequency = f"Freq: {frequency:.2f} Hz | Amp: {amplitude:.2f} V | Time: {current_time:.2f} s"
        text_state = f"State: {['NORMAL', 'APPROACH_BOTTLENECK', 'CROSS_BOTTLENECK'][state]}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (0, 255, 0)
        thickness = 2
        position_frequency = (10, 30)
        position_state = (10, 60)  # Position below the frequency text

        cv2.putText(frame, text_frequency, position_frequency, font, font_scale, font_color, thickness, cv2.LINE_AA)
        text_state = f"State: {['NORMAL', 'APPROACH_BOTTLENECK', 'CROSS_BOTTLENECK'][state]}"
        cv2.putText(frame, text_state, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        frame_filename = os.path.join(save_dir, f"tracked_output_{image_counter:04d}.png")
        cv2.imwrite(frame_filename, frame)
        image_counter += 1

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

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
        cols = segmented_copy[:, col_start:col_end]  # Extract the group of columns
        white_pixels = np.sum(cols)  # Count the number of white pixels

        if (min_white_pixels is None or white_pixels < min_white_pixels) and white_pixels > 3:
            min_white_pixels = white_pixels
            bottleneck_position = (col_start + col_end) // 2
            rows, cols_indices = np.where(cols == 1)
            if len(rows) > 0:
                bottleneck_center_row = np.mean(rows)
                bottleneck_center = (bottleneck_position, int(bottleneck_center_row))

    print(f"Bottleneck at column {bottleneck_position} with {min_white_pixels} white pixels")

    if plotting:
        print(f"Bottleneck center at pixel coordinates: {bottleneck_center}")
        cv2.circle(frame, bottleneck_center, 30, (0, 0, 255), -1)  # Draw a red circle at the bottleneck
        cv2.imshow("Bottleneck", frame)
        bottleneck_image_filename = os.path.join(save_dir, "bottleneck_center.png")
        cv2.imwrite(bottleneck_image_filename, frame)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

    return bottleneck_center

if __name__ == "__main__":
    main()
