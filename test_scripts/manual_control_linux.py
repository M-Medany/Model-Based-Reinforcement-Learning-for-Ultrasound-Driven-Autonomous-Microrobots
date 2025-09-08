from curtsies import Input
import time
import cv2
from utils.image_postprocessing import create_binary_bitmap, plot_cluster_on_image_blue, resize_and_crop_frame
from utils.tracking_CSRT import CSRT_tracker
from utils.segmentation import ImageSegmentation
from utils.actuator import Arduino, FunctionGenerator_1
from environments.game_env_8_actions import PIEZO_DIRECTIONS8
import yaml


def main(): 
    track = False

    with open(f'scripts/config.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    board=Arduino(config['Arduino_settings']['SERIAL_PORT_UBUNUTU'], baudrate=config['Arduino_settings']['BAUDRATE_ARDUINO'], config=config)
    print("Arduino initialized successfully")

    fgen = FunctionGenerator_1(config=config, instrument_descriptor=config['Tektronix_settings']['INSTR_DESCRIPTOR'])
    print("Function generator initialized successfully")

    piezo = PIEZO_MANUAL(board, fgen)

    x1, x2, x3, x4, P1, P2, P3, P4 = 0, 0, 0, 0, 0, 0, 0, 0

    cap = cv2.VideoCapture(0)

    fgen.set_frequency(2.0)
    fgen.set_vpp(12.0)

    if track:
        while True:
            tre, frame = cap.read()
            cv2.imshow('ROI', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        tre, frame = cap.read()

        good_enough = False
        mask_in = None
        segmentation = ImageSegmentation(frame, **config['sam_config'])
        while not good_enough:
            segmented, mask_in = create_binary_bitmap(frame, manual=True, mask_in=mask_in)
            frame_cleaned = plot_cluster_on_image_blue(segmented, frame, 80)
            cv2.imshow("Blue Image", frame_cleaned)
            cv2.imshow("Processed Image 2", segmented*255)
            cv2.waitKey(0)
            answer = input("Good Enough???? [y|N] ")
            if answer.lower() == "y":
                good_enough = True
        tracker = CSRT_tracker(initial_image=frame_cleaned)


    size = (1080, 720)
    start_time = time.time()

    while True:
        tre, frame = cap.read()

        if tre:
            #frame = (cv2.resize(frame, size)).astype("uint8")
            _, frame = resize_and_crop_frame(frame, 0, 0, frame.shape[1], frame.shape[0], *size)
            # frame = cv2.rotate(frame, cv2.ROTATE_180)
            current_time = time.time() - start_time

            if track:
                frame_cleaned = plot_cluster_on_image_blue(segmented, frame, 80)
                frame = tracker.track(frame_cleaned)

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
            
    
if __name__ == "__main__":
    main()