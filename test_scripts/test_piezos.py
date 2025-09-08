import serial
import time
from settings import *
from pynput import keyboard
import datetime
# import tektronix_func_gen as tfg

# Initiate contact with arduino
arduino = serial.Serial(port=SERIAL_PORT_ARDUINO, baudrate=BAUDRATE_ARDUINO)
print(arduino.readline().decode())
time.sleep(1)  # give serial communication time to establish

# Initiate status of 4 piëzo transducers
status_channels = [False, False, False, False]
arduino.write(b"9")  # Turn all outputs to LOW

quit = False

def move(action):

    global status_channels

    arduino.write(b"9")  # Turn old piezo off
    arduino.write(f"{action}".encode())  # Turn new piezo on

    # Reset channels

    channels = [False, False, False, False]
    channels[action] = True
    status_channels = channels

def on_press(key):

    if key == keyboard.Key.esc:
        return False  # stop listener
    try:
        k = key.char  # single-char keys
    except:
        k = key.name  # other keys

    # Turn on/off right piezo
    if k == "a":
        arduino.write(b'2')

    # Turn on/off left piezo
    elif k == "d":
        arduino.write(b'0')

    # Turn on/off bottom piezo
    elif k == "s":
        arduino.write(b'1')

    # Turn on/off top piezo
    elif k == "w":
        arduino.write(b'3')

    elif k == "q":
        getExit()

    return False

def getExit():
    global quit
    arduino.write(b'9')
    quit = True

while not quit:

    # stream = VideoStream()
    # observer = Observer()

    # f_name = f"C:\\Users\\Matthijs\\PycharmProjects\\AI_actuated_microswarm_2\\Include\\snapshots\\smaller_channel_runs\\detect_n_clusters_env\\{datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')}-{str(round(time.time(), 2))}"


    listener = keyboard.Listener(on_press=on_press)
    listener.start()  # start to listen on a separate thread
    listener.join()  # remove if main thread is polling self.keys

    # stream.snap(f_name=f_name)