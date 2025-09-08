import time
import numpy as np
import binascii
#import tektronix_func_gen as tfg
from utils.costum_fgen import FuncGen
import yaml
from pathlib import Path
import serial

class Actuator(object):
    id = None
    position = [0, 0, 0]

    def __init__(self, frequency, amplitude, config_path) -> None:
        
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        """Initialize the actuator with the given frequency and amplitude"""

        self.frequency = frequency
        self.amplitude = amplitude

#TODO: check piezo numbering
class ArduinoActuator(object):

    def __init__(self):

        # Initiate contact with arduino
        self.arduino = serial.Serial(port=self.config['SERIAL_PORT_ARDUINO'], baudrate=self.config['BAUDRATE_ARDUINO'])
        time.sleep(2)  # Give serial communication time to establish

        # Initiate status of 4 piëzo transducers
        self.arduino.write(b'0')  # Turn all outputs to LOW

    def move(self, action: int):
        self.arduino.write(f"{action}".encode())

    def close(self):
        self.arduino.write(b'0')


# Arduino board access
class Arduino():

    def __init__(self, port, baudrate, config):
        self.config = config
        # Initiate contact with arduino
        self.arduino = serial.Serial(port=port, baudrate=baudrate)
        time.sleep(2)  # Give serial communication time to establish

        # Initiate status of 4 piëzo transducers
        self.arduino.write(b'0')  # Turn all outputs to LOW

    def set_piezo_from_action(self, action: int):
        act_num = action % self.config['Action_space_settings']['NUMBER_PIEZOS']
        act_num += 1 # Arduino starts counting at !!!!!!1!!!!!!
        self.arduino.write(f"{act_num}".encode())
        return act_num

    def set_piezo_after_collision(self, safe_piezo: int):
        self.arduino.write(f"{safe_piezo}".encode()) # This is already indexed from 1

    def set_piezo_by_number(self, piezo: int):
        self.arduino.write(f"{piezo}".encode())

    def close(self):
        self.arduino.write(b'0')


#TODO: check function generator type
class FunctionGenerator_1(object):
    
    def __init__(self, config: dict, instrument_descriptor: str = None):
        self.AFG3000 = FuncGen(instrument_descriptor).ch1
        self.config = config
        self.reset()
        self.turn_on()

    def reset(self):
        vpp = self.config['Tektronix_settings']['DEFAULT_VPP']
        freq = self.config['Tektronix_settings']['DEFAULT_FREQUENCY']
        freq *= 1e6
        self.AFG3000.set_amplitude(float(vpp)) #in V
        self.AFG3000.set_frequency(float(freq), unit='Hz')
        self.set_waveform('SQUARE') # try with square wave


    def set_vpp(self, vpp: float):
        self.AFG3000.set_amplitude(vpp)

    def set_vpp_from_action(self, action: int):
        action //= (self.config['Action_space_settings']['NUMBER_PIEZOS'] * self.config['Action_space_settings']['NUMBER_FREQUENCIES'])
        ampl_index = action % self.config['Action_space_settings']['NUMBER_AMPLITUDES']
        ampl_value = float(self.config['Action_space_settings']['MIN_AMPLITUDE'] + (ampl_index * (self.config['Action_space_settings']['MAX_AMPLITUDE'] - self.config['Action_space_settings']['MIN_AMPLITUDE']) / (self.config['Action_space_settings']['NUMBER_AMPLITUDES'] - 1)))
        self.AFG3000.set_amplitude(amplitude=ampl_value)
        return ampl_value

    def get_vpp(self):
        return self.AFG3000.get_amplitude()

    def set_frequency(self, frequency: float):
        frequency *= 1e6
        self.AFG3000.set_frequency(frequency, unit='Hz')
    
    def set_frequency_from_action(self, action: int):
        action //= self.config['Action_space_settings']['NUMBER_PIEZOS']
        freq_index = action % self.config['Action_space_settings']['NUMBER_FREQUENCIES']
        # Calculate the actual frequency value based on the index
        freq_value = float(self.config['Action_space_settings']['MIN_FREQUENCY'] + (freq_index * (self.config['Action_space_settings']['MAX_FREQUENCY'] - self.config['Action_space_settings']['MIN_FREQUENCY']) / (self.config['Action_space_settings']['NUMBER_FREQUENCIES'] - 1)))
        freq_value *= 1e6
        self.AFG3000.set_frequency(freq=freq_value, unit='Hz') 
        return freq_value
    
    def set_sweep_mode(self):
        self.AFG3000.set_run_mode('SWEep')

    def set_fixed_mode(self):
        self.AFG3000.set_run_mode('FIXed')
    
    def set_sweep_limits(self, low: float, high: float, unit: str = 'Hz'):
        self.AFG3000.set_sweep_frequency(low, high, unit=unit)

    def get_frequency(self):
        return self.AFG3000.get_frequency()

    def set_waveform(self, waveform: str):
        assert waveform in ['SIN', 'SQUARE', 'RAMP'], f'Invalid waveform: {waveform}'
        self.AFG3000.set_function(waveform)

    def get_waveform(self):
        return self.AFG3000.get_function()

    def turn_on(self):
        self.AFG3000.set_output("ON")

    def turn_off(self):
        self.AFG3000.set_output(0) # maybe this works?

class ActuatorPiezos:

    def __init__(self, port, baudrate):

        # Initiate contact with arduino
        self.arduino = serial.Serial(port, baudrate)
        # print(f"Arduino: {self.arduino.readline().decode()}")
        time.sleep(2)  # Give serial communication time to establish

        # Initiate status of 4 piëzo transducers
        self.arduino.write(b'H')  # Turn all outputs to LOW

    #---------------maybe add a distinct turn off?
    def move(self, action: int):

        if action == -1:
            return

        # print(f"Arduino: {self.arduino.readline().decode()}")
        # self.arduino.write(b'H')  # Turn old piezo off
        self.arduino.write(f"{action}".encode())  # Turn new piezo on

    def close(self, action: int):
        self.arduino.write(f"{action}".encode())
        # self.arduino.write(b'H')

#---------------Do we need this as in do we want to move the microscope?
# class TranslatorLeica:

#     def __init__(self, port, baudrate):

#         # Open Leica
#         self.observer = serial.Serial(port=port,
#                                       baudrate=baudrate,  # Baudrate has to be 9600
#                                       timeout=2)  # 2 seconds timeout recommended by the manual
#         print(f"Opened port {port}: {self.observer.isOpen()}")
#         self.pos = (0, 0)  # Reset position to (0, 0)

#     def reset(self):
#         self.observer.write(bytearray([255, 82]))  # Reset device
#         time.sleep(2)  # Give the device time to reset before issuing new commands

#     def close(self):
#         self.observer.close()
#         print(f"Closed serial connection")

#     def get_status(self, motor: int):  # TODO --> use check_status to make sure the motor does not get command while still busy
#         assert motor in [0, 1, 2]  # Check if device number is valid
#         self.observer.write(bytearray([motor, 63, 58]))  # Ask for device status
#         received = self.observer.read(1)  # Read response (1 byte)
#         if received:
#             print(f"Received: {received.decode()}")  # Print received message byte
#         time.sleep(SLEEP_TIME)

#     def write_target_pos(self, motor: int, target_pos: int):

#         # Translate coordinate to a 3-byte message
#         msg = self.coord_to_msg(int(target_pos))
#         self.observer.write(bytearray([motor, 84, 3, msg[0], msg[1], msg[2], 58]))  # [device number, command, 3, message, stop signal]
#         time.sleep(SLEEP_TIME)

#     def get_motor_pos(self, motor):

#         # Ask for position
#         self.observer.write(bytearray([motor, 97, 58]))

#         # Initialise received variable
#         received = b''

#         # For timeout functionality
#         t0 = time.time()

#         # Check for received message until timeout
#         while not received:
#             received = self.observer.read(3)  # Read response (3 bytes)
#             if received == b'\x00\x00\x00':
#                 return 0
#             if time.time() - t0 >= SLEEP_TIME:  # Check if it takes longer than a second
#                 print(f"No bytes received: {received}")
#                 return 0

#         # Return translated message
#         translated = self.msg_to_coord(received)
#         # self.observer.write(bytearray([71, 58]))
#         # self.reset()
#         time.sleep(SLEEP_TIME)

#         return translated

#     def get_target_pos(self, motor: int):

#         # Ask for position
#         self.observer.write(bytearray([motor, 116, 58]))

#         # Initialise received variable
#         received = b''

#         # For timeout functionality
#         t0 = time.time()

#         # Check for received message until timeout
#         while not received:
#             received = self.observer.read(3)  # Read response (3 bytes)
#             if received == b'\x00\x00\x00':
#                 return 0
#             if time.time() - t0 >= SLEEP_TIME:  # Check if it takes longer than a second
#                 print(f"No bytes received: {received}")
#                 return 0

#         # Return translated message
#         translated = self.msg_to_coord(received)
#         time.sleep(SLEEP_TIME)

#         return translated
#     def move_to_target(self, motor: int, target_pos: int):  # TODO --> Coordinate boundaries so we don't overshoot the table and get the motor stuck, as we need to reset then

#         """
#         100k steps is 1 cm in real life000000000000000
#         """

#         # Write target position
#         self.write_target_pos(motor=motor, target_pos=target_pos)

#         # Move motor to target coordinate
#         self.observer.write(bytearray([motor, 71, 58]))

#         # Give motor time to move
#         time.sleep(SLEEP_TIME)

#     def coord_to_msg(self, coord: int):

#         # Convert from two's complement
#         if np.sign(coord) == -1:
#             coord += 16777216

#         # Convert to hexadecimal and pad coordinate with zeros of necessary (len should be 6 if data length is 3)
#         hexa = hex(coord).split("x")[-1].zfill(6)

#         # Get the four digit binary code for each hex value
#         four_digit_binaries = [bin(int(n, 16))[2:].zfill(4) for n in hexa]

#         # Convert to 3-byte message
#         eight_digit_binaries = [f"{four_digit_binaries[n] + four_digit_binaries[n + 1]}".encode() for n in range(0, 6, 2)][::-1]

#         return [int(m, 2) for m in eight_digit_binaries]

#     def msg_to_coord(self, msg: str):

#         # Read LSB first
#         msg = msg[::-1]

#         # Convert incoming hex to readable hex
#         hexa = binascii.hexlify(bytearray(msg))

#         # Convert hex to decimal
#         coord = int(hexa, 16)

#         # Convert to negative coordinates if applicable
#         if bin(coord).zfill(24)[2] == '1':
#             coord -= 16777216
#         return int(coord)

#     def pixels_to_increment(self, pixels: np.array):
#         return np.array([pixels[0]*PIXEL_MULTIPLIER_LEFT_RIGHT, pixels[1]*PIXEL_MULTIPLIER_UP_DOWN])

#     def move_increment(self, offset_pixels: np.array):

#         # Get increments and add to current position
#         self.pos += self.pixels_to_increment(pixels=offset_pixels)

#         # Move motors x and y
#         print("Moving...")
#         self.move_to_target(motor=1, target_pos=self.pos[0])  # Move left/right
#         self.move_to_target(motor=2, target_pos=self.pos[1])  # Move up/down
#         time.sleep(3)  # TODO --> check optimal sleep time


# class FunctionGenerator_2:

    # def __init__(self, instrument_descriptor=INSTR_DESCRIPTOR):
    #     self.AFG3000 = tfg.FuncGen(instrument_descriptor).ch1

    # def reset(self, vpp=1, frequency=1):

    #     self.set_vpp(vpp=vpp)
    #     self.set_frequency(frequency=frequency)
    #     self.set_waveform('SQUARE')
    #     self.turn_on()

    #     print(f'FG settings: {self.AFG3000.get_settings()}')

    # def set_vpp(self, vpp: float):
    #     self.AFG3000.set_amplitude(vpp)

    # def get_vpp(self):
    #     return self.AFG3000.get_amplitude()

    # def set_frequency(self, frequency: float):
    #     self.AFG3000.set_frequency(frequency * 1e3)

    # def get_frequency(self):
    #     return self.AFG3000.get_frequency()

    # def set_waveform(self, waveform: str):
    #     assert waveform in ['SIN', 'SQUARE', 'RAMP'], f'Invalid waveform: {waveform}'
    #     self.AFG3000.set_function(waveform)

    # def get_waveform(self):
    #     return self.AFG3000.get_function()

    # def turn_on(self):
    #     self.AFG3000.set_output("ON")

    # def turn_off(self):
    #     self.AFG3000.set_output("OFF")

def close_arduino():
    with open(f'scripts/config.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    arduino=Arduino(config['Arduino_settings']['SERIAL_PORT_UBUNUTU'], baudrate=config['Arduino_settings']['BAUDRATE_ARDUINO'], config=config)
    arduino.set_piezo_by_number(0)


if __name__ == '__main__':

    with open(f'scripts/config.yaml', 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)


    board=Arduino(config['Arduino_settings']['SERIAL_PORT_UBUNUTU'], baudrate=config['Arduino_settings']['BAUDRATE_ARDUINO'], config=config)

    counter_0 = 0
    counter_1 = 0
    counter_2 = 0
    counter_3 = 0
    counter_4 = 0


    for _ in range(1000):
        action = np.random.randint(1, config['Action_space_settings']['TOTAL_ACTIONS'])
        piezo = board.set_piezo(action)
        if piezo == 0:
            counter_0 += 1
            print(f"Action: {action}, Piezo: {piezo}")
        elif piezo == 1:
            counter_1 += 1
            print(f"Action: {action}, Piezo: {piezo}")
        elif piezo == 2:
            counter_2 += 1
            print(f"Action: {action}, Piezo: {piezo}")
        elif piezo == 3:
            counter_3 += 1
            print(f"Action: {action}, Piezo: {piezo}")
        elif piezo == 4:
            counter_4 += 1
            print(f"Action: {action}, Piezo: {piezo}")

    print(f"Counter 0: {counter_0}")
    print(f"Counter 1: {counter_1}")
    print(f"Counter 2: {counter_2}")
    print(f"Counter 3: {counter_3}")
    print(f"Counter 4: {counter_4}")