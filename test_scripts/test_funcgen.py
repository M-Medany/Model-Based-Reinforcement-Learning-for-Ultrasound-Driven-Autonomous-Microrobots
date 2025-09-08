from utils.actuator import Arduino, FunctionGenerator_1
from utils.costum_fgen import FuncGen
import yaml
import numpy as np
import time
import pyvisa

#board = Arduino('/dev/ttyACM0', baudrate=115200)

rm = pyvisa.ResourceManager()

print(rm.list_resources())

with open(f'scripts/config.yaml', 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)


time.sleep(4)
function_generator = FunctionGenerator_1(config=config, instrument_descriptor=config['Tektronix_settings']['INSTR_DESCRIPTOR'])
print("Function generator initialized successfully")

#function_generator.turn_on()
function_generator.set_waveform("SIN")

time.sleep(3)

# action = np.random.randint(1, config['Action_space_settings']['TOTAL_ACTIONS'])
# print('action', action)
# freq = function_generator.set_frequency_from_action(action)
# print('freq', freq)



for _ in range(20):

    action = np.random.randint(1, config['Action_space_settings']['TOTAL_ACTIONS'])
    print('action', action)
    freq = function_generator.set_frequency_from_action(action)
    print('freq', freq)
    vpp = function_generator.set_vpp_from_action(action)
    print('vpp', vpp)
    time.sleep(1)


function_generator.turn_off()

