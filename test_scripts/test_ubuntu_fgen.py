# import tektronix_func_gen as tfg
import pyvisa


rm = pyvisa.ResourceManager()

print(rm.list_resources())



with tfg.FuncGen('USB0::0x0699::0x034F::C020081::INSTR') as fgen:
    # fgen.ch1.set_function("SIN")
    # fgen.ch1.set_frequency(100000, unit="Hz")
    # # fgen.ch1.set_offset(0, unit="mV")
    # fgen.ch1.set_amplitude(1)
    # fgen.ch1.set_output("ON")
    # fgen.ch2.set_output("OFF")
    pass
# example_basic_control('ASRL2::INSTR')

# rm = pyvisa.ResourceManager()

# print(rm.list_resources())