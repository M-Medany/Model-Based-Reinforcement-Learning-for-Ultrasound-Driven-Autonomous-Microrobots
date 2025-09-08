import win32api
import cv2
from time import sleep
from pyfirmata import Arduino
# import pyvisa as visa
import utils.tektronix_func_gen as tfg
# import time, logging, os, struct, sys
import xlsxwriter
import pymmcore
import time
import os 

path = (rf"F:\Experiment_manual_control_8_11_2023\exp{time.time()}")
os.mkdir(path)
os.mkdir(rf"{path}\img")

workbook=xlsxwriter.Workbook(rf"{path}\actuators.xlsx")
worksheet=workbook.add_worksheet('experiment_08_11_2023')
worksheet.write('A1','Frequency')
worksheet.write('B1','Amplitude')
worksheet.write('C1','Piezo1')
worksheet.write('D1','Piezo2')
worksheet.write('E1','Piezo3')
worksheet.write('F1','Piezo4')
worksheet.write('G1', 'Time')
rowIndex =2
# alternatively fgen.ch1.print_settings() to show from one channel only
# fgen.print_settings()
# #using PYVISA we can setup the AFG3000 to be configured over USB in this example
# rm = visa.ResourceManager()
# print(rm.list_resources())
#
# instrumentdescriptor = 'USB0::0x0699::0x034F::C020081::INSTR' #example of your USB connected device
# AFG3000 = rm.open_resource(instrumentdescriptor)
#
# #ID = AFG3000.ask('*IDN?')
# #print(ID)
#
# AFG3000.write('*RST') #reset AFG
#
# #Filename for TFW to be read in from the PC
# # filelocation = '"AFGWaveformfile.tfw"'
# # wfm_magic_name = "TEKAFG3000"
# # wfm_version_check = '20050114'
# i=5
#     #setup output1
# AFG3000.write('source1:function user1') #sets the AFG source to user1 memory
# AFG3000.write('source1:Frequency 240E3') #set frequency to 20KHz
# AFG3000.write('source1:voltage:amplitude,' + str(i)) #sets voltage of CH1 to 2 Volts
# AFG3000.write('output1:state ON') #turns on output 1
# # self.AFG3000.set_amplitude(i)
#     #setup output2
# AFG3000.write('source2:function user1') #sets the AFG source to user1 memory
# AFG3000.write('source2:Frequency 20E3') #set frequency to 20KHz
# AFG3000.write('source2:voltage:amplitude 2') #sets voltage of CH1 to 2 Volts
# AFG3000.write('output2:state ON') #turns on output 2


port=('com5')
pin9 = 10
pin1 = 9
pin2 = 10
pin3 = 11
pin4 =12
# pin2 = 8
# pin3 = 7
# pin4 = 6
pin5 = 5
pin6 = 4
pin7 = 3
pin8 = 2

board=Arduino(port)
i=3
j=1.8e6
x1=0
x1=0
x2=0
x3=0
x4=0
x5=0
x6=0
x7=0
x8=0
x9=0
P1 = 0
P2 = 0
P3 = 0
P4 = 0
P5 = 0
P6 = 0
P7 = 0
P8 = 0
P9 = 0

# Hammamatsu settings
# EXPOSURE_TIME = 10 # Exposure time Hammamatsu
# IMG_SIZE = 500
EXPOSURE_TIME = 30 # Exposure time Hammamatsu
IMG_SIZE = 500
P1, P2, P3, P4 = 0, 0, 0, 0

size = (IMG_SIZE, IMG_SIZE)
counter = 1
# Initialize core object
mmc = pymmcore.CMMCore()
i = 0
# Find camera config --> get camera
mmc.setDeviceAdapterSearchPaths(["C:/Program Files/Micro-Manager-2.0gamma"])
mmc.loadSystemConfiguration('C:/Program Files/Micro-Manager-2.0gamma/mmHamamatsu.cfg')
label = mmc.getCameraDevice()
mmc.setExposure(EXPOSURE_TIME)
mmc.getLoadedDevices()
mmc.snapImage()
img = mmc.getImage()  # img - it's just numpy array

# cv2.namedWindow('Video')

mmc.startContinuousSequenceAcquisition(1)
start_time = time.time()


with tfg.FuncGen('USB0::0x0699::0x034F::C020081::INSTR') as fgen:
    fgen.ch1.set_function("SIN")
    fgen.ch1.set_frequency(j, unit="Hz")
    # fgen.ch1.set_offset(0, unit="mV")
    fgen.ch1.set_amplitude(i)
    fgen.ch1.set_output("ON")
    fgen.ch2.set_output("OFF")

    while True:
        print("--- %s seconds ---" % (time.time() - start_time))
        if mmc.getRemainingImageCount() > 0:
            frame = mmc.getLastImage()
            # img = mmc.getlastImage()
            # frame = mmc.popNextImage()
            frame = (cv2.resize(frame, size) / 256).astype("uint8")
            # cv2.resizeWindow('Video', 300, 300)
            current_time = time.time() - start_time
            # i=i+0.01
            # fgen.ch1.set_amplitude(i)
            a=win32api.GetKeyState(0x01)#left click
            b=win32api.GetKeyState(0x02)#righ t click
            c=win32api.GetKeyState(0x04)#mouse Forward
            d=win32api.GetKeyState(0x05)#mouse backward
            a1= win32api.GetKeyState(0x25)  # lef tarrow
            b1= win32api.GetKeyState(0x26)  # Up arrow
            c1= win32api.GetKeyState(0x27)  #right arrow
            d1= win32api.GetKeyState(0x28)  #dwon arrow
            a2= win32api.GetKeyState(0x21)  # up key + amplitude
            b2= win32api.GetKeyState(0x22)  # down key - amplitude
            c2= win32api.GetKeyState(0x23)  # End key + frequency
            d2= win32api.GetKeyState(0x24)  # Home key - frequency
            a3= win32api.GetKeyState(0x6B)  # up key + amplitude
            b3= win32api.GetKeyState(0x6D)  # down key - amplitude
            c3= win32api.GetKeyState(0x6A)  # End key + frequency
            d3= win32api.GetKeyState(0x6F)  # Home key - frequency

            piezo_1 = win32api.GetKeyState(0x31)
            piezo_2 = win32api.GetKeyState(0x32)
            piezo_3 = win32api.GetKeyState(0x33)
            piezo_4 = win32api.GetKeyState(0x34)
            piezo_5 = win32api.GetKeyState(0x35)
            piezo_6 = win32api.GetKeyState(0x36)
            piezo_7 = win32api.GetKeyState(0x37)
            piezo_8 = win32api.GetKeyState(0x38)
            piezo_1_off = win32api.GetKeyState(0x61)
            piezo_2_off = win32api.GetKeyState(0x62)
            piezo_3_off = win32api.GetKeyState(0x63)
            piezo_4_off = win32api.GetKeyState(0x64)
            piezo_5_off = win32api.GetKeyState(0x65)
            piezo_6_off = win32api.GetKeyState(0x66)
            piezo_7_off = win32api.GetKeyState(0x67)
            piezo_8_off = win32api.GetKeyState(0x68)



            if piezo_1 < 0:
                P1=1
                print("toggel1")
                board.digital[pin1].write(1)
                # worksheet.write('C' + str(rowIndex), P1)
            elif piezo_2 < 0:
                P2 = 1
                print("turn on 2" )
                board.digital[pin2].write(1)
                # worksheet.write('D' + str(rowIndex), P2)
            elif piezo_3 < 0:
                P3 = 1
                print("turn on 3" )
                board.digital[pin3].write(1)
                # worksheet.write('E' + str(rowIndex), P3)
            elif piezo_4 < 0:
                P4 = 1
                print("turn on 4" )
                # worksheet.write('F' + str(rowIndex), P4)
                board.digital[pin4].write(1)
            elif piezo_5 < 0:

                print("turn on 5" )
                board.digital[pin5].write(1)
            elif piezo_6 < 0:

                print("turn on 6" )
                board.digital[pin6].write(1)
            elif piezo_7 < 0:

                print("turn on 7" )
                board.digital[pin7].write(1)
            elif piezo_8 < 0:

                print("toggel 8")
                board.digital[pin8].write(1)
            elif piezo_1_off < 0:
                P1 = 0
                # worksheet.write('C' + str(rowIndex), P1)
                print("toggel1")
                board.digital[pin1].write(0)
            elif piezo_2_off < 0:
                P2 = 0
                # worksheet.write('C' + str(rowIndex), P2)
                print("turn on 2")
                board.digital[pin2].write(0)
            elif piezo_3_off < 0:
                P3 = 0
                # worksheet.write('C' + str(rowIndex), P3)
                print("turn on 3")
                board.digital[pin3].write(0)
            elif piezo_4_off < 0:
                P4 = 0
                # worksheet.write('C' + str(rowIndex), P4)
                print("turn on 4")
                board.digital[pin4].write(0)
            elif piezo_5_off < 0:

                print("turn on 5")
                board.digital[pin5].write(0)
            elif piezo_6_off < 0:

                print("turn on 6")
                board.digital[pin6].write(0)
            elif piezo_7_off < 0:

                print("turn on 7")
                board.digital[pin7].write(0)
            elif piezo_8_off < 0:

                print("toggel 8")
                board.digital[pin8].write(0)

            # if a1<0:
            #     print("Left")
            #     board.digital[pin1].write(1)
            #     # AFG3000.write('source1:Frequency 20E3')  # set frequency to 20KHz
            #
            # elif b1< 0:
            #     board.digital[pin2].write(1)
            #     print("up")
            # elif c1< 0:
            #     board.digital[pin3].write(1)
            #     print("right")
            # elif d1< 0:
            #     board.digital[pin4].write(1)
            #     print("down")
            #     i = i + 0.01
            #     # AFG3000.write('source1:voltage:amplitude i')  # sets voltage of CH1
            #     fgen.ch1.set_amplitude(i)
            #     print(i)
            elif a3 <0:
                i=i+0.01
                fgen.ch1.set_amplitude(i)
            elif b3 < 0:
                i=i-0.01
                # AFG3000.write('source1:voltage:i')  # sets voltage of CH1
                fgen.ch1.set_amplitude(i)
                print(i)
            elif c3 < 0:
                j=j+ 50
                # AFG3000.write('source1:Frequency j')  # set frequency
                fgen.ch1.set_frequency(j, unit="Hz")
            elif d3 < 0:
                j= j-50
                # AFG3000.write('source1:Frequency j')  # set frequency
                fgen.ch1.set_frequency(j, unit="Hz")
        
        #     board.digital[pin1].write(0)
        #     board.digital[pin2].write(0)
        #     board.digital[pin3].write(0)
        #     board.digital[pin4].write(0)
            j=fgen.ch1.get_frequency()
            i=fgen.ch1.get_amplitude()
            worksheet.write('A' + str(rowIndex), j)
            worksheet.write('B' + str(rowIndex), i)
            worksheet.write('C' + str(rowIndex), P1)
            worksheet.write('D' + str(rowIndex), P2)
            worksheet.write('E' + str(rowIndex), P3)
            worksheet.write('F' + str(rowIndex), P4)
            worksheet.write('G' + str(rowIndex), current_time)
            cv2.imshow('Video', frame)
            time.sleep(0.1)
            # plt.imshow(frame, cmap='gray')
            # plt.show()
            # print(f"{size}")
            cv2.imwrite(rf'{path}\img\img_' + str(rowIndex) + '.png', frame)
            # plt.imshow(img, cmap='gray')
            # plt.imshow(img)
            # plt.show()
            # plt.close(frame)

            if cv2.waitKey(1) == 27:
                break

            rowIndex += 1

workbook.close()