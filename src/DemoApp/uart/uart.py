


# import serial
# ser = serial.Serial("COM8",115200)
# print(ser)

# # import serial
# from serial.tools import list_ports

# ser = serial.Serial()
# devices = [info.device for info in list_ports.comports()]
# print(devices)

import serial
import time
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

#File Open
File = open('response.txt', 'w')
# File = open('CXD3277-R-025-024_AutoStartOff_LogOn.ttl', 'r') 

#Open SerialDevice
SerDevice = serial.Serial('COM8',115200) # serial.Serial('/dev/ttyS0', 115200, timeout=10)

#Set Command
#'GetVersion' という文字列のコマンドを出す
Command = "GetVersion\r"

#Send Command
SerDevice.write(Command.encode())

# Input_File = open('CXD3277-R-025-024_AutoStartOff_LogOn.ttl', 'r')
# SerDevice.write(Input_File) # .encode())


#******************************
#Recieve Response
while True:
    #Read data one Line (top->'\r\n')
    L_RecieveData=SerDevice.readline()
    RecieveData = L_RecieveData.decode()

    #Get Data Length
    DataLength = len(L_RecieveData)

    #If DataLength = 2, then Response END!!
    #1行の文字列が2文字があることが受信データの終わりという条件にした

    if DataLength==5: break
    else:
        File.writelines(RecieveData)
        print(RecieveData)
    # File.writelines(RecieveData)
    # print(RecieveData)

#******************************
#Close Serial Device
SerDevice.close()
#File Close
File.close()
