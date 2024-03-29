import serial
import time
import signal
import threading
signal.signal(signal.SIGINT, signal.SIG_DFL)

COM="COM8"
bitRate=115200
 
ser = serial.Serial(COM, bitRate, timeout=0.1)

#----------------------
# データを送信
#----------------------
def serial_write():
    # global ser
    Input_File = open('CXD3277-R-025-024_AutoStartOff_LogOn.ttl', 'r')
    
    # while(1):
    if ser !='':
        data=input(Input_File)+'\r\n'
        data=data.encode('utf-8')
        ser.write(data)

#----------------------
# データを受信
#----------------------
def serial_read():
    global ser
    # while(1):
    if ser !='':
        #data=ser.read(1)
        data=ser.readline()
        data=data.strip()
        data=data.decode('utf-8')
        print(data)


 
run_comp = "dnnrt result RUNNING"
run_comp = "Hop Step Jump Process"

while True:
 
    # time.sleep(0.1)
    # result = ser.read_all()
    # result = ser.readline()
    # print(result)
    L_RecieveData=ser.readline()
    RecieveData = L_RecieveData.decode()
    print(RecieveData)
    
    # if result == b'\r': # <Enter>で終了
    # if result == b'':
    # l_in = [s for s in result if run_str in s]
    
    if run_comp in RecieveData: # result: # l_in:

        print("**********")
        # print("Detected RUNNING !!!!!")
        print("Detected COMBO !!!!!")
        break
    
    # # print('program end')
    # # ser.write(b'test!!!!!')
    # data=input(Input_File)+'\r\n'
    # data=data.encode('utf-8')
    # ser.write(data)W

    # thread_1 = threading.Thread(target=serial_write)
    # thread_2 = threading.Thread(target=serial_read)

    # thread_1.start()
    # thread_2.start()
 
ser.close()