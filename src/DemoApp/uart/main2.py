import serial
import time
import signal
import threading
import subprocess

import serialstream
signal.signal(signal.SIGINT, signal.SIG_DFL)

COM="COM8"
bitRate=115200
 
ser = serial.Serial(COM, bitRate, timeout=0.1)
ser_2 = serialstream.SerialStream(port=COM, monitor=True)
Input_File = open('CXD3277-R-025-024_AutoStartOff_LogOn.ttl', 'r')

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
        subprocess.run(['C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/python.exe', 'c:/Users/0107409377/Downloads/API-App/PlaceAPI.py'])
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

    """
    文字列を待つ。
    ※ただしwaitForを連続した場合は正しい動きが期待できないので
    必ず何かしらをwriteしてから次のwaitForを実行する。
    """
    # ser.waitFor("login")
    """
    文字列を送る、改行文字も忘れずに。
    どうしても煩わしい場合はSerialStreamのwriteを修正すれば良い。
    業務にて改行が必要ないケースが多々あったためこのような仕様。
    """
    # ser.write("admin\n")
    # ser.waitFor("Password")
    # ser.write("admin\n")
    """
    複数パターンの文字列が期待される場合はタプルで渡す。
    ヒットした文字列が返ってくるので処理を分ける。
    """
    # if ser_2.waitFor((">", "Login incorrect")) == ">":
    """
    タイムアウトを指定して文字列を待つ場合このように処理
    """
    #if ser.waitFor("hoge", timeout=3) == False:
    #    print("hogeきません！")
    # ser_2.write("terminal length 0\n")
    # ser_2.waitFor(">")
    """
    好きなタイミングでログファイルへの書き込みを開始・終了できる。
    """
    if RecieveData in "nsh>":
        ser_2.startLog(logfile="C:/Users/0107409377/Desktop/code/AtCoder/src/DemoApp/uart/CXD3277-R-025-024_AutoStartOff_LogOn.ttl") # "log.txt")
        # ser_2.write("show run\n")
        # ser_2.waitFor("BaseApp::Run") # >")
        # ser_2.closeLog()
        
    """
    waitForでヒットした行に含まれる文字列が欲しい場合取り出すことが出来る
    （個体シリアルやデバイスナンバーを取得したい場合など）
    ※しかし、リアルタイム性が必要ないのであれば書き出したログから取得するほうが
    スマートなのでそちらを推奨。
    """
    print(ser_2.getBuffer())
 
ser.close()