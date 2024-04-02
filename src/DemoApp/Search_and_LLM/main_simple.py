import serial
import time
import signal
import threading

from rdflib import Graph

# TTLファイルのパスを指定してグラフを作成
g = Graph()
# g.parse("path/to/your/file.ttl", format="turtle")

import subprocess
# import serialstream
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
    print(Input_File)
    print("=====")

    # ファイルの読み込み
    # open = open("test_text.txt", "r")

    # 変数に代入
    data = Input_File.read()

    # ファイルを閉じる
    Input_File.close()

    # 出力
    # print(data)
    ser.write(("TEST 2024/03/27").encode())
    
    # while(1):
    if ser !='':
        data=input(data) # +'\r\n'
        data=data.encode() # 'utf-8')
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


# ***** シリアル通信テスト *****
import traceback
# データ送信関数
def send_serial(cmd): # self, cmd): 

    # print("send data : {0}".format(cmd))
    try:
        # 改行コードを必ずつけるために、1回削除して、送信時に再度付与する
        cmd = cmd.rstrip()
            # 改行コードを付与　文字列をバイナリに変換して送信
        # self.uartport.write((cmd + "\n").encode("utf-8"))
        ser.write((cmd + "\n").encode("utf-8"))
        # ser.write((" Input Data : " + cmd + "\n").encode("utf-8"))
    except serial.SerialException:
        print(traceback.format_exc())
# データ受信関数
def receive_serial() : # self):

    try:
        # rcvdata = self.uartport.readline()
        rcvdata = ser.readline()
    except serial.SerialException:
        print(traceback.format_exc())

    # 受信したバイナリデータを文字列に変換　改行コードを削除
    return rcvdata.decode("utf-8").rstrip() 
# ***** シリアル通信テスト *****


# 起動を簡単に
run_comp = "dnnrt result RUNNING"
# 組み合わせで起動
# run_comp = "Hop Step Jump Process"


# Mode 0 : 周辺レストラン検索
# Mode 1 : LLMに定型文入力（気軽に検索）
# Mode 2 : 音声認識 & chatGPT # ScreenShot

Mode = 2 # 1 # 0


DETECT_COUNT = 0

while True:

    L_RecieveData=ser.readline()
    RecieveData = L_RecieveData.decode()
    print(RecieveData)
    
    if run_comp in RecieveData:
        print("**********")
        # print("Detected RUNNING !!!!!")
        print("Detected COMBO !!!!!")
        DETECT_COUNT += 1

        if DETECT_COUNT > 20:
            if Mode == 0:
                subprocess.run(['C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/python.exe', 'c:/Users/0107409377/Downloads/API-App/PlaceAPI.py'])
            if Mode == 1:
                subprocess.run(['C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/python.exe', 'c:/Users/0107409377/Desktop/code/AtCoder/src/DemoApp/Search_and_LLM/search.py'])
            if Mode == 2:
                subprocess.run(['C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/python.exe', 'c:/Users/0107409377/Desktop/code/AtCoder/src/DemoApp/Search_and_LLM/chatGPT.py'])
            break

    # ***** シリアル通信テスト *****
    send_data = "Detected_COMBO_2024"
    send_serial(send_data)
    time.sleep(1)
    read_data = receive_serial()
    print("read data : ", read_data)
    time.sleep(1)
    # ***** シリアル通信テスト *****
 
ser.close()