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

# COM="COM8"
# COM="COM13" # 完全BLE接続にする場合、トランシーバー用基板に送信
# COM="COM11" # 完全BLE接続にする場合、トランシーバー用基板に送信
COM="COM17" # 完全BLE接続にする場合、トランシーバー用基板に送信

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
# run_comp = "dnnrt result RUNNING"
# run_comp = "dnnrt result WALKING"
run_comp_running = "Value = 0x02"
run_comp_walking = "Value = 0x01"
run_comp_stable = "Value = 0x00"

# 組み合わせで起動
# run_comp = "Hop Step Jump Process"




# ScreenShot
# LLMに定型文入力（気軽に検索）

# デバイス操作
# Mode 0 : カメラ
# Mode 1 : ショートカットキー操作で直接画面操作 & 検索

# Web検索
# Mode 3 : GoogleMap周辺レストラン検索
# Mode 4 : GoogleMap経路検索
# Mode 4 : Google検索 & ガイダンス
# Mode 5 : 音声認識 & ChatGPT & ガイダンス

Mode = 7 # 8


DETECT_COUNT = 0

while True:

    L_RecieveData=ser.readline()
    RecieveData = L_RecieveData.decode()
    print(RecieveData)
    
    # if run_comp_running in RecieveData:


    # if (run_comp_running in RecieveData): #  or (run_comp_walking in RecieveData): # RUNNING or WALKING で実行
    if (run_comp_walking in RecieveData): # デモ動画用

        
        print("**********")
        # print("Detected RUNNING !!!!!")
        # print("Detected COMBO !!!!!")
        # print("Detected WALKING !!!!!")
        print("Detected Gesture !!!!!")
        DETECT_COUNT += 1

        if DETECT_COUNT > 3: # 20:
            if Mode == 0:
                subprocess.run(['C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/python.exe', 'C:/Users/0107409377/Desktop/code/AtCoder/src/DemoApp/Search_and_LLM/Device_Contorol/Camera.py'])
            if Mode == 1:
                subprocess.run(['C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/python.exe', 'C:/Users/0107409377/Desktop/code/AtCoder/src/DemoApp/Search_and_LLM/search.py'])
            
            
            if Mode == 3:
                subprocess.run(['C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/python.exe', 'C:/Users/0107409377/Downloads/API-App/PlaceAPI.py'])
            if Mode == 4:
                subprocess.run(['C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/python.exe', 'C:/Users/0107409377/Downloads/API-App/DirectionsAPI.py'])
            if Mode == 5:
                subprocess.run(['C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/python.exe', 'C:/Users/0107409377/Desktop/code/AtCoder/src/DemoApp/Search_and_LLM/Google_serch_Guidance.py'])
            
            # LLM Only
            if Mode == 6:
                subprocess.run(['C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/python.exe', 'C:/Users/0107409377/Desktop/code/AtCoder/src/DemoApp/Search_and_LLM/ChatGPT.py'])
            # LLM Schedule
            if Mode == 7:
                subprocess.run(['C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/python.exe', 'C:/Users/0107409377/Desktop/code/AtCoder/src/DemoApp/Search_and_LLM/Schedule_and_Outlook_mask_data.py'])
                # ここでパワポにまとめたり
                # メールに添付して送信
                # のコードを記述する？ -> 上記のコード内に記述すればテキストを引き継げる
                # subprocess.run(['C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/python.exe', 'C:/Users/0107409377/Desktop/code/AtCoder/src/DemoApp/Search_and_LLM/Outlook_Schedule/PowerPoint.py'])
            # LLM Route Guidance(経路検索は別の経路案内APIから取得予定)
            if Mode == 8:
                subprocess.run(['C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/python.exe', 'C:/Users/0107409377/Desktop/code/AtCoder/src/DemoApp/Search_and_LLM/RouteSearch_API/LLM_RouteGuidance.py']) # C:/Users/0107409377/Desktop/code/AtCoder/src/DemoApp/Spotify_API/LLM_RouteGuidance.py'])
            
            # Spotify
            # if Mode == 8:
            #     print("********** Mode 8 **********")
            #     if run_comp_running in RecieveData: # HOUSE MUSIC を再生
            #         subprocess.run(['C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/python.exe', 'C:/Users/0107409377/Desktop/code/AtCoder/src/DemoApp/Spotify_API/ActionDetection_Play.py', 'RUNNING'])
            #     if run_comp_walking in RecieveData: # J-POP を再生
            #         subprocess.run(['C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/python.exe', 'C:/Users/0107409377/Desktop/code/AtCoder/src/DemoApp/Spotify_API/ActionDetection_Play.py', 'WALKING'])
            
            # DETECT_COUNT = 0
            # break
            # time.sleep(3)
            for i in range(100): # 残っている検出結果をリリースする
                L_RecieveData=ser.readline()
                RecieveData = L_RecieveData.decode()
                print(RecieveData)
            DETECT_COUNT = 0

            break
    
    # "今はRUNNINGとWALKINGの時しか送信していないので送信できない"
    # # Spotify一時停止用に追加
    # if run_comp_stable in RecieveData: # HOUSE MUSIC を再生
    #     subprocess.run(['C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/python.exe', 'C:/Users/0107409377/Desktop/code/AtCoder/src/DemoApp/Spotify_API/ActionDetection_Play.py', 'STABLE'])

    # ***** シリアル通信テスト *****
    # send_data = "Detected_COMBO_2024"
    # send_serial(send_data)
    # time.sleep(1)
    # read_data = receive_serial()
    # print("read data : ", read_data)
    # time.sleep(1)
    # ***** シリアル通信テスト *****
 
ser.close()