import serial
import time
import signal
import threading
import subprocess
# import serialstream

signal.signal(signal.SIGINT, signal.SIG_DFL)
COM="COM16" # 完全BLE接続にする場合、トランシーバー用基板に送信
bitRate=115200
ser = serial.Serial(COM, bitRate, timeout=0.1)


# 起動を簡単に
# run_comp = "dnnrt result RUNNING"
# run_comp = "dnnrt result WALKING"
run_comp_running = "Value = 0x02"
run_comp_walking = "Value = 0x01"
run_comp_stable = "Value = 0x00"


DETECT_COUNT = 0
Pre_RecieveData = run_comp_running # stable

# stable（終了）カウント
Stable_ProcessEnd = 0

while True:

    L_RecieveData=ser.readline()
    RecieveData = L_RecieveData.decode()
    print(RecieveData)
    
    # if run_comp_running in RecieveData:
    if (run_comp_running in RecieveData) or (run_comp_walking in RecieveData) or (run_comp_stable in RecieveData): # RUNNING or WALKING で実行
        print("**********")
        # print("Detected RUNNING !!!!!")
        # print("Detected COMBO !!!!!")
        # print("Detected WALKING !!!!!")
        print("Detect : ", RecieveData)
        if RecieveData in Pre_RecieveData:
            
            for i in range(100): # 残っている検出結果をリリースする
                L_RecieveData=ser.readline()
                RecieveData = L_RecieveData.decode()
                print(RecieveData)
            DETECT_COUNT = 0
        else:
            DETECT_COUNT += 1
            

        if DETECT_COUNT > 1: # 5: # 20: # 時間でもいいかも
            
            # Spotify
            print("********** Mode : Spotify **********")

            # 同じ状態が継続しているか判別するために1つ前に検出された状態を格納する
            Pre_RecieveData = RecieveData
            
            # Spotify一時停止用に追加
            if run_comp_stable in RecieveData:
                subprocess.run(['C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/python.exe', 'C:/Users/0107409377/Desktop/code/AtCoder/src/DemoApp/Spotify_API/ActionDetection_Play.py', 'STABLE'])
                Stable_ProcessEnd += 1
                print("STABLE検出回数: ", Stable_ProcessEnd)
                # Pre_RecieveData = run_comp_running # 前の検出と被らせないため
                # if Stable_ProcessEnd > 3:
                break
            # HOUSE MUSIC を再生
            if run_comp_running in RecieveData:
                subprocess.run(['C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/python.exe', 'C:/Users/0107409377/Desktop/code/AtCoder/src/DemoApp/Spotify_API/ActionDetection_Play.py', 'RUNNING'])
            # J-POP を再生
            if run_comp_walking in RecieveData:
                subprocess.run(['C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/python.exe', 'C:/Users/0107409377/Desktop/code/AtCoder/src/DemoApp/Spotify_API/ActionDetection_Play.py', 'WALKING'])
            
            # DETECT_COUNT = 0
            # break
            # time.sleep(3)
            for i in range(100): # 残っている検出結果をリリースする
                L_RecieveData=ser.readline()
                RecieveData = L_RecieveData.decode()
                print(RecieveData)
            DETECT_COUNT = 0
 
ser.close()