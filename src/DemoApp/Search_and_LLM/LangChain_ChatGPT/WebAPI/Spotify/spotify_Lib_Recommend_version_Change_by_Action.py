# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import strip

# #出発駅の入力
# departure_station = input("出発駅を入力してください：")
# #到着駅の入力
# destination_station = input("到着駅を入力してください：")

from langchain.tools.base import BaseTool

class MusicPlayback(): # BaseTool): # BaseToolの記述がなくても動く
    
    # def run(self, action_detection): # オプションの引数ありバージョン
    # def run(self): # 提案デモ用
    def run(self, playback_mode): # 提案デモ用

        # ***** 変更点（Playbackモード指定用）*****
        if "LinkedActionsMode" in playback_mode:
        # ***** 変更点（Playbackモード指定用）*****
        
            import serial
            import time
            import signal
            import threading
            import subprocess
            # import serialstream
            signal.signal(signal.SIGINT, signal.SIG_DFL)

            # COM="COM8"
            # COM="COM13" # 完全BLE接続にする場合、トランシーバー用基板に送信
            # COM="COM11" # 完全BLE接続にする場合、トランシーバー用基板に送信

            # COM="COM17" # 完全BLE接続にする場合、トランシーバー用基板に送信
            COM="COM16" # 完全BLE接続にする場合、ドングル基板に送信
            # COM="COM8"

            bitRate=115200

            ser = serial.Serial(COM, bitRate, timeout=0.1)


            # 起動を簡単に
            detect_running = "Value = 0x02"
            detect_walking = "Value = 0x01"
            detect_stable = "Value = 0x00"
            # detect_running = "RUNNING"
            # detect_walking = "WALKING"
            # detect_stable = "STABLE"

            # 組み合わせで起動
            # detect = "Hop Step Jump Process"

            Mode = 9 # 8 # 7 # 6 # 2 # 1 # 0


            DETECT_COUNT = 0
            Pre_RecieveData = detect_running # stable

            # STABLE_COUNT = 0

            # 次回Todo # Done
            # detect_headgesture = "0x05" # "0x04"
            detect_headgesture_horizontal = "0x05" # "0x04"

            while True:

                L_RecieveData=ser.readline()
                RecieveData = L_RecieveData.decode()
                print(RecieveData)

                # 次回Todo (Done)
                if (detect_headgesture_horizontal in RecieveData):
                    print("\n\n**********\nHead Gesture Detection!\n**********")
                    print("楽曲再生を終了します。")
                    # 終了時に楽曲を一時停止する場合
                    subprocess.run(['C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/python.exe', 'C:/Users/0107409377/Desktop/code/AtCoder/src/DemoApp/Spotify_API/ActionDetection_Play.py', 'STABLE'])
                    return "Success"
                
                # if detect_running in RecieveData:
                if (detect_running in RecieveData) or (detect_walking in RecieveData) or (detect_stable in RecieveData): # RUNNING or WALKING で実行
                    print("**********")
                    # print("Detected RUNNING !!!!!")
                    # print("Detected COMBO !!!!!")
                    # print("Detected WALKING !!!!!")
                    print("Detect : ", RecieveData)

                    # 追加
                    # if (detect_stable in RecieveData):
                    #     STABLE_COUNT += 1
                    #     if STABLE_COUNT > 30:
                    #         break
                    
                    if RecieveData in Pre_RecieveData:
                        # # 追加
                        # """
                        # 2024/6/12 次回Todo
                        # # または、ユーザーがジェスチャーモードにして、ジェスチャーを検出したら終了する
                        # SDKのジェスチャー検出の際の送信するGATTデータを「0x04, 0x05」のようにActionDetectionで使っていないものにする。
                        # if (detect_headgesture in RecieveData):
                        #     return "Success"
                        # # 上に記載
                        # """
                        # # 今はSTABLEが一定回数繰り返しで終了
                        # if (detect_stable in RecieveData):
                        #     STABLE_COUNT += 1
                        #     print("STABLE COUNT : ", STABLE_COUNT)
                        #     if STABLE_COUNT > 10: # 30:
                        #         # break
                        #         return "Success" # これを書くとAgentが処理が成功したことを認識して処理が終了する
                        
                        for i in range(100): # 残っている検出結果をリリースする
                            L_RecieveData=ser.readline()
                            RecieveData = L_RecieveData.decode()
                            print(RecieveData)
                        DETECT_COUNT = 0
                    else:
                        DETECT_COUNT += 1
                        # STABLE_COUNT = 0
                        
                    if DETECT_COUNT > 1: # 5: # 20: # 時間でもいいかも
                        
                        # Spotify
                        if Mode == 9:
                            print("********** Mode : Spotify **********")

                            # 同じ状態が継続しているか判別するために1つ前に検出された状態を格納する
                            Pre_RecieveData = RecieveData
                            
                            # Spotify一時停止用に追加
                            if detect_stable in RecieveData:
                                subprocess.run(['C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/python.exe', 'C:/Users/0107409377/Desktop/code/AtCoder/src/DemoApp/Spotify_API/ActionDetection_Play.py', 'STABLE'])
                                # return
                                # break
                            # HOUSE MUSIC を再生
                            if detect_running in RecieveData:
                                subprocess.run(['C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/python.exe', 'C:/Users/0107409377/Desktop/code/AtCoder/src/DemoApp/Spotify_API/ActionDetection_Play.py', 'RUNNING'])
                            # J-POP を再生
                            if detect_walking in RecieveData:
                                subprocess.run(['C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/python.exe', 'C:/Users/0107409377/Desktop/code/AtCoder/src/DemoApp/Spotify_API/ActionDetection_Play.py', 'WALKING'])
                        
                        # DETECT_COUNT = 0
                        # break
                        # time.sleep(3)
                        for i in range(100): # 残っている検出結果をリリースする
                            L_RecieveData=ser.readline()
                            RecieveData = L_RecieveData.decode()
                            print(RecieveData)
                        DETECT_COUNT = 0
        
        # ***** 変更点（Playbackモード指定用）*****
        elif "HouseMusicMode" in playback_mode:
            import serial
            import time
            import signal
            import threading
            import subprocess
            subprocess.run(['C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/python.exe', 'C:/Users/0107409377/Desktop/code/AtCoder/src/DemoApp/Spotify_API/ActionDetection_Play.py', 'RUNNING'])
            # return "Success"

            signal.signal(signal.SIGINT, signal.SIG_DFL)
            COM="COM16" # 完全BLE接続にする場合、ドングル基板に送信
            bitRate=115200
            ser = serial.Serial(COM, bitRate, timeout=0.1)

            detect_headgesture_horizontal = "0x05" # "0x04"

            while True:

                L_RecieveData=ser.readline()
                RecieveData = L_RecieveData.decode()
                print(RecieveData)

                if (detect_headgesture_horizontal in RecieveData):
                    print("\n\n**********\nHead Gesture Detection!\n**********")
                    print("楽曲再生を終了します。")
                    # 終了時に楽曲を一時停止する場合
                    subprocess.run(['C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/python.exe', 'C:/Users/0107409377/Desktop/code/AtCoder/src/DemoApp/Spotify_API/ActionDetection_Play.py', 'STABLE'])
                    return "Success"
            
        elif "RelaxMusicMode" in playback_mode:
            import serial
            import time
            import signal
            import threading
            import subprocess
            subprocess.run(['C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/python.exe', 'C:/Users/0107409377/Desktop/code/AtCoder/src/DemoApp/Spotify_API/ActionDetection_Play.py', 'WALKING'])
            # return "Success"

            signal.signal(signal.SIGINT, signal.SIG_DFL)
            COM="COM16" # 完全BLE接続にする場合、ドングル基板に送信
            bitRate=115200
            ser = serial.Serial(COM, bitRate, timeout=0.1)
            
            detect_headgesture_horizontal = "0x05" # "0x04"

            while True:

                L_RecieveData=ser.readline()
                RecieveData = L_RecieveData.decode()
                print(RecieveData)

                if (detect_headgesture_horizontal in RecieveData):
                    print("\n\n**********\nHead Gesture Detection!\n**********")
                    print("楽曲再生を終了します。")
                    # 終了時に楽曲を一時停止する場合
                    subprocess.run(['C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/python.exe', 'C:/Users/0107409377/Desktop/code/AtCoder/src/DemoApp/Spotify_API/ActionDetection_Play.py', 'STABLE'])
                    return "Success"
        else:
            print("該当するPlaybackモードはありません。終了します。")
            return "Success"
        # ***** 変更点（Playbackモード指定用）*****