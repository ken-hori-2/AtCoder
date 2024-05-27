# Triggered by DNN detection sent by BLE
# BLEで送信されたDNN検出でトリガー


import subprocess
import serial
import signal
# COM="COM17" # 完全BLE接続にする場合、トランシーバー用基板に送信
# bitRate=115200
# ser = serial.Serial(COM, bitRate, timeout=0.1)
# # import serialstream
# signal.signal(signal.SIGINT, signal.SIG_DFL)



# # self.trigger = False
# run_comp_running = "Value = 0x02"
# run_comp_walking = "Value = 0x01"
# run_comp_stable = "Value = 0x00"

# DETECT_COUNT = 0
# Pre_RecieveData = run_comp_running # stable

# while True:

class Trigger():

    def run(self):
        # COM="COM17" # 完全BLE接続にする場合、トランシーバー用基板に送信
        COM="COM16" # Dongleとの通信

        bitRate=115200
        ser = serial.Serial(COM, bitRate, timeout=0.1)
        # import serialstream
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        # self.trigger = False
        run_comp_running = "Value = 0x02"
        run_comp_walking = "Value = 0x01"
        run_comp_stable = "Value = 0x00"

        # DETECT_COUNT = 0
        DETECT_COUNT_STABLE = 0
        DETECT_COUNT_MOVING = 0

        # L_RecieveData=ser.readline()
        # RecieveData = L_RecieveData.decode()
        # if (run_comp_running in RecieveData) or (run_comp_walking in RecieveData):
        #     Pre_RecieveData = run_comp_stable # 一番最初に受け取った行動以外を格納する
        # else:
        #     Pre_RecieveData = run_comp_running # 一番最初に受け取った行動以外を格納する
        Pre_RecieveData = None # run_comp_running # とりあえずRUNNING状態を初期状態に格納
        
        

        while True: # 今はずっとセンシングモードだが、30秒だけセンシングするとかにしてみる→その間に検出された行動を返す

            L_RecieveData=ser.readline()
            RecieveData = L_RecieveData.decode()
            print(RecieveData)

            action_result = None
            
            # if run_comp_running in RecieveData:
            if (run_comp_running in RecieveData) or (run_comp_walking in RecieveData) or (run_comp_stable in RecieveData): # RUNNING or WALKING で実行
                print("**********")
                print("Detect : ", RecieveData)
                print("Count(stable): ", DETECT_COUNT_STABLE)
                print("Count(moving): ", DETECT_COUNT_MOVING)
                
                # 今回は一回検出したらこの関数は終了するのでいらない
                # if not (Pre_RecieveData is None): # 追加
                #     if RecieveData in Pre_RecieveData: # 前回検出された状態と同じだった場合
                        
                #         for i in range(100): # 残っている検出結果をリリースする
                #             L_RecieveData=ser.readline()
                #             RecieveData = L_RecieveData.decode()
                #             print(RecieveData)
                #         # DETECT_COUNT = 0
                #         DETECT_COUNT_STABLE = 0
                #         DETECT_COUNT_MOVING = 0
                #     else:
                #         # DETECT_COUNT += 1
                #         DETECT_COUNT_STABLE += 1
                #         DETECT_COUNT_MOVING += 1
                # else: # 追加
                #         # DETECT_COUNT += 1
                #         DETECT_COUNT_STABLE += 1
                #         DETECT_COUNT_MOVING += 1
                if (run_comp_stable in RecieveData):
                    DETECT_COUNT_STABLE += 1
                if (run_comp_walking in RecieveData) or (run_comp_running in RecieveData):
                    DETECT_COUNT_MOVING += 1
                    

                # どのくらい検知したら結果を返すか
                # if DETECT_COUNT > 100: # 時間でもいいかも # STABLEなら100回でもいいかも
                """
                それぞれの行動にカウントを設ける
                STABLEは割と長時間(同じ行動検出しやすいから)
                WALKINGとRUNNNINGは短時間でいいかも
                """
                # if DETECT_COUNT > 10: # 時間でもいいかも

                #     # self.trigger = True

                #     # 同じ状態が継続しているか判別するために1つ前に検出された状態を格納する
                #     Pre_RecieveData = RecieveData
                    
                    
                #     if run_comp_stable in RecieveData:
                #         # subprocess.run(['C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/python.exe', 'C:/Users/0107409377/Desktop/code/AtCoder/src/DemoApp/DM4L.py', 'STABLE'])
                #         action_result = 'STABLE'
                #         return action_result
                    
                #     if run_comp_running in RecieveData:
                #         # subprocess.run(['C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/python.exe', 'C:/Users/0107409377/Desktop/code/AtCoder/src/DemoApp/DM4L.py', 'RUNNING'])
                #         action_result = 'WALKING'
                #         return action_result
                    
                #     if run_comp_walking in RecieveData:
                #         # subprocess.run(['C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/python.exe', 'C:/Users/0107409377/Desktop/code/AtCoder/src/DemoApp/DM4L.py', 'WALKING'])
                #         action_result = 'RUNNING'
                #         return action_result
                    
                #     for i in range(100): # 残っている検出結果をリリースする
                #         L_RecieveData=ser.readline()
                #         RecieveData = L_RecieveData.decode()
                #         print(RecieveData)
                    
                #     return action_result # 一応残しておく

                if DETECT_COUNT_STABLE > 30: #  # 50: # 時間でもいいかも

                    # 同じ状態が継続しているか判別するために1つ前に検出された状態を格納する
                    Pre_RecieveData = RecieveData
                    if run_comp_stable in RecieveData:
                        action_result = 'STABLE'
                        return action_result
                    

                    for i in range(100): # 残っている検出結果をリリースする
                        L_RecieveData=ser.readline()
                        RecieveData = L_RecieveData.decode()
                        print(RecieveData)
                    
                    # DETECT_COUNT = 0
                    DETECT_COUNT_STABLE = 0
                    
                if DETECT_COUNT_MOVING > 5: # 10: # 時間でもいいかも

                    # 同じ状態が継続しているか判別するために1つ前に検出された状態を格納する
                    Pre_RecieveData = RecieveData
                    
                    if run_comp_running in RecieveData:
                        action_result = 'WALKING'
                        return action_result
                    
                    if run_comp_walking in RecieveData:
                        action_result = 'RUNNING'
                        return action_result
                    
                    for i in range(100): # 残っている検出結果をリリースする
                        L_RecieveData=ser.readline()
                        RecieveData = L_RecieveData.decode()
                        print(RecieveData)
                        
                    # DETECT_COUNT = 0
                    DETECT_COUNT_MOVING = 0





if __name__ == "__main__":

    trigger = Trigger()
    print("***** センシング中 *****")
    args = trigger.run()
    print("DNN検出：", args)