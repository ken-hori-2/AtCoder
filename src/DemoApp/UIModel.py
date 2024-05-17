import speech_recognition as sr
import pyttsx3

##################
# Pyttsx3を初期化 #
##################
import pyttsx3
engine = pyttsx3.init()
# 読み上げの速度を設定する
rate = engine.getProperty('rate')
engine.setProperty('rate', rate)
#volume デフォルト値は1.0、設定は0.0~1.0
volume = engine.getProperty('volume')
engine.setProperty('volume',1.0)
# Kyokoさんに喋ってもらう(日本語)
engine.setProperty('voice', "com.apple.ttsbundle.Kyoko-premium")


class UserInterfaceModel(): # ユーザーとのやり取りをするモデル
    ################
    # 音声認識関数 #
    ################
    def recognize_speech(self):

        recognizer = sr.Recognizer()    
        # Set timeout settings.
        recognizer.dynamic_energy_threshold = False

        
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
        
            while(True):
                print(">> Please speak now...")

                # engine.say("Please speak now")


                audio = recognizer.listen(source, timeout=1000.0)

                try:
                    # Google Web Speech API を使って音声をテキストに変換
                    text = recognizer.recognize_google(audio, language="ja-JP")
                    print("[You]")
                    print(text)
                    return text
                except sr.UnknownValueError:
                    print("Sorry, I could not understand what you said. Please speak again.")
                    #return ""
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
                    #return ""
    

    #######################
    # テキストから音声変換 #
    #######################
    def text_to_speach(self, response):
        # ストリーミングされたテキストを処理する
        fullResponse = ""
        RealTimeResponce = ""

        # 随時レスポンスを音声ガイダンス
        for chunk in response:
            text = chunk

            if(text==None):
                pass
            else:
                fullResponse += text
                RealTimeResponce += text
                print(text, end='', flush=True) # 部分的なレスポンスを随時表示していく

                target_char = ["。", "！", "？", "\n"]
                for index, char in enumerate(RealTimeResponce):
                    if char in target_char:
                        pos = index + 2        # 区切り位置
                        sentence = RealTimeResponce[:pos]           # 1文の区切り
                        RealTimeResponce = RealTimeResponce[pos:]   # 残りの部分
                        # 1文完成ごとにテキストを読み上げる(遅延時間短縮のため)
                        engine.say(sentence)
                        engine.runAndWait()
                        break
                    else:
                        pass