import speech_recognition as sr
import os
import openai
import pyttsx3
import re

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

from dotenv import load_dotenv
# .envファイルの内容を読み込見込む
load_dotenv()
# os.environを用いて環境変数を表示させます
print(os.environ['API_KEY'])
key = os.environ['API_KEY']

##############
# 音声認識関数 #
##############
def recognize_speech():

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


#################################
# Pyttsx3でレスポンス内容を読み上げ #
#################################
def text_to_speech(text):
    # テキストを読み上げる
    engine.say(text)
    engine.runAndWait()



def chat(conversationHistory):
    # APIリクエストを作成する
    response = openai.chat.completions.create(
        messages=conversationHistory,
        max_tokens=1024,
        n=1,
        stream=True,
        temperature=0.5,
        stop=None,
        presence_penalty=0.5,
        frequency_penalty=0.5,
        model="gpt-3.5-turbo"
    )

    # ストリーミングされたテキストを処理する
    fullResponse = ""
    RealTimeResponce = ""   
    for chunk in response:
        text = chunk.choices[0].delta.content # chunk['choices'][0]['delta'].get('content')

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

    # APIからの完全なレスポンスを返す
    return fullResponse


##############
# メインの関数 #
##############
if __name__ == '__main__':

    ##################
    # ChatGPTの初期化 #
    ##################
    openai.api_key=key # 自身のAPIキーを指定"
    # UserとChatGPTとの会話履歴を格納するリスト
    conversationHistory = []
    # setting = {"role": "system", "content": "句読点と読点を多く含めて応答するようにして下さい。また、1文あたりが長くならないようにして下さい。"}
    setting = {"role": "system", "content": "句読点と読点を多く含めて応答するようにして下さい。また、1文あたりが長くならないようにして下さい。100文字以内でお願いします。"}
    
    ##################
    # Pyttsx3を初期化 #
    ##################
    engine = pyttsx3.init()
    # 読み上げの速度を設定する
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate)
    # Kyokoさんに喋ってもらう(日本語)
    engine.setProperty('voice', "com.apple.ttsbundle.Kyoko-premium")

    # Ctrl-Cで中断されるまでChatGPT音声アシスタントを起動
    while True:
        
        engine.say("Detect Combo !!")
        engine.say("Please speak now")
        engine.runAndWait()
        # 音声認識関数の呼び出し
        text = recognize_speech()
        # text = "以下の条件の下でおいしい食べ物を教えてください。\
        #     \n条件1:和食 \
        #     \n条件2:甘い"
        
        # text = "本厚木駅周辺のお店 おいしいところ探して"

        if text:
            print(" >> Waiting for response from ChatGPT...")
            # ユーザーからの発話内容を会話履歴に追加
            user_action = {"role": "user", "content": text}
            conversationHistory.append(user_action)
            
            print("[ChatGPT]") #応答内容をコンソール出力
            res = chat(conversationHistory)
            
            # ChatGPTからの応答内容を会話履歴に追加
            chatGPT_responce = {"role": "assistant", "content": res}
            conversationHistory.append(chatGPT_responce) 
            print(conversationHistory)

            # 1回で終了する場合
            break
    
    engine.say("Guidance End")
    engine.runAndWait()

            

