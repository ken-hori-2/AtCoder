import openai
import os
from dotenv import load_dotenv
# .envファイルの内容を読み込見込む
load_dotenv()
# os.environを用いて環境変数を表示させます
print(os.environ['API_KEY'])
key = os.environ['API_KEY']

# APIキーの設定
openai.api_key = key # "APIキー" 

# GPTによる応答生成
prompt = "以下の条件の下でおいしい食べ物を教えてください。\
            \n条件1:和食 \
            \n条件2:甘い"

# response = openai.ChatCompletion.create(
#                     model = "gpt-3.5-turbo-16k-0613",
#                     messages = [
#                         {"role": "system", "content": "You are a helpful assistant."},
#                         {"role": "user", "content": prompt}
#                     ],
#                     temperature=0
#                 )
response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages= [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0
        )

# 応答の表示
# text = response['choices'][0]['message']['content']
text = response.choices[0].message.content
print(text)
print(type(text))

import pyttsx3

engine = pyttsx3.init()
# engine.say("こんにちは。こんばんは。")
voices = engine.getProperty('voices')
engine.setProperty("voice", voices[1].id)
rate = engine.getProperty('rate')
engine.setProperty('rate', rate-50)
# engine.setProperty('volume', 音量) # 0.0 ~ 1.0
engine.setProperty('volume', 1.0) # 0.0 ~ 1.0

engine.say(text) # 実行中にアナウンス
# engine.runAndWait()