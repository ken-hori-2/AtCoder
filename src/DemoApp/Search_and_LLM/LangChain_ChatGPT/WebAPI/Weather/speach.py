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
print(os.environ['OpenAI_API_KEY'])
key = os.environ['OpenAI_API_KEY']




##################
# Pyttsx3を初期化 #
##################
engine = pyttsx3.init()
# 読み上げの速度を設定する
rate = engine.getProperty('rate')
engine.setProperty('rate', rate)
# #rate デフォルト値は200
# rate = engine.getProperty('rate')
# engine.setProperty('rate',200)

#volume デフォルト値は1.0、設定は0.0~1.0
volume = engine.getProperty('volume')
engine.setProperty('volume',1.0)


# Kyokoさんに喋ってもらう(日本語)
engine.setProperty('voice', "com.apple.ttsbundle.Kyoko-premium")

def text_to_speach(response):
    # ストリーミングされたテキストを処理する
    fullResponse = ""
    RealTimeResponce = ""   
    # for chunk in response:
    #     text = chunk.choices[0].delta.content # chunk['choices'][0]['delta'].get('content')

    #     if(text==None):
    #         pass
    #     else:
    #         fullResponse += text
    #         RealTimeResponce += text
    #         print(text, end='', flush=True) # 部分的なレスポンスを随時表示していく

    #         target_char = ["。", "！", "？", "\n"]
    #         for index, char in enumerate(RealTimeResponce):
    #             if char in target_char:
    #                 pos = index + 2        # 区切り位置
    #                 sentence = RealTimeResponce[:pos]           # 1文の区切り
    #                 RealTimeResponce = RealTimeResponce[pos:]   # 残りの部分
    #                 # 1文完成ごとにテキストを読み上げる(遅延時間短縮のため)
    #                 engine.say(sentence)
    #                 engine.runAndWait()
    #                 break
    #             else:
    #                 pass

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

    # # # APIからの完全なレスポンスを返す
    # # return fullResponse
    # engine.say(response)
    # engine.runAndWait()


from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
# chain = ConversationChain(llm=ChatOpenAI(), memory=ConversationBufferMemory(return_messages=True))
from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)

# 人格を与えることで、そっけないレスポンスを軽減できる
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("あなたは優しくフレンドリーなAIです。「ねー」や「そだねー」のように、くだけた口調で話します。特にサッカーの話題は大好きなので、！を多用して話します。"),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])
chain = LLMChain(llm=ChatOpenAI(), prompt=prompt, memory=ConversationBufferMemory(return_messages=True))

response = chain.invoke("去年のワールドカップは本当に面白かったですね。")
# response = chain.run("去年のワールドカップは本当に面白かったですね。")
print(response['text'])
# response = response.split(",")
# conversationHistory = []
# conversationHistory.append(response)
print(type(response['text']))
# text_to_speach(conversationHistory)
text_to_speach(response['text'])