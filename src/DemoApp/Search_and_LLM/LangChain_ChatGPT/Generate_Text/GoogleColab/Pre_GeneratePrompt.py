import os
from dotenv import load_dotenv
load_dotenv()


# 2024/05/28
"""
トークン削減のためテスト用
"""
import speech_recognition as sr
import pyttsx3
import pyttsx3
engine = pyttsx3.init()
rate = engine.getProperty('rate')
engine.setProperty('rate', rate)
volume = engine.getProperty('volume')
engine.setProperty('volume',1.0)
engine.setProperty('voice', "com.apple.ttsbundle.Kyoko-premium")
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



from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain_openai import ChatOpenAI
from langchain.chains import SequentialChain
# from DemoApp.Search_and_LLM.LangChain_ChatGPT.UIModel_copy import UserInterfaceModel # ユーザーとのやり取りをするモデル
llm_4o=ChatOpenAI(
    model="gpt-4o",
    # model="gpt-3.5-turbo",
    temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
)
llm_3p5t=ChatOpenAI(
    # model="gpt-4o",
    model="gpt-3.5-turbo",
    temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
)

# 提案ツールからプロンプト生成

prompt_3 = PromptTemplate(
    input_variables=["SuggestedTool"],
    template = """
               LLMに入力するための短いプロンプトを生成して。
               具体的には「{SuggestedTool}」をもとに、ユーザーのニーズを満たせるようなプロンプトを簡潔に生成して。
               ###
               Prompt: (例：~を教えて。)
               """
)
# chain_3 = LLMChain(llm=llm_4o, prompt=prompt_3, output_key="output")
chain_3 = LLMChain(llm=llm_3p5t, prompt=prompt_3, output_key="output")
overall_chain_3 = SequentialChain(
    chains=[chain_3],
    # input_variables=["UserNeeds", "SuggestedTool", "UserAction"],
    input_variables=["SuggestedTool"],
    # output_variables=["response"], # あくまで辞書型のなんていう要素に出力が格納されるかの変数
    verbose=True,
)



Suggested_Tool = "経路検索"


# ここのタイミングでユーザーに確認を求める
# 例：「{Suggested_Tool}を提案します。実行しますか？」
print(f"「{Suggested_Tool}を提案します。実行しますか？」")
# UserInputExec = userinterface.recognize_speech() # 音声認識をする場合
UserInputExec = recognize_speech() # 音声認識をする場合
if ("はい" in UserInputExec) or ("YES" in UserInputExec) or ("イエス" in UserInputExec) or ("実行" in UserInputExec) or ("お願い" in UserInputExec):
    print("実行します。")
    Generated_Prompt = overall_chain_3({
    "SuggestedTool" : Suggested_Tool,
    })
    print("\n--------------------------------------------------")
    print(Generated_Prompt['output'])
    print("--------------------------------------------------")
"""
LLMで入力の揺らぎに対処する
"""