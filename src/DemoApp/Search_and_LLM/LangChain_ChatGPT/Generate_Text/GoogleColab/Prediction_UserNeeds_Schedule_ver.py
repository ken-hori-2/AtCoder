import os
from dotenv import load_dotenv
load_dotenv()

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import SimpleMemory
from langchain.memory import ConversationBufferMemory
"""
WARNING
"""
# LangChainDeprecationWarning: The class `LLMChain` was deprecated in LangChain 0.1.17 and will be removed in 0.3.0. Use RunnableSequence, e.g., `prompt | llm` instead.

from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
llm=ChatOpenAI(
    model="gpt-4o",
    # model="gpt-3.5-turbo",
    temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
) # チャット特化型モデル
# memory_key = "chat_history"
memory = ConversationBufferMemory(memory_key="chat_history", ai_prefix="") # , return_messages=True)

"""
Input Textに全部含める場合
"""
"""
テンプレート化する場合
"""
# テンプレート化
prompt_1 = PromptTemplate(
    # input_variables=["input", "time", "UserAction"], # , "chat_history"],
    input_variables=["schedule", "UserAction"],
    # template="{adjective}{job}に一番オススメのプログラミング言語は?\nプログラミング言語：",
    # template = """
    #            あなたは人間と話すチャットボットです。ユーザーの要求に答えてください。

    #            以下はユーザーの1日の行動と使った機能です。
    #            このユーザーの傾向を述べてください。(例: 1. 予定ごとの行動パターン: , 2. 行動状態: )
    #            その後、現在が{schedule}、ユーザーの行動状態が{UserAction}の場合どの機能を提案するか教えてください。
    #            その際、各機能の提案する確率と最終的な提案(Final Answer:)も教えてください。

    #            None, STABLE, 天気情報
    #            出勤, WALKING, 経路検索
    #            出社, STABLE, 楽曲再生
    #            出社, STABLE, 会議情報
    #            出社, WALKING, 会議情報
    #            昼食, STABLE, レストラン検索
    #            昼食, STABLE, 楽曲再生
    #            出社, WALKING, 会議情報
    #            帰宅, STABLE, 経路検索
    #            帰宅, STABLE, 楽曲再生
    #            ジム, RUNNING, 楽曲再生
    #            ジム, RUNNING, 楽曲再生



    #            あなたが提案できる機能は
    #            "会議情報", "楽曲再生", "経路検索", "リアルタイム情報検索", "レストラン検索", "ニュース情報", "天気情報"
    #            です。
    #            """
    # 行動状態ありバージョン
    template = """
               あなたはユーザーに合う機能を提案する専門家です。
               ユーザーの傾向は「{UserNeeds}」です。
               
               現在の予定が{schedule}、ユーザーの行動状態が{UserAction}の場合、どの機能を提案するかこのユーザーの傾向を加味して予測して。
               その際、各機能の提案する確率と最終的な提案(Final Answer:)、その理由も教えて。
               あなたが提案できる機能は、
               "会議情報", "楽曲再生", "経路検索", "リアルタイム情報検索", "レストラン検索", "ニュース情報", "天気情報"
               です。
               """
    # 行動状態無しバージョン
    # template = """
    #            あなたはユーザーに合う機能を提案する専門家です。
    #            現在の予定が{schedule}の場合、どの機能を提案しますか？
    #            その際、各機能の提案する確率と最終的な提案(Final Answer:)、その理由も教えて。
    #            あなたが提案できる機能は、
    #            "会議情報", "楽曲再生", "経路検索", "リアルタイム情報検索", "レストラン検索", "ニュース情報", "天気情報"
    #            です。
    #            """
)

# chain_1 = LLMChain(llm=llm, prompt=prompt_1)
chain_1 = LLMChain(llm=llm, prompt=prompt_1, output_key="response") # あくまで辞書型のなんていう要素に出力が格納されるかの変数


# ユーザーの過去の行動を入力せずにやってみる(2024/05/23)
overall_chain = SequentialChain(
    chains=[chain_1], # , chain_2], 
    input_variables=["schedule", "UserAction"],
    output_variables=["response"], # あくまで辞書型のなんていう要素に出力が格納されるかの変数
    verbose=True,
)
output = overall_chain({
    # "schedule" : "出勤",
    # "schedule" : "昼食",
    # "schedule" : "会議",
    "schedule" : "ジム",
    # "UserAction" : "STABLE",
    # "UserAction" : "WALKING",
    "UserAction" : "RUNNING",
})
# print(output['text'])
# print(output)
print(output['response'])
# """