
from langchain.llms import OpenAI
# from langchain.agents import create_pandas_dataframe_agent
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

# from langchain_openai import ChatOpenAI
# llm=ChatOpenAI(
#     # model="gpt-4o",
#     model="gpt-3.5-turbo",
#     temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
# ) # チャット特化型モデル

# # df = pd.read_csv('titanic.csv')
# # agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
# # # agent = create_pandas_dataframe_agent(llm, df, verbose=True)
# # agent.run("whats the square root of the average age?")


import pandas as pd

# df = pd.DataFrame({
#     "name": ["佐藤", "鈴木", "吉田"],
#     "age": ["20", "30", "40"],
# })

# # llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
# # agent = create_pandas_dataframe_agent(llm, df, verbose=True)
# agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

# agent.run("佐藤さんの年齢は？")
# agent.run("ユーザーの平均年齢は？")

# agent.run("さとうさんの年齢は？") # ひらがなでも対応してくれる

# これはcsvから取得することも可能
df = pd.DataFrame({
    "time":   ["8:00",     "8:30",    "9:00",     "9:30",     "12:00",         "15:00",    "17:30",   "19:00"],
    "action": ["STABLE",   "WALKING", "STABLE",   "STABLE",   "WALKING",       "STABLE",   "WALKING", "RUNNING"],
    "tool":   ["天気情報", "経路検索", "楽曲再生", "会議情報", "レストラン検索", "会議情報", "経路検索", "楽曲再生"],
})
agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
query = """
        あなたは人間と話すチャットボットです。ユーザーの要求に答えてください。
        dfはユーザーの1日の行動と使った機能です。
        このユーザーの傾向を述べてください。(例: 1. 時間帯ごとの行動パターン: , 2. 行動状態: )
        その後、現在が11時30分、ユーザーの行動状態がSTABLEの場合、どの機能を提案するか教えてください。
        その際、各機能の提案する確率と最終的な提案(Final Answer:)も教えてください。
        ユーザーの傾向を推測するモデルを作成し、推論してください。また、モデルのコードも出力してください。
        """

agent.run(query)
# response = agent.run(query)
# print(response)