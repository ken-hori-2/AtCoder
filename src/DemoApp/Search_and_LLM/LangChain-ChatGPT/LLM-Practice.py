# ライブラリのインストール
# pip install langchain
# pip install unstructured
# pip install pandas
# pip install chromadb
# pip install tiktoken
# pip install openai
# pip install requests

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import os
import pandas as pd
import json
import requests

from dotenv import load_dotenv
# .envファイルの内容を読み込見込む
load_dotenv()
# os.environを用いて環境変数を表示させます
print(os.environ['OpenAI_API_KEY'])
key = os.environ['OpenAI_API_KEY']

# ⭐️ここにOpenAIから取得したキーを設定します。⭐️
os.environ["OPENAI_API_KEY"] = key

# # インターネット上からファイルをダウンロード（BuffaloTerastation)
# import gdown
# # 産業一覧マスタ
# document = gdown.download('https://drive.google.com/uc?id=1hZmAK3G-PaYj1Y5CMGJwB6hSubMA-9gg', 'industries.csv', quiet=False)
# # 置換する元データ
# document = gdown.download('https://drive.google.com/uc?id=1cACvdILd-4RxdhmRcURP-dpSdNwZcWYP', 'input-data.csv', quiet=False)

# response schema 
# nameとその説明を加えることで、各カラムの中身を指定することができる
response_schema = [
    ResponseSchema(name="input_industory", description="これはユーザから入力してもらった産業の情報です"),
    ResponseSchema(name="standarized_industry", description="これはあなたユーザから入力してもらった情報が最も近いと感じたindustoryの情報です。 "),
    ResponseSchema(name="match_score",  description="インプットされた情報がどのくらいindustryに近いかどうかを示す上スコアを1-100で表現する"),
    
    # 入力情報が英語ならTrue,そうでないならFalseを出力する列を出力結果に追加する
    ResponseSchema(name="isEnglish",  description="to be true if input is written in English")
    # 他にもdescriptionに条件を定義すれば、いろんな結果を得られるかも
]
# どのようにレスポンスをパースするかを指定する  
output_parser = StructuredOutputParser.from_response_schemas(response_schema)

# アウトプットの指示をするプロンプトを作成する
format_instructions = output_parser.get_format_instructions()
# 生成されるプロンプトの中身でここで確認
print("***** format_instructions = output_parser *****")
print(output_parser.get_format_instructions())

template = """
You will be given a series of industry names from a user.
Find the best corresponding match on the list of standardized names.
The closest match will be the one with the closest semantic meaning. Not just string similarity.

{format_instructions}

Wrap your final output with closed and open brackets (a list of json objects)

input_industry INPUT:
{user_industries}

STANDARDIZED INDUSTRIES:
{standardized_industries}

YOUR RESPONSE:
"""
# ここでプロンプトを作成する
prompt = ChatPromptTemplate(
    messages=[HumanMessagePromptTemplate.from_template(template)],
    input_variables=["user_industries", "standardized_industries"],
    # outputのフォーマットを指定する
    partial_variables={"format_instructions": format_instructions}
)






# 統一したい産業名のリストを読み込む
df = pd.read_csv('industries.csv')
# df = pd.read_csv('Application.csv') # 2024/04/29

# 日本語に統一したい場合はIndustryJp, 英語にしたい場合はIndustryEngを使う
standardized_industries = ", ".join(df['IndustryJP'].values)
# print(standardized_industries)

# インプットデータを読み込み
input_df = pd.read_csv('./input-data.csv')
# input_df = pd.read_csv('./Application_input-data.csv') # 2024/04/29
# priny(input_df.head(5))

# 読み込んだデータの中から職業列を抜き出し、リストに変換
occu_list = input_df['職業'].to_list()
# print(occu_list[:5)]

# テンプレートに値を設定
_input = prompt.format_prompt(user_industries=occu_list, standardized_industries=standardized_industries)
# ここでChatGPTに渡すメッセージを確認

# print (f"There are {len(_input.messages)} message(s)")
# print (f"Type: {type(_input.messages[0])}")
# print ("---------------------------")
# print (_input.messages[0].content)

# temperatureには0〜2.0の値を指定するランダム性を高めたい場合は大きな値を指定する
# max_tokensには生成する最大トークン数を指定する(gpt3.5では4000が最高)
# chat_model = ChatOpenAI(temperature=0, max_tokens=3200)
chat_model = ChatOpenAI(temperature=0, max_tokens=1024)

# ここでChatGPTにリクエストを投げる
output = chat_model(_input.to_messages())

# print (output.content[:200])








# AIからの返答に```が含まれている場合は、それを削除する
if "```json" in output.content:
    json_string = output.content.split("```json")[1].strip()
else:
    json_string = output.content
# 後ろの```を削除
json_string=json_string.replace('```', '')

# # jsonをパースして、pythonのDataflame型に変換します
# ai_answer = json.loads(json_string)
# ai_answer_df = pd.DataFrame(ai_answer)

# # 最後に、もとのデータに得られた分類結果を追加します。
# final_df = input_df.copy()
# final_df.insert(3,"産業（振り分け結果）", ai_answer_df["standarized_industry"].to_list(),allow_duplicates=True)
# final_df.insert(4,"match_score", ai_answer_df["match_score"].to_list(),allow_duplicates=True)

# # 結果を確認
# pd.DataFrame(final_df)
print(json_string) # .to_csv("test.csv")
# json_string.to_json("test.json")