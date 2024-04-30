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
# from langchain.llms import OpenAI
from langchain_community.llms import OpenAI
# from langchain.chat_models import ChatOpenAI # 少し古いかも
from langchain_openai import ChatOpenAI # 新しいやり方
import os
import pandas as pd
import json
import requests

from dotenv import load_dotenv
# .envファイルの内容を読み込見込む
load_dotenv()
# os.environを用いて環境変数を表示させます
print(os.environ['OpenAI_API_KEY'])
# key = 
os.environ['OpenAI_API_KEY']

# # ⭐️ここにOpenAIから取得したキーを設定します。⭐️
# os.environ["OPENAI_API_KEY"] = key

# # インターネット上からファイルをダウンロード（BuffaloTerastation)
# import gdown
# # 産業一覧マスタ
# document = gdown.download('https://drive.google.com/uc?id=1hZmAK3G-PaYj1Y5CMGJwB6hSubMA-9gg', 'industries.csv', quiet=False)
# # 置換する元データ
# document = gdown.download('https://drive.google.com/uc?id=1cACvdILd-4RxdhmRcURP-dpSdNwZcWYP', 'input-data.csv', quiet=False)

# response schema 
# nameとその説明を加えることで、各カラムの中身を指定することができる
response_schema = [
    ResponseSchema(name="input_application", description="これはユーザから入力してもらった要求の情報です"),
    ResponseSchema(name="[1st]standarized_application", description="これはあなたがユーザから入力してもらった情報が最も近いと感じたapplicationの情報です。 "),
    ResponseSchema(name=" >[1st]match_score",  description="インプットされた情報がどのくらいapplicationに近いかどうかを示すスコアを1-100で表現する"),
    
    ResponseSchema(name="[2nd]standarized_application_2nd", description="これはあなたがユーザから入力してもらった情報が二番目に近いと感じたapplicationの情報です。 "),
    ResponseSchema(name=" >[2nd]match_score_2nd",  description="インプットされた情報がどのくらいapplicationに近いかどうかを示すスコアの二番目を1-100で表現する"),
    ResponseSchema(name="[3nd]standarized_application_3nd", description="これはあなたがユーザから入力してもらった情報が三番目に近いと感じたapplicationの情報です。 "),
    ResponseSchema(name=" >[3nd]match_score_3nd",  description="インプットされた情報がどのくらいapplicationに近いかどうかを示すスコアの三番目を1-100で表現する"),
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

input_applications INPUT:
{user_applications}

STANDARDIZED APPLICATIONS:
{standardized_applications}

YOUR RESPONSE:
"""
# ここでプロンプトを作成する
prompt = ChatPromptTemplate(
    messages=[HumanMessagePromptTemplate.from_template(template)],
    input_variables=["user_applications", "standardized_applications"],
    # outputのフォーマットを指定する
    partial_variables={"format_instructions": format_instructions}
)






# 統一したい産業名のリストを読み込む
# df = pd.read_csv('industries.csv')
df = pd.read_csv('Application.csv') # 2024/04/29

# 日本語に統一したい場合はApplicationJp, 英語にしたい場合はApplicationEngを使う
standardized_applications = ", ".join(df['ApplicationJP'].values)
# print(standardized_applications)

# インプットデータを読み込み
# input_df = pd.read_csv('./input-data.csv')
input_df = pd.read_csv('./Application_input-data.csv') # 2024/04/29
# priny(input_df.head(5))

# 読み込んだデータの中から職業列を抜き出し、リストに変換
occu_list = input_df['要求'].to_list()
# print(occu_list[:5)]

# テンプレートに値を設定
_input = prompt.format_prompt(user_applications=occu_list, standardized_applications=standardized_applications)
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

print (output.content[:200])








# AIからの返答に```が含まれている場合は、それを削除する
if "```json" in output.content:
    json_string = output.content.split("```json")[1].strip()
else:
    json_string = output.content
# 後ろの```を削除
json_string=json_string.replace('```', '')

# jsonをパースして、pythonのDataflame型に変換します
ai_answer = json.loads(json_string)
ai_answer_df = pd.DataFrame(ai_answer)

# 最後に、もとのデータに得られた分類結果を追加します。
final_df = input_df.copy()
# # final_df.insert(3,"産業（振り分け結果）", ai_answer_df["standarized_application"].to_list(),allow_duplicates=True)
# # final_df.insert(4,"match_score", ai_answer_df["match_score"].to_list(),allow_duplicates=True)
# final_df.insert(ai_answer_df["[1st]standarized_application"].to_list(),allow_duplicates=True)
# final_df.insert(ai_answer_df[" >[1st]match_score"].to_list(),allow_duplicates=True)


# 基準のDataFrameを作成
print("*****\n要求に対して一番近いApplicationを出力\n*****")
data1 = ai_answer_df["[1st]standarized_application"].to_list()
data2 = ai_answer_df[" >[1st]match_score"].to_list()
# data3 = [8, 9, 10, 11]
final_df = pd.DataFrame(
    {"[1st]Application": data1, "[1st]Score": data2},
    # index=['要求1', '要求2', '要求3', '要求4', '要求5', '要求6', '要求7']
    index=[input_df['要求'].to_list()]
)

# 結果を確認
# print(pd.DataFrame(final_df))
print(final_df)
final_df.to_csv("output.csv")


print("**********\n**********")
# df = pd.DataFrame([json_string])
# print(df)

# print(json_string) # .to_csv("test.csv") # 2024/04/29

# json_string.to_json("test.json")



# for i in (json_string):
#     print(i)
# print(json_string[0])