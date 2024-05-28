import os
from dotenv import load_dotenv
load_dotenv()

"""
2024/05/28
Scheduleバージョン

2024/05/24
Prediction_UserNeeds.py の改善版 (VectorStoreを追加して保管されているユーザーの履歴を参照し、ニーズを分析させる)

# ユーザーの傾向を分析、予測するLLM
# 分析結果から提案するLLM
"""

from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain_openai import ChatOpenAI
llm_4o=ChatOpenAI(
    model="gpt-4o",
    # model="gpt-3.5-turbo",
    temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
) # チャット特化型モデル
llm_3p5t=ChatOpenAI(
    # model="gpt-4o",
    model="gpt-3.5-turbo",
    temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
) # チャット特化型モデル

""" ################################################## GoogleColab\main_LangChain_Indexes_UserAction_類似度検索_UserNeeds.py """
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
# from langchain.document_loaders import TextLoader # txtはこっち
from langchain_community.document_loaders import TextLoader
# from langchain.document_loaders import CSVLoader # csvはこっち
from langchain_community.document_loaders import CSVLoader
# loader = TextLoader('./UserAction2_mini_loop_Schedule.txt', encoding='utf8')
loader = TextLoader('./userdata_schedule.txt', encoding='utf8')
# 100文字のチャンクで区切る
text_splitter = CharacterTextSplitter(        
    separator = "\n\n",
    chunk_size = 100,
    chunk_overlap = 0,
    length_function = len,
)
index = VectorstoreIndexCreator(
    vectorstore_cls=Chroma, # Default
    embedding=OpenAIEmbeddings(), # Default
    text_splitter=text_splitter, # text_splitterのインスタンスを使っている
).from_loaders([loader])
query = """
        あなたはニーズを予測する専門家です。以下に答えて。
        txtファイルの文書はユーザーの「どんな予定の時に、どの行動状態で、どの機能を使用したか」の履歴です。
        このユーザーの傾向を分析・予測し箇条書きでまとめて。
        """
        # (例：予定ごとの使う機能の傾向)
        # """
        # (例1：朝の時間帯(xx:xx - xx:xx): ,午前中(xx:xx - xx:xx), 昼の時間帯(xx:xx - xx:xx), 午後(xx:xx - xx:xx), 夕方(xx:xx - xx:xx), 夜の時間帯(xx:xx - xx:xx))
        # (例2：行動状態ごとの使う機能の傾向)
        # """
answer = index.query(query, llm=llm_4o)



prompt_2 = PromptTemplate(
    input_variables=["UserNeeds", "schedule", "UserAction"],
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
    # 行動状態無しバージョン、傾向無しバージョン
    # template = """
    #            あなたはユーザーに合う機能を提案する専門家です。

    #            現在の予定が{schedule}の場合、どの機能を提案しますか？
    #            その際、各機能の提案する確率と最終的な提案(Final Answer:)、その理由も教えて。
    #            あなたが提案できる機能は、
    #            "会議情報", "楽曲再生", "経路検索", "リアルタイム情報検索", "レストラン検索", "ニュース情報", "天気情報"
    #            です。
    #            会議時にSTABLEの場合は会議が長引いていることが多いです。
    #            """

    """
    # どちらともユーザーの傾向がないとあまり精度良くない
    """


)
chain_2 = LLMChain(llm=llm_4o, prompt=prompt_2, output_key="output")
# chain_2 = prompt_2 | llm_4o # 新しいやり方



from langchain.chains import SequentialChain
overall_chain = SequentialChain(
    chains=[chain_2],
    input_variables=["UserNeeds", "schedule", "UserAction"],
    # output_variables=["response"], # あくまで辞書型のなんていう要素に出力が格納されるかの変数
    verbose=True,
)
response = overall_chain({
    "UserNeeds" : answer,
    "schedule" : "通勤",
    # "schedule" : "昼食",
    # "schedule" : "定例",
    # "schedule" : "進捗確認",
    # "schedule" : "面談",
    # "UserAction" : "STABLE",
    "UserAction" : "WALKING",
})
print("\n--------------------------------------------------")
print("User Needs: \n", response['UserNeeds'])
print("\n--------------------------------------------------")
print("schedule: ", response['schedule'])
print("User Action: ", response['UserAction'])
print("\n--------------------------------------------------")
print("Output: ", response['output'])
print("\n--------------------------------------------------")


""" ################################################## GoogleColab\main_LangChain_Indexes_UserAction_類似度検索_UserNeeds.py """

# from typing import Dict, List

# class ConcatenateChain(Chain):
#     chain_1: LLMChain
#     chain_2: LLMChain

#     @property
#     def input_keys(self) -> List[str]:
#         all_input_vars = set(self.chain_1.input_keys).union(set(self.chain_2.input_keys))
#         return list(all_input_vars)

#     @property
#     def output_keys(self) -> List[str]:
#         return ['concat_output']

#     def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
#         output_1 = self.chain_1.run(inputs)
#         output_2 = self.chain_2.run(inputs)
#         return {'concat_output': output_1 + "\n" + output_2}

# prompt_1 = PromptTemplate(
#     # input_variables=["job"],
#     # template="{job}に一番オススメのプログラミング言語は?\nプログラミング言語：",
#     input_variables=["schedule", "UserAction"],
#     template =  """
#                 あなたはニーズを予測する専門家です。以下に答えて。
#                 txtファイルの文書はユーザーの「どの時刻に、どの行動状態で、どの機能を使用したか」の履歴です。
#                 このユーザーの傾向を予測し箇条書きでまとめて。
#                 (例：朝の時間帯(xx:xx - xx:xx): ,午前中(xx:xx - xx:xx), 昼の時間帯(xx:xx - xx:xx), 午後(xx:xx - xx:xx), 夕方(xx:xx - xx:xx), 夜の時間帯(xx:xx - xx:xx))
#                 """
# )
# chain_1 = LLMChain(llm=llm_4o, prompt=prompt_1)

# prompt_2 = PromptTemplate(
#     # input_variables=["job"],
#     # template="{job}の平均年収は？\n平均年収：",
#     input_variables=["schedule", "UserAction"],
#     template =  """
#                 現在が{schedule}、ユーザーの行動状態が{UserAction}の場合、どの機能を提案するかこのユーザーの傾向を加味して予測して。
#                 その際、各機能の提案する確率と最終的な提案(Final Answer:)、その理由も教えて。
#                 あなたが提案できる機能は、
#                 "会議情報", "楽曲再生", "経路検索", "リアルタイム情報検索", "レストラン検索", "ニュース情報", "天気情報"
#                 です。
#                 """
# )
# # chain_2 = LLMChain(llm=llm_3p5t, prompt=prompt_2)
# chain_2 = LLMChain(llm=llm_4o, prompt=prompt_2)



# concat_chain = ConcatenateChain(chain_1=chain_1, chain_2=chain_2, verbose=True)
# # print(concat_chain.run("データサイエンティスト"))

# from langchain.chains import SequentialChain
# overall_chain = SequentialChain(
#     chains=[concat_chain],
#     input_variables=["schedule", "UserAction"],
#     # output_variables=["response"], # あくまで辞書型のなんていう要素に出力が格納されるかの変数
#     verbose=True,
# )

# response = overall_chain({
#     "schedule" : "11時41分",
#     "UserAction" : "WALKING",
# })

# print(response) # ['response'])
# print("----------")
# print(response['concat_output'])