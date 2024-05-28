import os
from dotenv import load_dotenv
load_dotenv()


# 2024/05/28
# GoogleColab\main_PredUserNeeds_by_VectorStore_to_Chain_Schedule_ver.py のコピー



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
    # # 行動状態ありバージョン
    # template = """
    #            あなたはユーザーに合う機能を提案する専門家です。
    #            ユーザーの傾向は「{UserNeeds}」です。

    #            現在の予定が{schedule}、ユーザーの行動状態が{UserAction}の場合、どの機能を提案するかこのユーザーの傾向を加味して予測して。
    #            その際、各機能の提案する確率と最終的な提案(Final Answer:)、その理由も教えて。
    #            あなたが提案できる機能は、
    #            "会議情報", "楽曲再生", "経路検索", "リアルタイム情報検索", "レストラン検索", "ニュース情報", "天気情報"
    #            です。
    #            ### 回答形式
    #            Final Answer:
    # 理由なし
    template = """
               あなたはユーザーに合う機能を提案する専門家です。
               ユーザーの傾向は「{UserNeeds}」です。

               現在の予定が{schedule}、ユーザーの行動状態が{UserAction}の場合、どの機能を提案するかこのユーザーの傾向を加味して最終的な提案(Final Answer:)のみを教えて。
               あなたが提案できる機能は、
               "会議情報", "楽曲再生", "経路検索", "リアルタイム情報検索", "レストラン検索", "ニュース情報", "天気情報"
               です。
               ###
               Final Answer:
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

UserActionState = "WALKING"

response = overall_chain({
    "UserNeeds" : answer,
    "schedule" : "通勤",
    # "schedule" : "昼食",
    # "schedule" : "定例",
    # "schedule" : "進捗確認",
    # "schedule" : "面談",
    # "UserAction" : "STABLE",
    "UserAction" : UserActionState,
})
# print("\n--------------------------------------------------")
# print("User Needs: \n", response['UserNeeds'])
# print("\n--------------------------------------------------")
# print("schedule: ", response['schedule'])
# print("User Action: ", response['UserAction'])
# print("\n--------------------------------------------------")
# print("Output: ", response['output'])
print("\n--------------------------------------------------")
Suggested_Tool = response['output']
print(Suggested_Tool)
print("--------------------------------------------------")


""" ################################################## GoogleColab\main_LangChain_Indexes_UserAction_類似度検索_UserNeeds.py """


"""
トークン削減のためテスト用で別ファイル(GoogleColab\GeneratePrompt.py)に記述
"""
# # 提案ツールからプロンプト生成

# prompt_3 = PromptTemplate(
#     input_variables=["UserNeeds", "SuggestedTool", "UserAction"],
#     # template = """
#     #            あなたはユーザーの傾向からLLMに入力するためのプロンプトを生成する専門家です。
#     #            ユーザーの傾向は「{UserNeeds}」、提案機能は「{SuggestedTool}」です。
#     #            これらを踏まえ、ユーザーのニーズを満たせるようなLLM用のプロンプトを簡潔に生成して。
#     #            ###
#     #            Prompt:
#     #            """
#     template = """
#                あなたはLLMに入力するためのプロンプトを生成する専門家です。
#                提案機能は「{SuggestedTool}」をもとに、ユーザーのニーズを満たせるようなLLM用のプロンプトを簡潔に生成して。
#                ###
#                Prompt:
#                """
# )
# chain_3 = LLMChain(llm=llm_4o, prompt=prompt_3, output_key="output")
# overall_chain_3 = SequentialChain(
#     chains=[chain_3],
#     input_variables=["UserNeeds", "SuggestedTool", "UserAction"],
#     # output_variables=["response"], # あくまで辞書型のなんていう要素に出力が格納されるかの変数
#     verbose=True,
# )
# Generated_Prompt = overall_chain_3({
#     "UserNeeds" : answer,
#     "SuggestedTool" : Suggested_Tool,
#     "UserAction" : UserActionState,
# })
# print("\n--------------------------------------------------")
# print(Generated_Prompt['output'])
# print("--------------------------------------------------")

"""
トークン削減のためテスト用で別ファイル(GoogleColab\Within5min.py)に記述
"""
# # 5分以内に次の予定が始まる場合（トリガー）
# # self.isMtgStart = self.schedule.isMtgStartWithin5min(margin, dt_now)
# # print("MtgStart:", self.isMtgStart)
# import win32com
# import datetime
# Outlook = win32com.client.Dispatch("Outlook.Application").GetNameSpace("MAPI")
# items = Outlook.GetDefaultFolder(9).Items
# # 定期的な予定の二番目以降の予定を検索に含める
# items.IncludeRecurrences = True
# # 開始時間でソート
# items.Sort("[Start]")
# # self.select_items = [] # 指定した期間内の予定を入れるリスト
# select_items = []

# # 2024/05/28 一旦コメントアウト
# # dt_now = datetime.datetime.now()
# dt_now = datetime.datetime(2024, 5, 24, 8, 00) # テスト用

# start_date = datetime.date(dt_now.year, dt_now.month, dt_now.day)
# end_date = datetime.date(dt_now.year, dt_now.month, dt_now.day + 1)
# strStart = start_date.strftime('%m/%d/%Y %H:%M %p')
# strEnd = end_date.strftime('%m/%d/%Y %H:%M %p')
# sFilter = f"[Start] >= '{strStart}' And [End] <= '{strEnd}'"
# # フィルターを適用し表示
# FilteredItems = items.Restrict(sFilter)
# for item in FilteredItems:
#     if start_date <= item.start.date() <= end_date:
#         # self.select_items.append(item)
#         select_items.append(item)
        
# # print("今日の予定の件数:", len(self.select_items))
# print("今日の予定の件数:", len(select_items))
# # for select_item in self.select_items:
# for select_item in select_items:
#     print(select_item.Start.Format("%Y/%m/%d %H:%M"))