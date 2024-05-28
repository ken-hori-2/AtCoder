import os
from dotenv import load_dotenv
load_dotenv()

# GoogleColab\main_PredUserNeeds_by_VectorStore_to_Chain.py のユーザーの傾向の要約無しバージョン

"""
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
# loader = TextLoader('./UserAction2_mini_loop.txt', encoding='utf8')
loader = TextLoader('./userdata.txt', encoding='utf8')
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

# query = """
#         あなたはニーズを予測する専門家です。以下に答えて。
#         txtファイルの文書はユーザーの「どの時刻に、どの行動状態で、どの機能を使用したか」の履歴です。
#         このユーザーの傾向を分析・予測し箇条書きでまとめて。
#         (例：朝の時間帯(xx:xx - xx:xx): ,午前中(xx:xx - xx:xx), 昼の時間帯(xx:xx - xx:xx), 午後(xx:xx - xx:xx), 夕方(xx:xx - xx:xx), 夜の時間帯(xx:xx - xx:xx))
#         """
#         # (例1：朝の時間帯(xx:xx - xx:xx): ,午前中(xx:xx - xx:xx), 昼の時間帯(xx:xx - xx:xx), 午後(xx:xx - xx:xx), 夕方(xx:xx - xx:xx), 夜の時間帯(xx:xx - xx:xx))
#         # (例2：行動状態ごとの使う機能の傾向)
#         # """
# answer = index.query(query, llm=llm_4o)

# ユーザーの傾向を要約せずに直接入力するバージョン
query = """
        あなたはニーズを予測する専門家です。以下に答えて。
        txtファイルの文書はユーザーの「どの時刻に、どの行動状態で、どの機能を使用したか」の履歴です。
        このユーザーの行動履歴を時系列順に並び替えて箇条書きで教えて。
        """
        # このユーザーの行動履歴を時系列順に並び替えて箇条書きでまとめて教えて。
        # """
answer = index.query(query, llm=llm_4o)



prompt_2 = PromptTemplate(
    input_variables=["UserNeeds", "time", "UserAction"],
    # template =  """
    #             あなたはユーザーに合う機能を提案する専門家です。
    #             ユーザーの傾向は「{UserNeeds}」です。

    #             現在が{time}、ユーザーの行動状態が{UserAction}の場合、どの機能を提案するかこのユーザーの傾向を加味して予測して。
    #             その際、各機能の提案する確率と最終的な提案(Final Answer:)、その理由も教えて。
    #             あなたが提案できる機能は、
    #             "会議情報", "楽曲再生", "経路検索", "リアルタイム情報検索", "レストラン検索", "ニュース情報", "天気情報"
    #             です。
    #             """
    
    # ユーザーの傾向から提案タイミングを予測するLLM
    # template =  """
    #             あなたはユーザーに合う機能を適切なタイミングで提案する専門家です。
    #             ユーザーの傾向は「{UserNeeds}」です。
    #             ユーザーの傾向からユーザーが機能を使いそうなタイミングを時刻を指定して教えて。時刻が連続する場合はその時刻周辺で最も適切な時刻を指定して。
    #             (例 ... 時刻：) 
    #             """ # 同じ機能が連続する場合はマージして。数分単位で連続する場合
    #             # ユーザーの傾向からユーザーのニーズが発生しそうなタイミングを時刻を指定して教えて。時刻が連続する場合は最も適切だと考えられる時刻を指定して。
    #             # 時刻：
    #             # """
    #             # 上記二つのいずれかがよさそう
    template =  """
                あなたはユーザーに合う機能を適切なタイミングで提案する専門家です。
                ユーザーの行動履歴は「{UserNeeds}」です。
                ユーザーが機能を使いそうなタイミングを時刻を指定して教えて。時刻が連続する場合はその時刻周辺で最も適切な時刻を指定して。
                (例 ... 時刻：) 
                """ # 同じ機能が連続する場合はマージして。数分単位で連続する場合
                # ユーザーの傾向からユーザーのニーズが発生しそうなタイミングを時刻を指定して教えて。時刻が連続する場合は最も適切だと考えられる時刻を指定して。
                # 時刻：
                # """
                # 上記二つのいずれかがよさそう
)
chain_2 = LLMChain(llm=llm_4o, prompt=prompt_2, output_key="output")
# chain_2 = prompt_2 | llm_4o # 新しいやり方



from langchain.chains import SequentialChain
overall_chain = SequentialChain(
    chains=[chain_2],
    input_variables=["UserNeeds", "time", "UserAction"],
    # output_variables=["response"], # あくまで辞書型のなんていう要素に出力が格納されるかの変数
    verbose=True,
)
response = overall_chain({
    "UserNeeds" : answer,
    "time" : "11時41分",
    "UserAction" : "WALKING",
})
print("\n--------------------------------------------------")
print("User Needs: \n", response['UserNeeds'])
print("\n--------------------------------------------------")
print("time: ", response['time'])
print("User Action: ", response['UserAction'])
print("\n--------------------------------------------------")
print("Output: ", response['output'])
print("\n--------------------------------------------------")


""" ################################################## GoogleColab\main_LangChain_Indexes_UserAction_類似度検索_UserNeeds.py """