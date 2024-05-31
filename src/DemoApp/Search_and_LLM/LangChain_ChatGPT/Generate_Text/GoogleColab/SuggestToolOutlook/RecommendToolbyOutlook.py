import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
# from langchain.document_loaders import TextLoader # txtはこっち
from langchain_community.document_loaders import TextLoader
# from langchain.document_loaders import CSVLoader # csvはこっち
from langchain_community.document_loaders import CSVLoader
from langchain.chains import SequentialChain


# 2024/05/29
# GoogleColab\Recommend_Tool_bySchedule.py のクラス化

"""
モデルもどこか一か所にまとめる
"""
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

class RecommendTool():
    def __init__(self):

        # loader = TextLoader('./UserAction2_mini_loop_Schedule.txt', encoding='utf8')
        loader = TextLoader('./userdata_schedule.txt', encoding='utf8')
        # 100文字のチャンクで区切る
        text_splitter = CharacterTextSplitter(        
            separator = "\n\n",
            chunk_size = 100,
            chunk_overlap = 0,
            length_function = len,
        )
        self.index = VectorstoreIndexCreator(
            vectorstore_cls=Chroma, # Default
            embedding=OpenAIEmbeddings(), # Default
            text_splitter=text_splitter, # text_splitterのインスタンスを使っている
        ).from_loaders([loader])


        prompt_2 = PromptTemplate(
            input_variables=["UserNeeds", "schedule", "UserAction"],
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
            """
            ユーザーの傾向がないとあまり精度良くない
            """
        )
        chain_2 = LLMChain(llm=llm_4o, prompt=prompt_2, output_key="output") # chain_2 = prompt_2 | llm_4o # 新しいやり方
        self.overall_chain = SequentialChain(
            chains=[chain_2],
            input_variables=["UserNeeds", "schedule", "UserAction"],
            # output_variables=["response"], # あくまで辞書型のなんていう要素に出力が格納されるかの変数
            verbose=True,
        )

        self.UserActionState = "WALKING"

    def getUserTrends(self):
        query = """
                あなたはニーズを予測する専門家です。以下に答えて。
                txtファイルの文書はユーザーの「どんな予定の時に、どの行動状態で、どの機能を使用したか」の履歴です。
                このユーザーの傾向を分析・予測し箇条書きでまとめて。
                """
        self.answer = self.index.query(query, llm=llm_4o)
        # return self.answer # classに結果は保持されるからなくてもいいかも


    def getToolAnswer(self): # , UserActionState):
        

        response = self.overall_chain({
            "UserNeeds" : self.answer,
            "schedule" : "通勤",
            # "schedule" : "昼食",
            # "schedule" : "定例",
            # "schedule" : "進捗確認",
            # "schedule" : "面談",
            # "UserAction" : "STABLE",
            "UserAction" : self.UserActionState,
        })
        
        Suggested_Tool = response['output']

        return Suggested_Tool

if __name__ == "__main__":

    recommend_tool = RecommendTool()
    recommend_tool.getUserTrends()
    suggested_tool = recommend_tool.getToolAnswer()

    print("\n--------------------------------------------------")
    print(suggested_tool)
    print("--------------------------------------------------")