import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain_openai import ChatOpenAI
# from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
# from langchain.document_loaders import TextLoader # txtはこっち
from langchain_community.document_loaders import TextLoader
# from langchain.document_loaders import CSVLoader # csvはこっち
from langchain_community.document_loaders import CSVLoader
from langchain.chains import SequentialChain


# 2024/05/30
# LangChain_ChatGPT\Generate_Text\GoogleColab\main_PredUserNeeds_by_VectorStore_to_Chain.py のクラス化

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
    def __init__(self, dt_now_for_time_action, UserActionState):
    #     # loader = TextLoader('./UserAction2_mini_loop.txt', encoding='utf8')
    #     # loader = TextLoader('./SuggestToolTimeAction/userdata_context_version.txt', encoding='utf8')
    #     loader = TextLoader('./SuggestToolTimeAction/userdata_context_version_random.txt', encoding='utf8')

    #     # loader = TextLoader('./userdata.txt', encoding='utf8') # 単体テスト用
    #     # 100文字のチャンクで区切る
    #     text_splitter = CharacterTextSplitter(        
    #         separator = "\n\n",
    #         chunk_size = 100,
    #         chunk_overlap = 0,
    #         length_function = len,
    #     )
    #     self.index = VectorstoreIndexCreator(
    #         vectorstore_cls=Chroma, # Default
    #         embedding=OpenAIEmbeddings(), # Default
    #         text_splitter=text_splitter, # text_splitterのインスタンスを使っている
    #     ).from_loaders([loader])

        self.dt_now_for_time_action = dt_now_for_time_action
        self.UserActionState = UserActionState # "WALKING"
        
    # def getUserTrends(self):
    #     time = self.dt_now_for_time_action
    #     UserAction = self.UserActionState

    #     # # query = f"""
    #     # #         あなたはニーズを予測する専門家です。以下に答えて。
    #     # #         txtファイルの文書はユーザーの「どの時刻に、どの行動状態で、どの機能を使用したか」の履歴です。
    #     # #         このユーザーの傾向を分析・予測し箇条書きでまとめて。
                
                
    #     # #         また、現在の曜日と時刻が{time}、ユーザーの行動状態が{UserAction}の場合どんなニーズが考えられますか。
    #     # #         """
    #     # #         # 2024/6/17 一旦コメントアウト
    #     # #         # (例：朝の時間帯(xx:xx - xx:xx): ,午前中(xx:xx - xx:xx), 昼の時間帯(xx:xx - xx:xx), 午後(xx:xx - xx:xx), 夕方(xx:xx - xx:xx), 夜の時間帯(xx:xx - xx:xx))
    #     # #         # """
                
    #     # #         # (例1：朝の時間帯(xx:xx - xx:xx): ,午前中(xx:xx - xx:xx), 昼の時間帯(xx:xx - xx:xx), 午後(xx:xx - xx:xx), 夕方(xx:xx - xx:xx), 夜の時間帯(xx:xx - xx:xx))
    #     # #         # (例2：行動状態ごとの使う機能の傾向)
    #     # #         # """
    #     # query = f"The txt file is a user trend. If the current day and time is {time} and the user's action state is {UserAction}, what are the possible needs?"
    #     # self.UserTrendAnswer = self.index.query(query, llm=llm_4o)
    #     # return self.UserTrendAnswer

    #     results = self.index.vectorstore.similarity_search(f"The txt file is a user trend. If the current day and time is {time} and the user's action state is {UserAction}, what are the possible needs?", k=4)
    #     context = "\n".join([document.page_content for document in results])
    #     template = """
    #     Please use the following context to answer questions.
    #     Context: {context}
    #     ---
    #     Question: {question}
    #     Answer: Let's think step by step.
    #     Final output must be in Japanese.
    #     """
    #     prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)
    #     llm_chain = LLMChain(prompt=prompt, llm=llm_4o)
    #     response = llm_chain.invoke(f"The txt file is a user trend. If the current day and time is {time} and the user's action state is {UserAction}, what are the possible needs?")
    #     return response['text']
    
    # def getUserTrends_original(self):
    #     query = f"""
    #             あなたはニーズを予測する専門家です。以下に答えて。
    #             txtファイルの文書はユーザーの「どの時刻に、どの行動状態で、どの機能を使用したか」の履歴です。
    #             このユーザーの傾向を分析・予測し箇条書きでまとめて。
    #             """
    #     self.UserTrendAnswer = self.index.query(query, llm=llm_4o)
    #     return self.UserTrendAnswer # classに結果は保持されるからなくてもいいかも
    
    # RAG.py
    def getUserTrends_ver2(self):
        time = self.dt_now_for_time_action
        UserAction = self.UserActionState

        from langchain_community.embeddings import HuggingFaceEmbeddings

        # どっちでもあまり変わらない
        self.loader = TextLoader('./SuggestToolTimeAction/userdata_context_version.txt', encoding='utf8')
        # self.loader = TextLoader('./SuggestToolTimeAction/userdata_context_version_random.txt', encoding='utf8')
        # text_splitter = CharacterTextSplitter(        
        #     separator = "\n\n",
        #     chunk_size = 100,
        #     chunk_overlap = 0,
        #     length_function = len,
        # )
        self.index = VectorstoreIndexCreator(

            embedding= HuggingFaceEmbeddings() # RAG.pyのバージョンのdefaultはこれのみ
            # vectorstore_cls=Chroma,
            # embedding=OpenAIEmbeddings(),
            # text_splitter=text_splitter,

        ).from_loaders([self.loader])
        

        # これまで
        # results = self.index.vectorstore.similarity_search(f"The txt file is a user trend. If the current day and time is {time} and the user's action state is {UserAction}, what are the possible needs?", k=4)
        # 今回 (2024/7/18)
        results = self.index.vectorstore.similarity_search(f"The txt file is a user trend. If the current day and time is {time} and the user's action state is {UserAction}, What possible needs are there?", k=4) #  Generate a context that represents the user's situation in as much detail as possible.", k=4)

        context = "\n".join([document.page_content for document in results])
        # print(f"{context}")
        template = """
        Please use the following context to answer questions.
        Context: {context}
        ---
        Question: {question}
        Answer: Let's think step by step.
        Final output must be in Japanese.
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)
        llm_chain = LLMChain(prompt=prompt, llm=llm_4o)
        # これまで
        # response = llm_chain.invoke(f"The txt file is a user trend. If the current day and time is {time} and the user's action state is {UserAction}, what are the possible needs?")
        # 今回 (2024/7/18)
        response = llm_chain.invoke(f"If the current day and time is {time} and the user's action state is {UserAction}, What possible needs are there? Generate a context that represents the user's situation in as much detail as possible.")

        # print(response['text'])

        return response['text']