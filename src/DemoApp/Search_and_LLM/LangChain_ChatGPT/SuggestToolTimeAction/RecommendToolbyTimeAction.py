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


# 2024/05/30
# LangChain_ChatGPT\Generate_Text\GoogleColab\main_PredUserNeeds_by_VectorStore_to_Chain.py のクラス化

"""
モデルもどこか一か所にまとめる
"""
llm_4o=ChatOpenAI(
    model="gpt-4o",
    # model="gpt-4o-mini",
    temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
) # チャット特化型モデル
llm_3p5t=ChatOpenAI(
    # model="gpt-4o",
    model="gpt-3.5-turbo",
    temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
) # チャット特化型モデル

class RecommendTool_Context_withTrends_Ver():
    def __init__(self, dt_now_for_time_action, UserActionState):

        self.dt_now_for_time_action = dt_now_for_time_action
        self.UserActionState = UserActionState # "WALKING"
    # RAG.py
    # def getUserTrends_ver2(self):
    def getUserTrends(self):
        time = self.dt_now_for_time_action
        UserAction = self.UserActionState

        from langchain_community.embeddings import HuggingFaceEmbeddings

        # Context_withTrends.pyはこっち
        # self.loader = TextLoader('./SuggestToolTimeAction/userdata_context_version.txt', encoding='utf8')
        # # self.loader = TextLoader('./SuggestToolTimeAction/userdata_context_version_random.txt', encoding='utf8')
        
        # # self.loader = TextLoader('./SuggestToolTimeAction/userdata.txt', encoding='utf8')
        self.loader = TextLoader('./SuggestToolTimeAction/userdata_NotRecommend.txt', encoding='utf8')
        # self.loader = TextLoader('./userdata_NotRecommend.txt', encoding='utf8') # 単体テスト用


        # A こっちは類似度検索してなかった(promptにtxt全部入力するので精度は良い) : Pythonのバージョンを解決できないならこっちの方がいい
        # # text_splitter = CharacterTextSplitter(        
        # #     separator = "\n\n",
        # #     chunk_size = 100,
        # #     chunk_overlap = 0,
        # #     length_function = len,
        # # )
        # self.index = VectorstoreIndexCreator(

        #     embedding= HuggingFaceEmbeddings(), # RAG.pyのバージョンのdefaultはこれのみ
        #     # vectorstore_cls=Chroma,
        #     # embedding=OpenAIEmbeddings(),
        #     # text_splitter=text_splitter,

        # ).from_loaders([self.loader])
        

        # # これまで
        # # results = self.index.vectorstore.similarity_search(f"The txt file is a user trend. If the current day and time is {time} and the user's action state is {UserAction}, what are the possible needs?", k=4)
        # # 今回 (2024/7/18) 類似度検索するので、時刻と行動は必要
        # results = self.index.vectorstore.similarity_search(f"The txt file is a user trend. If the current day and time is {time} and the user's action state is {UserAction}, What possible needs are there?", k=4) #  Generate a context that represents the user's situation in as much detail as possible.", k=4)
        # # 類似度検索なので、時刻と行動のみ入力してみる
        # # results = self.index.vectorstore.similarity_search(f"The current Time is {time} and The user's action state is {UserAction}", k=4)
        # # 以下にしてしまうと類似度検索できないかも
        # # results = self.index.vectorstore.similarity_search("You are an expert in predicting needs. Answer the following: The txt document is a history of the user's “at what time, in what state of activity, and using what features”.Analyze, predict, and itemize these user trends.", k=4)
        
        # context = "\n".join([document.page_content for document in results])
        # # print(f"results:{results}")
        # print(f"context:{context}") # 類似度検索しているのか怪しい:2024/7/18の見解...機能していない。普通にpromptにtxt全部入力しているだけ。
        # # for result in results:
        # #     # メタデータにアクセスしてIDとスコアを取得
        # #     document_id = result.metadata.get('id', 'N/A')
        # #     similarity_score = result.metadata.get('score', 'N/A')
        # #     print(f"Document ID: {document_id}, Similarity Score: {similarity_score}")

        # template = """
        # Please use the following context to answer questions.
        # Context: {context}
        # ---
        # Question: {question}
        # Answer: Let's think step by step.
        # Final output must be in Japanese.
        # """
        # # template = """
        # # Please use the following context to answer questions.
        # # Context: {context}
        # # ---
        # # Question: {question}
        # # Final output must be in Japanese.
        # # """
        # prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)
        # llm_chain = LLMChain(prompt=prompt, llm=llm_4o)
        # # これまで
        # # response = llm_chain.invoke(f"The txt file is a user trend. If the current day and time is {time} and the user's action state is {UserAction}, what are the possible needs?")
        # # 今回 (2024/7/18) # Generate...の文を追加したことで、ユーザーの状況を表すコンテキスト生成までしている
        # # response = llm_chain.invoke(f"If the current day and time is {time} and the user's action state is {UserAction}, What possible needs are there?") # Generate a context that represents the user's situation in as much detail as possible.")
        
        # # 可能なニーズと翻訳されないように文末のpossibleを変更
        # response = llm_chain.invoke(f"If the current date and time is {time} and the user action state is {UserAction}, what needs can be predicted based on past trends?")
        # # response = llm_chain.invoke("What trends do you see in user needs?")
        # # response = llm_chain.invoke(f"Generate a statement that describes the user's situation in as much detail as possible when the current date and time is {time} and the user's action state is {UserAction}.")
        # # print(response['text'])

        # self.UserTrendAnswer = response['text']

        # B これまでの方法
        text_splitter = CharacterTextSplitter(        
            separator = "\n\n",
            chunk_size = 100,
            chunk_overlap = 0,
            length_function = len,
        )
        self.index = VectorstoreIndexCreator(
            vectorstore_cls=Chroma, # Default
            embedding=OpenAIEmbeddings(), # Default # Context_withTrends.pyのやり方にした方がいいかも
            text_splitter=text_splitter, # text_splitterのインスタンスを使っている
        ).from_loaders([self.loader])
        query = f"""
                あなたはニーズを予測する専門家です。以下に答えて。
                txtファイルの文書はユーザーの「どの時刻に、どの行動状態で、どの機能を使用したか」の履歴です。
                このユーザーの傾向を分析・予測し箇条書きでまとめて。
                
                また、現在の日時が{time}で、ユーザーアクションの状態が{UserAction}の場合、過去の傾向からどのようなニーズが予測できるか考えなさい。
                """
                # 最後の一文がないと精度下がる
        self.UserTrendAnswer = self.index.query(query, llm=llm_4o)

        return self.UserTrendAnswer

        
    # ほぼ、getUserTrends()で答えは出ている : UserTrendAnswer=ユーザーのニーズが明らかになっているので、この関数の意味は薄いかも
    # - getUserTrends()にユーザーの傾向しか出力させなければ以下の関数の意味はある
    def getContextDirectly(self):
        # 2024/7/18 test
        template_context = """
        You are the secretary who is close to the user and an expert in predicting the user's current situation.
        Here is what you know about the user.
        [The past trend is {UserTrendAnswer}, The current date and time is {time}, and The current user action state is {UserAction}.]
        Predict and generate only sentences that describe the user's situation in as much detail as possible. (e.g., The user is doing XXX and has needs like XX)
        Also, add one most likely user request to the output.
        Answer in Japanese.
        """
        # prompt = PromptTemplate(template=template_context, input_variables=["context", "question"]).partial(context=context)
        prompt_context = PromptTemplate(template=template_context, input_variables=["UserTrendAnswer", "time", "UserAction"])
        llm_chain_context = LLMChain(prompt=prompt_context, llm=llm_4o)
        # # context_test_res = llm_chain_context.invoke(f"Generate only statements that describe the user's situation in as much detail as possible, which can be predicted if the past trend is {self.UserTrendAnswer}, the current date and time is {time}, and the current user action state is {UserAction}. (e.g., “The user is doing XXX and has needs like XX”)")
        # context_test_res = llm_chain_context.invoke(f"Predict and generate only sentences that describe the user's situation in as much detail as possible. (e.g., The user is doing XXX and has needs like XX)")
        overall_chain = SequentialChain(
            chains=[llm_chain_context],
            input_variables=["UserTrendAnswer", "time", "UserAction"],
            # output_variables=["ContextResponse"], # あくまで辞書型のなんていう要素に出力が格納されるかの変数
            verbose=True,
        )
        context_test_res = overall_chain({
            "UserTrendAnswer" : self.UserTrendAnswer,
            "time" : self.dt_now_for_time_action,
            "UserAction" : self.UserActionState,
        })

        self.ContextDirectly_Answer = context_test_res['text']
        
        return context_test_res['text'] # ['ContextResponse']
        # 2024/7/18 test
    
    # getContextDirectly()と少しかぶっているが、これはいずれgetContextDirectly()のみにしようとしているので重複している
    def getToolAnswer(self):
        prompt_2 = PromptTemplate(
            input_variables=["UserNeeds", "time", "UserAction"],
            
            # 最終的に機能のみ提案するプロンプト
            # template = """
            #         あなたはユーザーに合う機能を提案する専門家です。
            #         ユーザーの傾向は「{UserNeeds}」です。

            #         現在が{time}、ユーザーの行動状態が{UserAction}の場合、どの機能を提案するかこのユーザーの傾向を加味して最終的な提案(Final Answer:)のみを教えて。
            #         あなたが提案できる機能は、
            #         "会議情報", "楽曲再生", "経路検索", "リアルタイム情報検索", "レストラン検索", "ニュース情報", "天気情報",     "何もしない"
            #         です。
            #         ###
            #         Final Answer:
            #         """
            # 2024/7/18
            # template = """
            #         あなたはユーザーを気に掛ける親友であり、ユーザーの現在状況に合う機能を提案する専門家です。
            template = """
                    あなたはユーザーに寄り添う秘書であり、ユーザーの現在状況に合う機能を提案する専門家です。
                    ユーザーの傾向は「{UserNeeds}」です。

                    現在が{time}、ユーザーの行動状態が{UserAction}の場合、ユーザーの気分がよくなるようなねぎらいの声をかけて。その後、どの機能を提案するかこのユーザーの傾向を加味して最終的な提案(Final Answer:)のみを教えて。
                    あなたが提案できる機能は、
                    "会議情報", "楽曲再生", "経路検索", "リアルタイム情報検索", "レストラン検索", "ニュース情報", "天気情報",     "何もしない"
                    です。
                    ###
                    Final Answer:
                    """
        )
        chain_2 = LLMChain(llm=llm_4o, prompt=prompt_2, output_key="output")
        # chain_2 = prompt_2 | llm_4o # 新しいやり方

        # self.overall_chain = SequentialChain(
        overall_chain = SequentialChain(
            chains=[chain_2],
            input_variables=["UserNeeds", "time", "UserAction"],
            # output_variables=["response"], # あくまで辞書型のなんていう要素に出力が格納されるかの変数
            verbose=True,
        )
        response = overall_chain({
            "UserNeeds" : self.UserTrendAnswer,
            "time" : self.dt_now_for_time_action,
            "UserAction" : self.UserActionState,
        })
        RED = '\033[31m'
        YELLOW = '\033[33m'
        END = '\033[0m'
        BOLD = '\033[1m'
        print(BOLD + YELLOW + "\n--------------------------------------------------")
        print("User Needs: \n", response['UserNeeds'])
        print("\n--------------------------------------------------")
        print("time: ", response['time'])
        print("User Action: ", response['UserAction'])
        print("\n--------------------------------------------------")
        print("Output: ", response['output'])
        print("\n--------------------------------------------------" + END)

        Suggested_Tool = response['output']

        return Suggested_Tool
    
    def getToolAnswer_after_getContext(self): # ほぼねぎらいの言葉のために使っている # 2024/7/29
        prompt_2 = PromptTemplate(
            input_variables=["ContextDirectlyAnswer"],
            
            # 最終的に機能のみ提案するプロンプト
            # template = """
            #         あなたはユーザーに合う機能を提案する専門家です。
            #         ユーザーの傾向は「{UserNeeds}」です。

            #         現在が{time}、ユーザーの行動状態が{UserAction}の場合、どの機能を提案するかこのユーザーの傾向を加味して最終的な提案(Final Answer:)のみを教えて。
            #         あなたが提案できる機能は、
            #         "会議情報", "楽曲再生", "経路検索", "リアルタイム情報検索", "レストラン検索", "ニュース情報", "天気情報",     "何もしない"
            #         です。
            #         ###
            #         Final Answer:
            #         """
            # 2024/7/18
            # template = """
            #         あなたはユーザーを気に掛ける親友であり、ユーザーの現在状況に合う機能を提案する専門家です。
            
            
            
            
            # ***** 2024/7/29 *****
            # 以前のやり方
            # template = """
            #         あなたはユーザーに寄り添う秘書であり、ユーザーの現在状況に合う機能を提案する専門家です。
            #         ユーザーの情報が{ContextDirectlyAnswer}の場合、ユーザーの気分がよくなるようなねぎらいの声をかけて。その後、どの機能を提案するかこのユーザーの傾向を加味して最終的な提案(Final Answer:)のみを教えて。
            #         あなたが提案できる機能は、
            #         "会議情報", "楽曲再生", "経路検索", "リアルタイム情報検索", "レストラン検索", "ニュース情報", "天気情報",     "何もしない"
            #         です。
            #         ###
            #         Final Answer:
            #         """
            template = """
                    あなたはユーザーに寄り添う秘書であり、ユーザーの現在状況に合う機能を提案する専門家です。
                    ユーザーの情報が{ContextDirectlyAnswer}の場合、ユーザーの気分がよくなるようなねぎらいの声をかけて。その後、どの機能を提案するかこのユーザーの傾向を加味して最終的な提案(Final Answer:)のみを教えて。
                    """
                    # また、楽曲再生が必要かどうかの判定フラグ[is_musicplayback:]をTrueかFalseで返して。
            # ***** 2024/7/29 *****


        )

        chain_2 = LLMChain(llm=llm_4o, prompt=prompt_2, output_key="output")
        # chain_2 = prompt_2 | llm_4o # 新しいやり方

        # self.overall_chain = SequentialChain(
        overall_chain = SequentialChain(
            chains=[chain_2],
            input_variables=["ContextDirectlyAnswer"],
            # output_variables=["response"], # あくまで辞書型のなんていう要素に出力が格納されるかの変数
            verbose=True,
        )
        response = overall_chain({
            "ContextDirectlyAnswer" : self.ContextDirectly_Answer
        })
        RED = '\033[31m'
        YELLOW = '\033[33m'
        END = '\033[0m'
        BOLD = '\033[1m'
        print(BOLD + YELLOW + "\n--------------------------------------------------")
        print("User ContextDirectlyAnswer: \n", response['ContextDirectlyAnswer'])
        print("\n--------------------------------------------------" + END)

        Suggested_Tool = response['output']

        return Suggested_Tool # 以前のやり方


        # 一旦保留
        # """ 2024/7/29 """
        # # ***** 2024/7/29 *****
        # prompt_3 = PromptTemplate(
        #     input_variables=["SuggestedToolAns"],
        #     template = """
        #             あなたはis_musicplaybackのみを回答しなければならない。
        #             {SuggestedToolAns}を実行する場合、楽曲再生が必要かどうかの判定フラグをTrueかFalseのBool型のみを返して。
        #             is_musicplayback:
        #             """
        # )
        # chain_3 = LLMChain(llm=llm_4o, prompt=prompt_3, output_key="output")
        # # overall_chain_3 = SequentialChain(
        # #     chains=[chain_3],
        # #     input_variables=["SuggestedToolAns"],
        # #     verbose=True,
        # # )
        # # response_flag = overall_chai_3n({
        # #     "SuggestedToolAns" : self.ContextDirectly_Answer
        # # })
        # # RED = '\033[31m'
        # # YELLOW = '\033[33m'
        # # END = '\033[0m'
        # # BOLD = '\033[1m'
        # # print(BOLD + YELLOW + "\n--------------------------------------------------")
        # # print("User Suggested Tool Answer: \n", response_flag['SuggestedToolAns'])
        # # print("\n--------------------------------------------------" + END)
        # response_flag = chain_3.invoke(Suggested_Tool) # 引数が一つの場合はこっちでOK
        # is_MusicPlayback = response_flag['output']
        # print("*****\n[Before] User Suggested Tool Answer: \n", response_flag['output'])
        # if ('True' in is_MusicPlayback) or ('true' in is_MusicPlayback):
        #     is_MusicPlayback = True
        # else:
        #     is_MusicPlayback = False
        # print("*****\n[After] User Suggested Tool Answer: \n", is_MusicPlayback)
        # print("*****\n[After] User Suggested Tool Answer: \n", type(is_MusicPlayback))
        # return Suggested_Tool, is_MusicPlayback # 2024/7/29
        # # # ***** 2024/7/29 *****
        # # """ 2024/7/29 """

        # # prompt_4 = PromptTemplate(
        # #     input_variables=["info"],
        # #     template = """
        # #             あなたが保持するToolの一覧を出力して。
        # #             Tool List:
        # #             また、{info}を実行する場合、Tool ListのうちどのToolが最も適切か教えて。
        # #             Tool:
        # #             """
        # # )
        # # chain_4 = LLMChain(llm=llm_4o, prompt=prompt_4, output_key="output")
        # # response_tool = chain_4.invoke(Suggested_Tool) # Suggested_Tool)
        # # print("*****\n[Tool] \n", response_tool['output'])

        # # return Suggested_Tool, is_MusicPlayback # 2024/7/29
    

    # コンテキスト生成
    # Context_withTrends.pyから移植
    """
    # 正直、getToolAnswerのprompt部分だけ、書き換えればあとはやっていることは同じ
    (異なるのはover_allを使うことで、responseの中のテキスト名を任意の名前に設定できる：response['UserNeeds']など)
    (prompt_templateの部分は下記中が異なるくらいで、やっていることは同じ)
    """
    def getContext(self):

        # sensor_data = self.randomize_sensor_data(self.sensor_data)
        self.sensor_data = {
            "time":self.dt_now_for_time_action,
            "user_action": self.UserActionState,
            "user_trends": self.UserTrendAnswer,
        }
        # "sound": "quiet",
        prompt_template = """
        Based on the following sensor data, prompts representing the user's current intentions and needs are generated in a straightforward manner:

        - Time: {time}
        - User Action: {user_action}
        - User Trends: {user_trends}

        Provide a detailed description of the user's likely intention or need.
        Final output must be in Japanese.
        """
        # - Sound: {sound}
        # prompt = PromptTemplate(template=prompt_template, input_variables=["location", "user_action", "sound", "heart_rate", "day_and_time", "user_trends"])
        prompt = PromptTemplate(template=prompt_template, input_variables=["time", "user_action", "user_trends"])

        # チェーンを設定
        chain = LLMChain(llm=llm_4o, prompt=prompt) # chain = prompt | llm_4o

        # プロンプトを生成
        print("Sensor Data:", self.sensor_data)
        result = chain.invoke(self.sensor_data)
        print("Generated Prompt: ", result['text']) # ["content"])

        return result['text']


class RecommendTool():
    def __init__(self, dt_now_for_time_action, UserActionState):
        # loader = TextLoader('./UserAction2_mini_loop.txt', encoding='utf8')

        # loader = TextLoader('./SuggestToolTimeAction/userdata.txt', encoding='utf8')
        loader = TextLoader('./SuggestToolTimeAction/userdata_NotRecommend.txt', encoding='utf8')
        # loader = TextLoader('./userdata.txt', encoding='utf8') # 単体テスト用

        # 100文字のチャンクで区切る
        text_splitter = CharacterTextSplitter(        
            separator = "\n\n",
            chunk_size = 100,
            chunk_overlap = 0,
            length_function = len,
        )
        self.index = VectorstoreIndexCreator(
            vectorstore_cls=Chroma, # Default
            embedding=OpenAIEmbeddings(), # Default # Context_withTrends.pyのやり方にした方がいいかも
            text_splitter=text_splitter, # text_splitterのインスタンスを使っている
        ).from_loaders([loader])

        self.dt_now_for_time_action = dt_now_for_time_action
        self.UserActionState = UserActionState # "WALKING"

    def getUserTrends(self):
        query = """
                あなたはニーズを予測する専門家です。以下に答えて。
                txtファイルの文書はユーザーの「どの時刻に、どの行動状態で、どの機能を使用したか」の履歴です。
                このユーザーの傾向を分析・予測し箇条書きでまとめて。
                """
                # 2024/6/17 一旦コメントアウト
                # (例：朝の時間帯(xx:xx - xx:xx): ,午前中(xx:xx - xx:xx), 昼の時間帯(xx:xx - xx:xx), 午後(xx:xx - xx:xx), 夕方(xx:xx - xx:xx), 夜の時間帯(xx:xx - xx:xx))
                # """
                
                # (例1：朝の時間帯(xx:xx - xx:xx): ,午前中(xx:xx - xx:xx), 昼の時間帯(xx:xx - xx:xx), 午後(xx:xx - xx:xx), 夕方(xx:xx - xx:xx), 夜の時間帯(xx:xx - xx:xx))
                # (例2：行動状態ごとの使う機能の傾向)
                # """
        self.UserTrendAnswer = self.index.query(query, llm=llm_4o)
        return self.UserTrendAnswer # classに結果は保持されるからなくてもいいかも



        
    def getToolAnswer(self):
        prompt_2 = PromptTemplate(
            input_variables=["UserNeeds", "time", "UserAction"],
            # ユーザーの傾向を加味して機能を提案するLLM
            # template =  """
            #             あなたはユーザーに合う機能を提案する専門家です。
            #             ユーザーの傾向は「{UserNeeds}」です。

            #             現在が{time}、ユーザーの行動状態が{UserAction}の場合、どの機能を提案するかこのユーザーの傾向を加味して予測して。
            #             その際、各機能の提案する確率（の高いものを最大三つまで示し）と最終的な提案(Final Answer:)、その理由も教えて。
            #             あなたが提案できる機能は、
            #             "会議情報", "楽曲再生", "経路検索", "リアルタイム情報検索", "レストラン検索", "ニュース情報", "天気情報"
            #             です。
            #             """
            template = """
                    あなたはユーザーに合う機能を提案する専門家です。
                    ユーザーの傾向は「{UserNeeds}」です。

                    現在が{time}、ユーザーの行動状態が{UserAction}の場合、どの機能を提案するかこのユーザーの傾向を加味して最終的な提案(Final Answer:)のみを教えて。
                    あなたが提案できる機能は、
                    "会議情報", "楽曲再生", "経路検索", "リアルタイム情報検索", "レストラン検索", "ニュース情報", "天気情報",     "何もしない"
                    です。
                    ###
                    Final Answer:
                    """
            # 機能のみ提案させるバージョン


    
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
            # 上記二つのいずれかがよさそう
        )
        chain_2 = LLMChain(llm=llm_4o, prompt=prompt_2, output_key="output")
        # chain_2 = prompt_2 | llm_4o # 新しいやり方

        # self.overall_chain = SequentialChain(
        overall_chain = SequentialChain(
            chains=[chain_2],
            input_variables=["UserNeeds", "time", "UserAction"],
            # output_variables=["response"], # あくまで辞書型のなんていう要素に出力が格納されるかの変数
            verbose=True,
        )
        response = overall_chain({
            "UserNeeds" : self.UserTrendAnswer,
            "time" : self.dt_now_for_time_action,
            "UserAction" : self.UserActionState,
        })
        RED = '\033[31m'
        YELLOW = '\033[33m'
        END = '\033[0m'
        BOLD = '\033[1m'
        print(BOLD + YELLOW + "\n--------------------------------------------------")
        print("User Needs: \n", response['UserNeeds'])
        print("\n--------------------------------------------------")
        print("time: ", response['time'])
        print("User Action: ", response['UserAction'])
        print("\n--------------------------------------------------")
        print("Output: ", response['output'])
        print("\n--------------------------------------------------" + END)

        Suggested_Tool = response['output']

        return Suggested_Tool

if __name__ == "__main__":
    import datetime
    # dt_now_for_time_action = datetime.datetime(2024, 5, 24, 11, 41) # 11時41分"
    # margin = 5
    # margin = datetime.timedelta(minutes=margin)
    dt_now_for_time_action = datetime.timedelta(hours=8, minutes=36) # 経路案内
    # dt_now_for_time_action = datetime.timedelta(hours=10, minutes=41) # 経路案内
    dt_now_for_time_action = datetime.timedelta(hours=11, minutes=58) # レストラン検索(STABLE:何もしない)
    dt_now_for_time_action = datetime.timedelta(hours=12, minutes=5) # レストラン検索(STABLE:何もしない)
    # dt_now_for_time_action = datetime.timedelta(hours=17, minutes=28) # 楽曲再生
    # dt_now_for_time_action = datetime.timedelta(hours=17, minutes=58) # 経路案内
    
    # UserActionState = "WALKING"
    UserActionState = "STABLE"
    # recommend_tool = RecommendTool(dt_now_for_time_action, UserActionState)
    recommend_tool = RecommendTool_Context_withTrends_Ver(dt_now_for_time_action, UserActionState)

    recommend_tool.getUserTrends()
    # A 機能提案が先
    suggested_tool = recommend_tool.getToolAnswer()
    print("\n--------------------------------------------------")
    print(suggested_tool)
    print("--------------------------------------------------")

    # # Context_withTrends.pyから移植
    # context = recommend_tool.getContext()

    context_test_res = recommend_tool.getContextDirectly()
    print("\n---------------------- [test response] ----------------------------")
    print(context_test_res)
    print("--------------------------------------------------")

    # B 機能提案は後
    suggested_tool = recommend_tool.getToolAnswer_after_getContext()
    print("\n--------------------------------------------------")
    print(suggested_tool)
    print("--------------------------------------------------")