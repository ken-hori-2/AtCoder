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

class RecommendTool_Context_withTrends_Ver():
    def __init__(self, UserSchedule, UserActionState):

        # self.dt_now_for_time_action = dt_now_for_time_action
        self.schedule = UserSchedule.getScheduleContents()
        self.UserActionState = UserActionState # "WALKING"
    # RAG.py
    # def getUserTrends_ver2(self):
    def getUserTrends(self):
        # time = self.dt_now_for_time_action
        # schedule = self.schedule
        UserAction = self.UserActionState

        from langchain_community.embeddings import HuggingFaceEmbeddings

        # Context_withTrends.pyはこっち
        # self.loader = TextLoader('./SuggestToolTimeAction/userdata_context_version.txt', encoding='utf8')
        # # self.loader = TextLoader('./SuggestToolTimeAction/userdata_context_version_random.txt', encoding='utf8')
        
        # # self.loader = TextLoader('./SuggestToolTimeAction/userdata.txt', encoding='utf8')
        self.loader = TextLoader('./SuggestToolOutlook/userdata_schedule.txt', encoding='utf8')
        # self.loader = TextLoader('./userdata_schedule.txt', encoding='utf8') # 単体テスト用 (SuggestToolOutlook\GeneratePrompt_具体的な提案の検討.py)


        # A こっちは類似度検索してなかった(promptにtxt全部入力するので精度は良い) : Pythonのバージョンを解決できないならこっちの方がいい
        # # text_splitter = CharacterTextSplitter(        
        # #     separator = "\n\n",
        # #     chunk_size = 100,
        # #     chunk_overlap = 0,
        # #     length_function = len,
        # # )
        # self.index = VectorstoreIndexCreator(

        #     embedding= HuggingFaceEmbeddings() # RAG.pyのバージョンのdefaultはこれのみ
        #     # vectorstore_cls=Chroma,
        #     # embedding=OpenAIEmbeddings(),
        #     # text_splitter=text_splitter,

        # ).from_loaders([self.loader])
        

        # # これまで
        # # results = self.index.vectorstore.similarity_search(f"The txt file is the user's trend. If the current schedule is {self.schedule} and the user action state is {UserAction}, what are the possible needs?", k=4)
        # # 今回 (2024/7/18)
        # results = self.index.vectorstore.similarity_search(f"The txt file is the user's trend. If the current schedule is {self.schedule} and the user action state is {UserAction}, What possible needs are there?", k=4) #  Generate a context that represents the user's situation in as much detail as possible.", k=4)
        # context = "\n".join([document.page_content for document in results])
        # # print(f"results:{results}")
        # # print(f"context:{context}")

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
        # # response = llm_chain.invoke(f"The txt file is the user's trend. If the current schedule is {self.schedule} and the user action state is {UserAction}, what are the possible needs?")
        # # 今回 (2024/7/18) # Generate...の文を追加したことで、ユーザーの状況を表すコンテキスト生成までしている
        # # response = llm_chain.invoke(f"If the current schedule is {self.schedule} and the user action state is {UserAction}, What possible needs are there? Generate a context that represents the user's situation in as much detail as possible.")
        
        # # 可能なニーズと翻訳されないように文末のpossibleを変更
        # response = llm_chain.invoke(f"If the current schedule is {self.schedule} and the user action state is {UserAction}, what needs can be predicted based on past trends?")
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
                txtファイルの文書はユーザーの「どんな予定の時に、どの行動状態で、どの機能を使用したか」の履歴です。
                このユーザーの傾向を分析・予測し箇条書きでまとめて。
                
                また、現在の予定が{self.schedule}で、ユーザーアクションの状態が{UserAction}の場合、過去の傾向からどのようなニーズが予測できるか考えなさい。
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
        [The past trend is {UserTrendAnswer}, The current schedule is {schedule}, and The current user action state is {UserAction}.]
        Predict and generate only sentences that describe the user's situation in as much detail as possible. (e.g., The user is doing XXX and has needs like XX)
        Also, add one most likely user request to the output.
        Answer in Japanese.
        """
        # prompt = PromptTemplate(template=template_context, input_variables=["context", "question"]).partial(context=context)
        prompt_context = PromptTemplate(template=template_context, input_variables=["UserTrendAnswer", "schedule", "UserAction"])
        llm_chain_context = LLMChain(prompt=prompt_context, llm=llm_4o)
        # # context_test_res = llm_chain_context.invoke(f"Generate only statements that describe the user's situation in as much detail as possible, which can be predicted if the past trend is {self.UserTrendAnswer}, the current date and time is {time}, and the current user action state is {UserAction}. (e.g., “The user is doing XXX and has needs like XX”)")
        # context_test_res = llm_chain_context.invoke(f"Predict and generate only sentences that describe the user's situation in as much detail as possible. (e.g., The user is doing XXX and has needs like XX)")
        overall_chain = SequentialChain(
            chains=[llm_chain_context],
            input_variables=["UserTrendAnswer", "schedule", "UserAction"],
            # output_variables=["ContextResponse"], # あくまで辞書型のなんていう要素に出力が格納されるかの変数
            verbose=True,
        )
        context_test_res = overall_chain({
            "UserTrendAnswer" : self.UserTrendAnswer,
            "schedule" : self.schedule,
            "UserAction" : self.UserActionState,
        })
        return context_test_res['text'] # ['ContextResponse']
        # 2024/7/18 test

    # getContextDirectly()と少しかぶっているが、これはいずれgetContextDirectly()のみにしようとしているので重複している
    # # def getToolAnswer(self, check_schedule): # , UserActionState):
    def getToolAnswer(self):
        # getToolAnswer()に移動する？
        prompt_2 = PromptTemplate(
            input_variables=["UserNeeds", "schedule", "UserAction"],
            # 最終的に機能のみ提案するプロンプト
            # template = """
            #         あなたはユーザーに合う機能を提案する専門家です。
            #         ユーザーの傾向は「{UserNeeds}」です。

            #         現在の予定が{schedule}、ユーザーの行動状態が{UserAction}の場合、どの機能を提案するかこのユーザーの傾向を加味して最終的な提案(Final Answer:)のみを教えて。
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

                    現在の予定が{schedule}、ユーザーの行動状態が{UserAction}の場合、ユーザーの気分がよくなるようなねぎらいの声をかけて。その後、どの機能を提案するかこのユーザーの傾向を加味して最終的な提案(Final Answer:)のみを教えて。
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
            input_variables=["UserNeeds", "schedule", "UserAction"],
            # output_variables=["response"], # あくまで辞書型のなんていう要素に出力が格納されるかの変数
            verbose=True,
        )
        # initから移動⇑

        

        # schedule_contents = self.schedule # check_schedule.getScheduleContents()
        # # print(schedule_contents)
        # if schedule_contents:
        #     print("True")
        #     print("現在の予定：", schedule_contents)
        if self.schedule:
            print("True")
            print("self.schedule 現在の予定：", self.schedule)
        

        # response = self.overall_chain({
        response = overall_chain({
            "UserNeeds" : self.UserTrendAnswer, # ただ、こっちはAgent化できないかも(VectorStoreを使うことを考えた場合)
            # ただ、ユーザーの傾向だけ別のLLMで出力させて、その結果とEdgeの検出結果をAgentに入力すればできそう


            # "schedule" : schedule_contents, # "通勤", # ここはAgentに今の予定は何？と聞いてもいい？(outlookのツールを使ってくれる)
            "schedule" : self.schedule,
            # プロンプトを生成してくれるこの関数もツール化する？

            
            # "schedule" : "昼食",
            # "schedule" : "定例",
            # "schedule" : "進捗確認",
            # "schedule" : "面談",
            # "UserAction" : "STABLE",
            "UserAction" : self.UserActionState, # 行動検出をAgentに組み込めば統合できる
        })
        RED = '\033[31m'
        YELLOW = '\033[33m'
        END = '\033[0m'
        BOLD = '\033[1m'
        print(BOLD + YELLOW + "\n--------------------------------------------------")
        print("User Needs: \n", response['UserNeeds'])
        print("\n--------------------------------------------------")
        print("schedule: ", response['schedule'])
        print("User Action: ", response['UserAction'])
        print("\n--------------------------------------------------")
        print("Output: ", response['output'])
        print("\n--------------------------------------------------" + END)

        Suggested_Tool = response['output']

        return Suggested_Tool
        # return Suggested_Tool, self.UserTrendAnswer # 傾向も渡す場合
    

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
            "schedule" : self.schedule,
            "user_action": self.UserActionState,
            "user_trends": self.UserTrendAnswer,
        }
        # "sound": "quiet",
        prompt_template = """
        Based on the following sensor data, prompts representing the user's current intentions and needs are generated in a straightforward manner:

        - Schedule: {schedule}
        - User Action: {user_action}
        - User Trends: {user_trends}

        Provide a detailed description of the user's likely intention or need.
        Final output must be in Japanese.
        """
        # - Sound: {sound}
        # prompt = PromptTemplate(template=prompt_template, input_variables=["location", "user_action", "sound", "heart_rate", "day_and_time", "user_trends"])
        prompt = PromptTemplate(template=prompt_template, input_variables=["schedule", "user_action", "user_trends"])

        # チェーンを設定
        chain = LLMChain(llm=llm_4o, prompt=prompt) # chain = prompt | llm_4o

        # プロンプトを生成
        print("Sensor Data:", self.sensor_data)
        result = chain.invoke(self.sensor_data)
        print("Generated Prompt: ", result['text']) # ["content"])

        return result['text']


class RecommendTool():
    def __init__(self, UserActionState):

        # loader = TextLoader('./UserAction2_mini_loop_Schedule.txt', encoding='utf8')
        # loader = TextLoader('./userdata_schedule.txt', encoding='utf8')
        import sys
        # JudgeModel.pyから呼び出すときに必要
        sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
        loader = TextLoader('./SuggestToolOutlook/userdata_schedule.txt', encoding='utf8')
        # loader = TextLoader('./userdata_schedule.txt', encoding='utf8') # 単体テスト用 (SuggestToolOutlook\GeneratePrompt_具体的な提案の検討.py)

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


        # # getToolAnswer()に移動する？
        # prompt_2 = PromptTemplate(
        #     input_variables=["UserNeeds", "schedule", "UserAction"],
        #     template = """
        #             あなたはユーザーに合う機能を提案する専門家です。
        #             ユーザーの傾向は「{UserNeeds}」です。

        #             現在の予定が{schedule}、ユーザーの行動状態が{UserAction}の場合、どの機能を提案するかこのユーザーの傾向を加味して最終的な提案(Final Answer:)のみを教えて。
        #             あなたが提案できる機能は、
        #             "会議情報", "楽曲再生", "経路検索", "リアルタイム情報検索", "レストラン検索", "ニュース情報", "天気情報"
        #             です。
        #             ###
        #             Final Answer:
        #             """
        #     """
        #     ユーザーの傾向がないとあまり精度良くない
        #     """
        # )
        # chain_2 = LLMChain(llm=llm_4o, prompt=prompt_2, output_key="output") # chain_2 = prompt_2 | llm_4o # 新しいやり方
        # self.overall_chain = SequentialChain(
        #     chains=[chain_2],
        #     input_variables=["UserNeeds", "schedule", "UserAction"],
        #     # output_variables=["response"], # あくまで辞書型のなんていう要素に出力が格納されるかの変数
        #     verbose=True,
        # )

        self.UserActionState = UserActionState # "WALKING"

    def getUserTrends(self):
        query = """
                あなたはニーズを予測する専門家です。以下に答えて。
                txtファイルの文書はユーザーの「どんな予定の時に、どの行動状態で、どの機能を使用したか」の履歴です。
                このユーザーの傾向を分析・予測し箇条書きでまとめて。
                """
        self.UserTrendAnswer = self.index.query(query, llm=llm_4o)
        return self.UserTrendAnswer # classに結果は保持されるからなくてもいいかも


    def getToolAnswer(self, check_schedule): # , UserActionState):
        # getToolAnswer()に移動する？
        prompt_2 = PromptTemplate(
            input_variables=["UserNeeds", "schedule", "UserAction"],
            template = """
                    あなたはユーザーに合う機能を提案する専門家です。
                    ユーザーの傾向は「{UserNeeds}」です。

                    現在の予定が{schedule}、ユーザーの行動状態が{UserAction}の場合、どの機能を提案するかこのユーザーの傾向を加味して最終的な提案(Final Answer:)のみを教えて。
                    あなたが提案できる機能は、
                    "会議情報", "楽曲再生", "経路検索", "リアルタイム情報検索", "レストラン検索", "ニュース情報", "天気情報",     "何もしない"
                    です。
                    ###
                    Final Answer:
                    """
            """
            ユーザーの傾向がないとあまり精度良くない
            """
        )
        chain_2 = LLMChain(llm=llm_4o, prompt=prompt_2, output_key="output")
        # chain_2 = prompt_2 | llm_4o # 新しいやり方

        # self.overall_chain = SequentialChain(
        overall_chain = SequentialChain(
            chains=[chain_2],
            input_variables=["UserNeeds", "schedule", "UserAction"],
            # output_variables=["response"], # あくまで辞書型のなんていう要素に出力が格納されるかの変数
            verbose=True,
        )
        # initから移動⇑

        

        schedule_contents = check_schedule.getScheduleContents()
        # print(schedule_contents)
        if schedule_contents:
            print("True")
            print("現在の予定：", schedule_contents)
        

        # response = self.overall_chain({
        response = overall_chain({
            "UserNeeds" : self.UserTrendAnswer, # ただ、こっちはAgent化できないかも(VectorStoreを使うことを考えた場合)
            # ただ、ユーザーの傾向だけ別のLLMで出力させて、その結果とEdgeの検出結果をAgentに入力すればできそう


            "schedule" : schedule_contents, # "通勤", # ここはAgentに今の予定は何？と聞いてもいい？(outlookのツールを使ってくれる)
            # プロンプトを生成してくれるこの関数もツール化する？

            
            # "schedule" : "昼食",
            # "schedule" : "定例",
            # "schedule" : "進捗確認",
            # "schedule" : "面談",
            # "UserAction" : "STABLE",
            "UserAction" : self.UserActionState, # 行動検出をAgentに組み込めば統合できる
        })
        RED = '\033[31m'
        YELLOW = '\033[33m'
        END = '\033[0m'
        BOLD = '\033[1m'
        print(BOLD + YELLOW + "\n--------------------------------------------------")
        print("User Needs: \n", response['UserNeeds'])
        print("\n--------------------------------------------------")
        print("schedule: ", response['schedule'])
        print("User Action: ", response['UserAction'])
        print("\n--------------------------------------------------")
        print("Output: ", response['output'])
        print("\n--------------------------------------------------" + END)

        Suggested_Tool = response['output']

        return Suggested_Tool
        # return Suggested_Tool, self.UserTrendAnswer # 傾向も渡す場合

if __name__ == "__main__":

    from Within5min import CheckScheduleTime
    import datetime
    # dt_now = datetime.datetime(2024, 5, 24, 8, 30)
    # dt_now = datetime.datetime(2024, 5, 24, 10, 55)
    dt_now = datetime.datetime(2024, 7, 17, 12, 5)
    check_schedule = CheckScheduleTime(dt_now)

    # recommend_tool = RecommendTool()
    # recommend_tool.getUserTrends()
    # suggested_tool = recommend_tool.getToolAnswer(check_schedule)

    UserActionState = "WALKING"
    recommend_tool = RecommendTool_Context_withTrends_Ver(check_schedule, UserActionState)
    recommend_tool.getUserTrends()
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