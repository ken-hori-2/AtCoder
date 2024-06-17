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
    # model="gpt-3.5-turbo",
    temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
) # チャット特化型モデル
llm_3p5t=ChatOpenAI(
    # model="gpt-4o",
    model="gpt-3.5-turbo",
    temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
) # チャット特化型モデル

class RecommendSpotifyPlaylist():
    def __init__(self, dt_now_for_time_action, UserActionState):
        import sys
        # JudgeModel.pyから呼び出すときに必要
        sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
        # loader = TextLoader('./SuggestToolOutlook/userdata_schedule.txt', encoding='utf8')
        # # loader = TextLoader('./userdata_schedule.txt', encoding='utf8') # 単体テスト用 (SuggestToolOutlook\GeneratePrompt_具体的な提案の検討.py)
        loader_playlist = TextLoader('./SuggestToolTimeAction/userdata_spotify_playlist.txt', encoding='utf8')
        # loader_playlist = TextLoader('./userdata_spotify_playlist.txt', encoding='utf8') # 単体テスト用

        # 100文字のチャンクで区切る
        text_splitter = CharacterTextSplitter(        
            separator = "\n\n",
            chunk_size = 100,
            chunk_overlap = 0,
            length_function = len,
        )
        self.index_playlist = VectorstoreIndexCreator(
            vectorstore_cls=Chroma, # Default
            embedding=OpenAIEmbeddings(), # Default
            text_splitter=text_splitter, # text_splitterのインスタンスを使っている
        ).from_loaders([loader_playlist])

        self.dt_now_for_time_action = dt_now_for_time_action
        self.UserActionState = UserActionState # "WALKING"
        
    def getUserTrends(self):
        query = """
                あなたは楽曲再生プレイリストのニーズを予測する専門家です。以下に答えて。
                このtxtファイルの文書はユーザーの「どの時刻に、どの行動状態で、どのプレイリストを使用したか」の履歴です。
                このユーザーの傾向を「userdata_spotify_playlist.txt」の情報だけで分析・予測し箇条書きでまとめて。
                """
                # userdata_spotify_playlist.txtだけといってもベクトル化されているせいか前の情報も反映されてしまう

                # (例：朝の時間帯(xx:xx - xx:xx): ,午前中(xx:xx - xx:xx), 昼の時間帯(xx:xx - xx:xx), 午後(xx:xx - xx:xx), 夕方(xx:xx - xx:xx), 夜の時間帯(xx:xx - xx:xx))
                # """
        
                # (例1：朝の時間帯(xx:xx - xx:xx): ,午前中(xx:xx - xx:xx), 昼の時間帯(xx:xx - xx:xx), 午後(xx:xx - xx:xx), 夕方(xx:xx - xx:xx), 夜の時間帯(xx:xx - xx:xx))
                # (例2：行動状態ごとの使う機能の傾向)
                # """
        self.UserTrendAnswer = self.index_playlist.query(query, llm=llm_4o)
        # self.UserTrendAnswer = self.index.query(query, llm=llm_3p5t)
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

                    現在の時刻が{time}、ユーザーの行動状態が{UserAction}の場合、どの機能を提案するかこのユーザーの傾向を加味して最終的な提案(Final Answer:)のみを教えて。
                    あなたが提案できる機能は、
                    "楽曲再生:HouseMusic", "楽曲再生:RelaxMusic", "楽曲再生:PlaybackLinkedToActions"
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
            input_variables=["UserNeeds", "time", "UserAction"],
            # output_variables=["response"], # あくまで辞書型のなんていう要素に出力が格納されるかの変数
            verbose=True,
        )
        # initから移動⇑

        

        response = self.overall_chain({
            "UserNeeds" : self.UserTrendAnswer,
            "time" : self.dt_now_for_time_action,
            "UserAction" : self.UserActionState,
        })
        print("\n--------------------------------------------------")
        print("User Needs (Playlist): \n", response['UserNeeds'])
        print("\n--------------------------------------------------")
        print("time: ", response['time'])
        print("User Action: ", response['UserAction'])
        print("\n--------------------------------------------------")
        print("Output: ", response['output'])
        print("\n--------------------------------------------------")

        Suggested_Tool = response['output']

        return Suggested_Tool
        # return Suggested_Tool, self.UserTrendAnswer # 傾向も渡す場合

if __name__ == "__main__":

    # from Within5min import CheckScheduleTime
    import datetime
    # dt_now = datetime.datetime(2024, 5, 24, 8, 30)
    # dt_now = datetime.datetime(2024, 5, 24, 10, 55)
    dt_now = datetime.datetime(2024, 6, 12, 19, 5) # ジム(run:up tempo, walk:slow tempo, stable:stop) # 行動検出と連動モード
    dt_now = datetime.datetime(2024, 6, 12, 12, 5) # 昼食（ゆっくり休みたい）                          # リラックスモード
    dt_now = datetime.datetime(2024, 6, 12, 8, 30) # 出勤（気分上げたい）                              # アップテンポモード

    print("***** センシング中 *****")
    # UserActionState = trigger.run()
    # UserActionState = "WALKING" # テスト用
    UserActionState = "STABLE" # テスト用
    print("DNN検出結果：", UserActionState)
    print("***** センシング終了 *****")
    
    # check_schedule = CheckScheduleTime(dt_now)
    # recommend_tool = RecommendSpotifyPlaylist(UserActionState)
    # recommend_tool.getUserTrends()
    # suggested_tool = recommend_tool.getToolAnswer(check_schedule)
    
    dt_now_for_time_action = datetime.timedelta(hours=dt_now.hour, minutes=dt_now.minute) # 経路案内 # datetime.timedelta(hours=17, minutes=58) # 経路案内
    print("\n\n【テスト】現在時刻：", dt_now_for_time_action)
    recommend_tool_time_action = RecommendSpotifyPlaylist(dt_now_for_time_action, UserActionState)
    recommend_tool_time_action.getUserTrends()
    suggested_tool = recommend_tool_time_action.getToolAnswer()

    print("\n--------------------------------------------------")
    print(suggested_tool)
    print("--------------------------------------------------")

    if "PlaybackLinkedToActions" in suggested_tool:
        print("PlaybackLinkedToActionsモードに移行します。")
    if "HouseMusic" in suggested_tool:
        print("アップテンポな楽曲再生モードに移行します。")
    if "RelaxMusic" in suggested_tool:
        print("リラックスな楽曲再生モードに移行します。")