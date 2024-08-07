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

class RecommendSpotifyPlaylist():
    def __init__(self, UserActionState):
        import sys
        # JudgeModel.pyから呼び出すときに必要
        sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
        # loader = TextLoader('./SuggestToolOutlook/userdata_schedule.txt', encoding='utf8')
        # # loader = TextLoader('./userdata_schedule.txt', encoding='utf8') # 単体テスト用 (SuggestToolOutlook\GeneratePrompt_具体的な提案の検討.py)
        
        
        loader_playlist = TextLoader('./SuggestToolOutlook/userdata_spotify_playlist.txt', encoding='utf8')
        # loader = TextLoader('./userdata_spotify_playlist.txt', encoding='utf8') # 単体テスト用

        # 100文字のチャンクで区切る
        text_splitter = CharacterTextSplitter(        
            separator = "\n\n",
            chunk_size = 100,
            chunk_overlap = 0,
            length_function = len,
        )
        self.index_playlist = VectorstoreIndexCreator(
            vectorstore_cls=Chroma, # Default
            embedding=OpenAIEmbeddings(), # Default # Context_withTrends.pyのやり方にした方がいいかも
            text_splitter=text_splitter, # text_splitterのインスタンスを使っている
        ).from_loaders([loader_playlist])

        self.UserActionState = UserActionState # "WALKING"

    def getUserTrends(self):
        query = """
                あなたは楽曲再生プレイリストのニーズを予測する専門家です。以下に答えて。
                このtxtファイルの文書はユーザーの「どんな予定の時に、どの行動状態で、どのプレイリストを使用したか」の履歴です。
                このユーザーの傾向を「userdata_spotify_playlist.txt」の情報だけで分析・予測し箇条書きでまとめて。
                """
                # userdata_spotify_playlist.txtだけといってもベクトル化されているせいか前の情報も反映されてしまう
                
        self.UserTrendAnswer = self.index_playlist.query(query, llm=llm_4o)
        # self.UserTrendAnswer = self.index.query(query, llm=llm_3p5t)
        return self.UserTrendAnswer # classに結果は保持されるからなくてもいいかも


    def getToolAnswer(self, check_schedule): # , UserActionState):
        # getToolAnswer()に移動する？
        prompt_2 = PromptTemplate(
            input_variables=["UserNeeds", "schedule", "UserAction"],
            # template = """
            #         あなたはユーザーに合う機能を提案する専門家です。
            #         ユーザーの傾向は「{UserNeeds}」です。

            #         現在の予定が{schedule}、ユーザーの行動状態が{UserAction}の場合、どの機能を提案するかこのユーザーの傾向を加味して最終的な提案(Final Answer:)のみを教えて。
            #         あなたが提案できる機能は、
            #         "会議情報", "楽曲再生", "経路検索", "リアルタイム情報検索", "レストラン検索", "ニュース情報", "天気情報"
            #         です。
            #         ###
            #         Final Answer:
            #         """
            template = """
                    あなたはユーザーに合う機能を提案する専門家です。
                    ユーザーの傾向は「{UserNeeds}」です。

                    現在の予定が{schedule}、ユーザーの行動状態が{UserAction}の場合、どの機能を提案するかこのユーザーの傾向を加味して最終的な提案(Final Answer:)のみを教えて。
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
        print("\n--------------------------------------------------")
        print("User Needs (Playlist): \n", response['UserNeeds'])
        print("\n--------------------------------------------------")
        print("schedule: ", response['schedule'])
        print("User Action: ", response['UserAction'])
        print("\n--------------------------------------------------")
        print("Output: ", response['output'])
        print("\n--------------------------------------------------")
        
        Suggested_Tool = response['output']

        return Suggested_Tool
        # return Suggested_Tool, self.UserTrendAnswer # 傾向も渡す場合

if __name__ == "__main__":

    from Within5min import CheckScheduleTime
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
    
    check_schedule = CheckScheduleTime(dt_now)
    recommend_tool = RecommendSpotifyPlaylist(UserActionState)
    recommend_tool.getUserTrends()
    suggested_tool = recommend_tool.getToolAnswer(check_schedule)

    print("\n--------------------------------------------------")
    print(suggested_tool)
    print("--------------------------------------------------")

    if "PlaybackLinkedToActions" in suggested_tool:
        print("PlaybackLinkedToActionsモードに移行します。")
    if "HouseMusic" in suggested_tool:
        print("アップテンポな楽曲再生モードに移行します。")
    if "RelaxMusic" in suggested_tool:
        print("リラックスな楽曲再生モードに移行します。")