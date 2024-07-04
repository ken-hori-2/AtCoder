# ### 必要なライブラリのインポート

# # ```python
# from langchain import OpenAI, LLMChain
# from langchain.prompts import PromptTemplate
# from langchain.chains import SimpleChain
# import random
# # ```

# ### センサーデータの擬似入力

# # ここでは、センサーデータをシミュレートします。実際のアプリケーションでは、センサーデータはリアルタイムに取得される必要がありますが、この例では簡略化のために手動でデータを設定します。

# # ```python
# # センサーデータの擬似入力
# sensor_data = {
#     "location": "office",
#     "motion": "sitting",
#     "sound": "quiet",
#     "light": "bright",
#     "heart_rate": "stable"
# }

# # データを変動させるための擬似ランダム化
# def randomize_sensor_data(data):
#     motions = ["sitting", "walking", "running"]
#     sounds = ["quiet", "noisy", "talking"]
#     lights = ["bright", "dim", "dark"]
#     heart_rates = ["stable", "elevated", "low"]
    
#     data["motion"] = random.choice(motions)
#     data["sound"] = random.choice(sounds)
#     data["light"] = random.choice(lights)
#     data["heart_rate"] = random.choice(heart_rates)
    
#     return data

# sensor_data = randomize_sensor_data(sensor_data)
# # ```

# ### プロンプトテンプレートの作成

# # センサーデータに基づいてユーザーの意図を予測するためのプロンプトテンプレートを作成します。

# # ```python
# prompt_template = """
# Based on the following sensor data, generate a prompt that describes the user's current intention or need:

# - Location: {location}
# - Motion: {motion}
# - Sound: {sound}
# - Light: {light}
# - Heart Rate: {heart_rate}

# Provide a detailed description of the user's likely intention or need.
# """
# # ```

# ### LangChainを使用してプロンプトを生成

# # 次に、LangChainのチェーンを設定し、LLMを使用してプロンプトを生成します。

# # ```python
# # OpenAIのAPIキーを設定
# openai_api_key = "your-openai-api-key"

# # LLMのインスタンスを作成
# llm = OpenAI(api_key=openai_api_key, model="gpt-4")

# # プロンプトテンプレートを設定
# prompt = PromptTemplate(template=prompt_template, input_variables=["location", "motion", "sound", "light", "heart_rate"])

# # チェーンを設定
# chain = LLMChain(llm=llm, prompt=prompt)

# # プロンプトを生成
# result = chain.run(sensor_data)

# print("Generated Prompt: ", result)
# # ```

### 完全なコードの例

# 以下に、全体のコードをまとめます。これにより、センサーデータを元にユーザーの意図を生成するプロンプトをLangChainで生成できます。

# ```python
# from langchain import OpenAI, LLMChain
from langchain_openai import ChatOpenAI # 新しいやり方
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# from langchain.chains import SimpleChain
import random

from dotenv import load_dotenv
load_dotenv()

llm_4o=ChatOpenAI(
    model="gpt-4o",
    # model="gpt-3.5-turbo",
    temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
)
llm_3p5t=ChatOpenAI(
    # model="gpt-4o",
    model="gpt-3.5-turbo",
    temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
)

# OpenAIのAPIキーを設定
# openai_api_key = "your-openai-api-key"



class GenerateContext():

    def __init__(self):
        
        # センサーデータの擬似入力
        self.sensor_data = {
            "location": "office",
            "motion": "sitting",
            "sound": "quiet",
            # "light": "bright",
            "heart_rate": "stable",

            "time": "7:00"
        }

    # データを変動させるための擬似ランダム化
    def randomize_sensor_data(self, data):
        locations = ["office", "train", "home", "downtown"]
        data["location"] = random.choice(locations)

        # motions = ["sitting", "walking", "running"]
        motions = ["sitting", "standup", "gesture", "stable", "walking", "running"]
        sounds = ["quiet", "noisy", "talking"]
        # lights = ["bright", "dim", "dark"]
        heart_rates = ["stable", "elevated", "low"]
        
        data["motion"] = random.choice(motions)
        data["sound"] = random.choice(sounds)
        # data["light"] = random.choice(lights)
        data["heart_rate"] = random.choice(heart_rates)


        time = ["7:00", "8:00", "10:00", "12:00", "15:00", "17:30", "19:00"]
        # data["time"] = random.choice(time)
        # 今は指定
        data["time"] = "10:58" # "8:00"
        
        return data
    
    def run(self):

        sensor_data = self.randomize_sensor_data(self.sensor_data)

        # プロンプトテンプレートを作成
        # prompt_template = """
        # Based on the following sensor data, generate a prompt that describes the user's current intention or need:

        # - Location: {location}
        # - Motion: {motion}
        # - Sound: {sound}
        # - Light: {light}
        # - Heart Rate: {heart_rate}

        # Provide a detailed description of the user's likely intention or need.
        # """
        prompt_template = """
        Based on the following sensor data, prompts representing the user's current intentions and needs are generated in a straightforward manner:

        - Location: {location}
        - Motion: {motion}
        - Sound: {sound}
        - Heart Rate: {heart_rate}
        - Time: {time}

        Provide a detailed description of the user's likely intention or need.
        Final output must be in Japanese.
        """

        # LLMのインスタンスを作成
        # llm = OpenAI(api_key=openai_api_key, model="gpt-4")

        # プロンプトテンプレートを設定
        # # prompt = PromptTemplate(template=prompt_template, input_variables=["location", "motion", "sound", "light", "heart_rate"])
        # prompt = PromptTemplate(template=prompt_template, input_variables=["location", "motion", "sound", "heart_rate"])
        """ 時刻を追加 """
        prompt = PromptTemplate(template=prompt_template, input_variables=["location", "motion", "sound", "heart_rate", "time"])

        # チェーンを設定
        chain = LLMChain(llm=llm_4o, prompt=prompt)
        # chain = prompt | llm_4o

        # プロンプトを生成
        print("Sensor Data:", sensor_data)
        result = chain.invoke(sensor_data)

        print("Generated Prompt: ", result['text']) # ["content"])

        return result['text']


if __name__ == "__main__":

    generate_context = GenerateContext()
    context = generate_context.run()

    # このファイルを直接実行する場合のみ必要
    from dotenv import load_dotenv
    # load_dotenv()
    load_dotenv('WebAPI\\Secret\\.env') # たぶんload_dotenv()のみでいい
    from UIModel_copy import UserInterfaceModel # ユーザーとのやり取りをするモデル
    import datetime
    from SuggestToolTimeAction.GeneratePrompt import GeneratePromptbyTool
    from SuggestToolTimeAction.RecommendToolbyTimeAction import RecommendTool
    from SuggestToolTimeAction.TriggerByEdgeAI import Trigger
    from Context_JudgeModel import Langchain4Judge
    from WebAPI.DateTime.WhatTimeIsItNow import SetTime

    userinterface = UserInterfaceModel()
    set_time = SetTime()
    dt_now = set_time.run()

    """
    Langchainの定義
    """
    model = Langchain4Judge(dt_now)
    # ここのパスが重要 (DM4L.pyはDemoApp/にあるので、その直下のSearch_and_LLM/から指定)
    credentials_file = "WebAPI\\Secret\\credentials.json"
    agent = model.run(credentials_file)
    
    """
    EdgeAIによるトリガーの定義
    """
    trigger = Trigger()

    # check_schedule = CheckScheduleTime(dt_now)
    """
    ここにGoogleColab\main_PredUserNeeds_by_VectorStore_to_Chain_Direct_Timing.py
    """
    # recommend_tool = RecommendTool(UserActionState)
    # generate_prompt = GeneratePromptbyTool(suggested_tool)
    
    

    for i in range(1): # 2):
        # is5min = False
        # is5min = check_schedule.isScheduleStartWithin5min() # LLMデモアプリ起動
        """
        ここにGoogleColab\main_PredUserNeeds_by_VectorStore_to_Chain_Direct_Timing.pyで取得した時刻で実行するようにする
        """
        isTrigger = True

        prompt_answer = ""

        # if isTrigger:
        #     print("デモアプリ起動。")

        #     # 2024/05/29 次回TODO
        #     # ユースケース
        #     # ある時刻でアプリ起動 → センシング開始 → 行動に応じて機能が変わる（ユーザーの傾向をもとにしている）
        #     # 2周目、検出された行動が変われば機能も変わることを示す

        #     print("***** センシング中 *****")
        #     # UserActionState = trigger.run()
        #     UserActionState = "WALKING" # テスト用
        #     # UserActionState = "STABLE" # テスト用
        #     print("DNN検出結果：", UserActionState)
        #     print("***** センシング終了 *****")
            
        #     dt_now_for_time_action = datetime.timedelta(hours=dt_now.hour, minutes=dt_now.minute) # 経路案内 # datetime.timedelta(hours=17, minutes=58) # 経路案内
        #     print("\n\n【テスト】現在時刻：", dt_now_for_time_action)
        #     # UserActionState = "WALKING"
        #     """ # RAG version """
        #     recommend_tool_time_action = RecommendTool(dt_now_for_time_action, UserActionState)
        #     recommend_tool_time_action.getUserTrends()
        #     suggested_tool = recommend_tool_time_action.getToolAnswer()
        #     """ # Prediction Tool version """
        #     # from PredictionTool4TimeAction import PredctionModel
        #     # predictionmodel = PredctionModel()
        #     # suggested_tool = predictionmodel.run(str(dt_now_for_time_action), UserActionState)

        #     isMusicPlayback = False

        #     if suggested_tool:
        #         print("\n--------------------------------------------------")
        #         print(suggested_tool)
        #         print("--------------------------------------------------")

        #         if "楽曲再生" in suggested_tool:
        #             print("**********\n楽曲再生なので、ガイダンス処理は実行しません。\n直接楽曲再生に移行します。\n**********")
        #             isMusicPlayback = True

        #             """
        #             2024/6/17
        #             ここでユーザーの傾向からプレイリストを決定する処理を追加
        #             """
        #             from SuggestToolTimeAction.RecommendPlaylistTimeAction import RecommendSpotifyPlaylist

        #             dt_now_for_time_action = datetime.timedelta(hours=dt_now.hour, minutes=dt_now.minute) # 経路案内 # datetime.timedelta(hours=17, minutes=58) # 経路案内
        #             print("\n\n【テスト】現在時刻：", dt_now_for_time_action)
        #             recommend_playlist_time_action = RecommendSpotifyPlaylist(dt_now_for_time_action, UserActionState)
        #             recommend_playlist_time_action.getUserTrends()
        #             suggested_playlist = recommend_playlist_time_action.getToolAnswer()
        #             # a 予定を使うバージョン
        #             # ・ユーザーの傾向から再生モードを決定（db[予定/行動/機能]から傾向取得、現在時刻から予定取得、行動状態→入力）

        #             print("\n--------------------------------------------------")
        #             print(suggested_playlist)
        #             print("--------------------------------------------------")
        #             if "PlaybackLinkedToActions" in suggested_playlist:
        #                 print("「Playback Linked to Actions」モードに移行します。")
        #                 # print("行動検出連動の楽曲再生モードに移行します。")
        #                 # 今実装済みの機能を使えばいい
        #                 suggested_tool = suggested_playlist
        #             elif "HouseMusic" in suggested_playlist:
        #                 print("「Playback UpTempo-Music」モードに移行します。")
        #                 # print("アップテンポな楽曲再生モードに移行します。")
        #                 suggested_tool = suggested_playlist
        #             elif "RelaxMusic" in suggested_playlist:
        #                 print("Playback Relax-Music」モードに移行します。")
        #                 # print("リラックスな楽曲再生モードに移行します。")
        #                 suggested_tool = suggested_playlist
        #             else:
        #                 print("該当するプレイリストはありません。")
                    
        #         else:
        #             isMusicPlayback = False

        #             generate_prompt = GeneratePromptbyTool(suggested_tool)
        #             isExecuteTool = generate_prompt.getJudgeResult()
        #             if isExecuteTool:
        #                 # A
        #                 prompt_answer = generate_prompt.getGeneratedPrompt() # 機能からプロンプト生成する場合

        #                 # B
        #                 # # prompt_answer = suggested_tool + "を実行して。" # プロンプトに機能をそのまま入力する場合
        #                 # prompt_answer = suggested_tool + "して。" # プロンプトに機能をそのまま入力する場合

        
        #     if not isMusicPlayback:
        #         if prompt_answer:
        #             print("プロンプト：", prompt_answer)
        #             text = "現在時刻は" + str(dt_now) + "です。" + prompt_answer
                    
        #             if text:
        #                 print(" >> Waiting for response from Agent...")
        #                 """
        #                 Output
        #                 """
        #                 print("\n\n******************** [User Input] ********************\n", text)
        #                 # try:
        #                 response = agent.invoke(text) # できた(その後エラーはあるが)
        #                 # text_to_speach(final_response)

        #                 print("\n\n******************** [AI Answer] ********************\n")
        #                 # model.text_to_speach(response['output'])
        #                 userinterface.text_to_speach(response['output'])
        #                 # except:
        #                 #     print("\n##################################################\nERROR! ERROR! ERROR!\n##################################################")
        #                 #     print("もう一度入力してください。")
        #     else:
        #         # 楽曲再生
        #         text = f"{suggested_tool}を実行して。" # プロンプトを入力（楽曲再生）
        #         # text = f"{suggested_tool}を一回だけ実行して。一回実行したら終了して。"

        #         # """
        #         # 2024/6/17
        #         # ここでユーザーの傾向からプレイリストを決定する処理を追加
        #         # """
                
        #         if text:
        #             print(" >> Waiting for response from Agent... (Execute music playback)")
        #             # print(" >> Execute music playback...")
        #             """
        #             Output
        #             """
        #             response = agent.invoke(text) # できた(その後エラーはあるが)
        # else:
        #     print("処理を終了します。")

        # context += "また、現在のユーザーの状況に最も合いそうな機能を実行して。最終的にユーザーの気持ちに沿ったガイダンスをして。\
        #             ユーザーに関する情報が足りない場合は予定表(Schedule)を参照し出社場所を取得して。検索結果が複数ある場合は2件までにして。"
        # context += "現在時刻は" + str(dt_now) + "です。" + "また、現在のユーザーの状況に最も合いそうな機能を実行して。ユーザーに関する情報が足りない場合は予定表(Schedule)を参照し出社場所を取得して。"
        context += "現在時刻は" + str(dt_now) + "です。" + "また、現在のユーザーの状況に最も合いそうな機能を実行して。"

        """ # RAG version """
        # dt_now_for_time_action = datetime.timedelta(hours=dt_now.hour, minutes=dt_now.minute)
        recommend_tool_time_action = RecommendTool(dt_now_for_time_action=None, UserActionState=None)
        UserTrend = recommend_tool_time_action.getUserTrends()
        # suggested_tool = recommend_tool_time_action.getToolAnswer()
        print("User Trend:", UserTrend)
        context += "ユーザーの傾向は「" + UserTrend + "」です。"


        response = agent.invoke(context) # できた(その後エラーはあるが)