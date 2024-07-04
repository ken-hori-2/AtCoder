
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


# UserActionState = "walking"
UserActionState = "stable"
# day_and_time = "Wednesday 8:00"
day_and_time = "Tuesday 8:00"
# day_and_time = "Tuesday 12:00"
# location = "玉川学園前"
# location = "本厚木"
location = "電車"

class GenerateContext():

    def __init__(self):
        """ Recommend Tool """
        recommend_tool_time_action = RecommendTool(day_and_time, UserActionState)

        # UserTrend = recommend_tool_time_action.getUserTrends() # 以前のやり方 # UserTrends_original()
        UserTrend = recommend_tool_time_action.getUserTrends_ver2() # RAG.py

        # """ RAG.py """ # ver2()と同じなので使わない
        # # from RAG import OutputUserTrends
        # # output_usertrends = OutputUserTrends()
        # # UserTrend = output_usertrends.getUserTrends()

        print("User Trend:", UserTrend)
        # user_trends = "ユーザーの傾向は「" + UserTrend + "」です。"
        user_trends = "「" + UserTrend + "」"

        
        
        # 初期化
        # センサーデータの擬似入力
        self.sensor_data = {
            "location": "office",
            "user_action": "sitting",
            "sound": "quiet",
            # "light": "bright",
            "heart_rate": "stable",

            # "time": "7:00",
            "user_trends": user_trends,
            # "day_of_the_week":"monday"
            "day_and_time":"none"
        }

    # データを変動させるための擬似ランダム化
    def randomize_sensor_data(self, data):
        locations = ["office", "train", "home", "downtown"] # 現在地(GPS想定)
        data["location"] = random.choice(locations)

        # motions = ["sitting", "walking", "running"]
        actions = ["sitting", "standup", "gesture", "stable", "walking", "running"]
        sounds = ["quiet", "noisy", "talking"]
        # lights = ["bright", "dim", "dark"]
        heart_rates = ["stable", "elevated", "low"]
        
        data["user_action"] = random.choice(actions) # motions)
        data["sound"] = random.choice(sounds)
        # data["light"] = random.choice(lights)
        data["heart_rate"] = random.choice(heart_rates)


        time = ["7:00", "8:00", "10:00", "12:00", "15:00", "17:30", "19:00"]
        # data["time"] = random.choice(time)
        
        # 今は指定
        # data["time"] = "8:00" # "12:00" # "19:00" # "10:58" # "8:00"
        data["location"] = location # "玉川学園前" # "大崎" # "玉川学園前" # "本厚木" # 現在地を指定(GPS想定)
        # data["day_of_the_week"] = 
        data["day_and_time"] = day_and_time # "Wednesday" # "Tuesday" # "wednesday"
        data["user_action"] = UserActionState # "walking"
        data["heart_rate"] = "elevated"
        
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
        # prompt_template = """
        # Based on the following sensor data, prompts representing the user's current intentions and needs are generated in a straightforward manner:

        # - Location: {location}
        # - Motion: {motion}
        # - Sound: {sound}
        # - Heart Rate: {heart_rate}
        # - Time: {time}
        # - Day of the week: {day_of_the_week}
        # - User Trends: {user_trends}

        # Provide a detailed description of the user's likely intention or need.
        # Final output must be in Japanese.
        # """
        prompt_template = """
        Based on the following sensor data, prompts representing the user's current intentions and needs are generated in a straightforward manner:

        - Location: {location}
        - User Action: {user_action}
        - Sound: {sound}
        - Heart Rate: {heart_rate}
        - Day and Time: {day_and_time}
        - User Trends: {user_trends}

        Provide a detailed description of the user's likely intention or need.
        Final output must be in Japanese.
        """

        # LLMのインスタンスを作成
        # llm = OpenAI(api_key=openai_api_key, model="gpt-4")

        # プロンプトテンプレートを設定
        # # prompt = PromptTemplate(template=prompt_template, input_variables=["location", "motion", "sound", "light", "heart_rate"])
        # prompt = PromptTemplate(template=prompt_template, input_variables=["location", "motion", "sound", "heart_rate"])
        """ 時刻を追加 """
        # prompt = PromptTemplate(template=prompt_template, input_variables=["location", "motion", "sound", "heart_rate", "time",  "user_trends"])
        # prompt = PromptTemplate(template=prompt_template, input_variables=["location", "motion", "sound", "heart_rate", "time",  "day_of_the_week", "user_trends"])
        prompt = PromptTemplate(template=prompt_template, input_variables=["location", "user_action", "sound", "heart_rate", "day_and_time", "user_trends"])

        # チェーンを設定
        chain = LLMChain(llm=llm_4o, prompt=prompt)
        # chain = prompt | llm_4o

        # プロンプトを生成
        print("Sensor Data:", sensor_data)
        result = chain.invoke(sensor_data)

        print("Generated Prompt: ", result['text']) # ["content"])

        return result['text']


if __name__ == "__main__":

    # generate_context = GenerateContext()
    # context = generate_context.run()

    # このファイルを直接実行する場合のみ必要
    from dotenv import load_dotenv
    # load_dotenv()
    load_dotenv('WebAPI\\Secret\\.env') # たぶんload_dotenv()のみでいい
    from UIModel_copy import UserInterfaceModel # ユーザーとのやり取りをするモデル
    import datetime
    from SuggestToolTimeAction.GeneratePrompt import GeneratePromptbyTool
    from SuggestToolTimeAction.RecommendToolbyTimeAction_context_version import RecommendTool
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
        
        generate_context = GenerateContext()
        context = generate_context.run()

        # context += "現在時刻は" + str(dt_now) + "です。" + "また、現在のユーザーの状況に最も合いそうな機能を実行して。ユーザーに関する情報が足りない場合は予定表(Schedule)を参照し出社場所を取得して。"
        
        # context += "現在時刻は" + str(dt_now) + "です。" + "また、現在のユーザーの状況に最も合いそうな機能を実行して。"
        # 1つのみ実行させる場合
        context += "現在時刻は" + str(dt_now) + "です。" + "また、現在のユーザーの状況に最も合いそうな機能を予測して実行して(実行するアプリは1つだけでなければならない)。検索結果の表示は多くても3つまでにしなければならない。"


        response = agent.invoke(context) # できた(その後エラーはあるが)