
# JudgeModel.pyを予定表×行動のリコメンドアプリに変更したバージョン

import os
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
# from pathlib import Path
# sys.path.append(str(Path('__file__').resolve().parent.parent)) # LangChain_ChatGPTまでのパス
# print(sys.path)
# sys.path.append(os.path.join(os.path.dirname(__file__), '.')) # '.\\')) # './'))
# このファイルを直接実行する場合のみ必要
from dotenv import load_dotenv
# load_dotenv()
load_dotenv('WebAPI\\Secret\\.env') # たぶんload_dotenv()のみでいい
# load_dotenv('LangChain_ChatGPT\\WebAPI\\Secret\\.env') # たぶんload_dotenv()のみでいい
# このファイルを直接実行する場合のみ必要

# from pathlib import Path
# sys.path.append(str(Path('__file__').resolve().parent.parent)) # LangChain_ChatGPTまでのパス
# print(sys.path)

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
# from langchain.llms import OpenAI
from langchain_community.llms import OpenAI
# from langchain import LLMMathChain, SerpAPIWrapper
from langchain_openai import ChatOpenAI # 新しいやり方
# Memory
from langchain.memory import ConversationBufferMemory
from langchain.agents import load_tools
# Tool
from langchain_google_community import GoogleSearchAPIWrapper # 新しいやり方
# agent が使用するGoogleSearchAPIWrapperのツールを作成
search = GoogleSearchAPIWrapper()
from langchain.chains.llm_math.base import LLMMathChain

# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from WebAPI.Weather.weather_api import OpenWeatherMapQueryRun
from WebAPI.Wikipedia.wikipedia_api import WikipediaQueryRun
from langchain.agents import load_tools, AgentExecutor, Tool, create_react_agent # 新しいやり方
# from Calendar.google_calendar_api import GoogleCalendarTool
from WebAPI.RouteSearch.route_api import RouteSearchQueryRun

# 予定表×行動デモ用にアップテンポな曲のみ再生
# from WebAPI.Spotify.spotify_api import MusicPlaybackQueryRun
from WebAPI.Spotify.spotify_api_Recommend_version import MusicPlaybackQueryRun


from WebAPI.RestaurantSearch.hotpepper_api import RestaurantSearchQueryRun
from WebAPI.Localization.place_api import LocalizationQueryRun
from WebAPI.Schedule.OutlookSchedule_api import ScheduleQueryRun # 2024/05/28 追加
from WebAPI.Calendar.GCalTool import GoogleCalendarTool
from langchain.chains.api.base import APIChain
from langchain.chains.api import news_docs, open_meteo_docs, podcast_docs, tmdb_docs
from UIModel_copy import UserInterfaceModel # ユーザーとのやり取りをするモデル
# from langchain_community.tools.human.tool import HumanInputRun
import datetime


"""
モデルもどこか一か所にまとめる
"""
llm_4o=ChatOpenAI(
    model="ft:gpt-3.5-turbo-1106:personal:demoapp-3p5t-model:9qYtaL0i",
    # model="ft:gpt-3.5-turbo-1106:personal:demoapp-model-2:9qaUOYsT",
    # model="gpt-4o",
    # model="gpt-4o-mini",
    temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
)
# llm_3p5t=ChatOpenAI(
#     # model="gpt-4o",
#     model="gpt-3.5-turbo",
#     temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
# )


userinterface = UserInterfaceModel()


# """
# デモするユースケースに応じて手動で時刻を設定する
# """
# # runningは基本的に運動中と認識されやすい
# dt_now = datetime.datetime(2024, 6, 17, 7, 10)    # 天気情報（今日より前の日付だとエラーになるかも）
# # dt_now = datetime.datetime(2024, 6, 17, 8, 30)    # 出勤(stable:楽曲再生[house-music], walking:経路検索)
# # # dt_now = datetime.datetime(2024, 6, 17, 10, 55) # 定例(stable, walk:会議情報)
# # # dt_now = datetime.datetime(2024, 6, 17, 12, 5)  # 昼食(walk:restaurant, stable:music[relax-music])
# # dt_now = datetime.datetime(2024, 6, 17, 19, 5)  # ジム(run:up tempo, walk:slow tempo, stable:stop)   # 行動検出と連動モード

from WebAPI.DateTime.WhatTimeIsItNow import SetTime
set_time = SetTime()
dt_now = set_time.run()

# dt_now_str = dt_now.strftime("%Y-%m-%d %H:%M:%S") # 2024/07/01


class Langchain4Judge():

    def run(self, credentials_file):
        """
        天気用のツール（二つ目なので現在使っていない）
        """
        # chain_open_meteo = APIChain.from_llm_and_api_docs(
        #     llm_3p5t,
        #     open_meteo_docs.OPEN_METEO_DOCS,
        #     limit_to_domains=["https://api.open-meteo.com/"],
        # )
        """
        NEWS用のツール 2024/05/05
        """
        news_api_key = os.environ["NEWS_API_KEY"] # kwargs["news_api_key"]
        chain_news = APIChain.from_llm_and_api_docs(
            # llm_3p5t,
            llm_4o,
            news_docs.NEWS_DOCS,
            headers={"X-Api-Key": news_api_key},
            limit_to_domains=["https://newsapi.org/"],
        )
        """
        ラジオ用
        """
        # listen_api_key = "" # kwargs["listen_api_key"]
        # chain = APIChain.from_llm_and_api_docs(
        #     llm,
        #     podcast_docs.PODCAST_DOCS,
        #     headers={"X-ListenAPI-Key": listen_api_key},
        #     limit_to_domains=["https://listen-api.listennotes.com/"],
        # )

        # ツールを作成
        # # カレントディレクトリの取得(作業中のフォルダ)
        # current_dir = os.getcwd()
        # print(current_dir)
        # credentials_file = f'{current_dir}/secret/credentials.json' # "credentials.json"
        calendar_tool = GoogleCalendarTool(credentials_file, llm=ChatOpenAI(temperature=0), memory=None)

        # ツールを定義
        tools = [
            Tool(
                name = "Search",
                func=search.run,
                # description="useful for when you need to answer questions about current events"
                description = "Useful when you need to answer questions about current events. It can also be used to solve a problem when you can't find the answer by searching using other tools."
            ),
            Tool(
                name="Calculator",
                description="Useful for when you need to answer questions about math.",
                # func=LLMMathChain.from_llm(llm=llm).run,
                # coroutine=LLMMathChain.from_llm(llm=llm).arun,
                func=LLMMathChain.from_llm(llm=llm_4o).run,
                coroutine=LLMMathChain.from_llm(llm=llm_4o).arun,
                # func=LLMMathChain.from_llm(llm=llm_3p5t).run,
                # coroutine=LLMMathChain.from_llm(llm=llm_3p5t).arun,
            ),
            
            # 天気（本厚木だとエラーになるので、ここはコメントアウトして、Searchを使うようにする）
            OpenWeatherMapQueryRun(),
            # # こっちの天気でもできる
            # Tool(
            #     name="Open-Meteo-API",
            #     description="Useful for when you want to get weather information from the OpenMeteo API. The input should be a question in natural language that this API can answer.",
            #     func=chain_open_meteo.run,
            # ),

            
            WikipediaQueryRun(),
            
            
            # Outlookを優先させたいので、一旦Google Calendarはコメントアウト
            # Tool(
            #     name = "Calendar",
            #     func = calendar_tool.run,
            #     description="Useful for keeping track of appointments."
            # ),
            
            Tool(
                name="News-API",
                description="Use this when you want to get information about the top headlines of current news stories. The input should be a question in natural language that this API can answer.",
                func=chain_news.run,
            ),
            # Tool(
            #     name="Podcast-API",
            #     description="Use the Listen Notes Podcast API to search all podcasts or episodes. The input should be a question in natural language that this API can answer.",
            #     func=chain.run,
            # )


            # なぜか省略事実引数ならいける（通常の引数だとエラー）
            RouteSearchQueryRun(),
            # RouteSearchQueryRun(dt_now_arg = dt_now), # RouteSearchQueryRun(),
            
            MusicPlaybackQueryRun(),

            # LocalizationQueryRun(), # 2024/05/30 一旦コメントアウト

            RestaurantSearchQueryRun(),

            # HumanInputRun(), # ユーザーに入力を求める

            # なぜか省略事実引数ならいける（通常の引数だとエラー）
            ScheduleQueryRun() # dt_now_arg = dt_now) # ScheduleQueryRun() # 2024/05/28 追加

        ]
        # agent が使用する memory の作成
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        agent_kwargs = {
            "suffix": 
            """
            You are an AI who responds to user Input.
            Please provide an answer to the human's question.
            Additonaly, you are having a conversation with a human based on past interactions.
            """
            # From now on, you must communicate in Japanese.
            # """,
        }
        # 解答形式は以下のようにしなければならない。 # 2024/05/28 変更点

        # 最終回答：ここに文字列のみで最終回答をしてください。


        # agentと書いているが、実際はagent_executor
        agent = initialize_agent( # 非推奨
        # agent = create_react_agent(
            tools,

            # llm_3p5t,
            llm_4o,

            # agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, # 複数引数の場合は新しいagentにする必要がある
            # handle_parsing_errors=True, # パースエラーを例外処理で回避しているだけかも（上のモデルとセット）
            # agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, # パースエラー回避
            memory=memory,
            agent_kwargs=agent_kwargs,
            verbose=True,
            handle_parsing_errors=True, # パースエラーを例外処理で回避

            max_iterations=10 # 5 # これがあるとagentのイタレーションに制限をかけられる
        )

        return agent

if __name__ == "__main__":
    """
    Output
    """

    """
    Langchainの定義
    """
    model = Langchain4Judge()
    # ここのパスが重要 (DM4L.pyはDemoApp/にあるので、その直下のSearch_and_LLM/から指定)
    credentials_file = "WebAPI\\Secret\\credentials.json"
    agent = model.run(credentials_file)

    """
    ここにGoogleColab\main_PredUserNeeds_by_VectorStore_to_Chain_Direct_Timing.py
    """
    # recommend_tool = RecommendTool(UserActionState)
    # generate_prompt = GeneratePromptbyTool(suggested_tool)
    
    

    for i in range(1):
        # UserNeeds = '経路検索して。'
        # UserNeeds = '楽曲再生して。'
        # text = 'あなたが保持するToolの一覧を教えて。'#  + f'そのうち、{UserNeeds}に最も適切なToolを選定して。(Tool:)'
        # text = f'あなたが保持するToolの中から、「{UserNeeds}」に最も適切なToolを選定し、Tool名のみ答えて。' # (The tool used is *)' # text = f'あなたが保持するToolの中から、{UserNeeds}に最も適切なToolを選定し、Tool名を答えて。(The tool used is *)'
        
        time = '17:30'
        time = '19:30'
        time = '8:00'
        # time = '9:10' # 30'
        time = '10:50'
        time = '12:05'
        # time = '17:30'
        # time = '20:00'
        status = 'WALKING'
        # status = 'STABLE'
        # status = 'RUNNNING'

        # text = f"inputは「time:{time}, status:{status}」です。ユーザーのニーズに最適な機能「activity」を予測し、time, status, activityからどのようなニーズが考えられますか？箇条書きで教えて。" # 最後にユーザーが気持ちよくなるようなねぎらいの言葉も加えて出力して。"
        # print("input: ", text)
        # response = agent.invoke(text)
        # res = response['output']
        # print(res)

        # print("***********************************************************************************************************************")
        
        # text = f"inputは「time:{time}, status:{status}」です。ユーザーのニーズに最適な機能「activity」名のみを教えて。" # を提案して。"
        # # text = "ユーザーの現在状態は「time:10:30, status:WALKING」です。ユーザーのニーズに最適な「activity」を提案して。"

        # print("input: ", text)
        # response = agent.invoke(text)
        # res = response['output']
        # print(res)
        # # text = f'{res}が楽曲再生ならTrueのみを、そうでないならFalseのみを返して。'
        # # judge_flag= agent.invoke(text)
        # # # print('judge:', judge_flag)
        # # if judge_flag['output'] in 'True':
        # #     print("Spotify再生")

        print("***********************************************************************************************************************")

        # promptの工夫点
        # あなたが保持するToolの中から提案して。何もしないことが適切なら何もしないと出力して。

        text = f"inputは「time:{time}, status:{status}」です。ユーザーのニーズに最適な機能「activity」名のみを、あなたが保持するToolの中から提案して。" # Do Nothingが適切な場合のみ何もしないと出力して。" # 何もしないことが適切なら何もしないで。" # Tool名で出力して。"
        text = f"inputは「time:{time}, status:{status}」です。inputに対して適切な「activity」に最も合うものを考え、さらにあなたが保持するToolの中で最も近いものを選定しTool名のみ答えて。" # 提案して。Do Nothingが適切な場合のみ何もしないと出力して。" # 何もしないことが適切なら何もしないで。" # Tool名で出力して。"

        print("input: ", text)
        response = agent.invoke(text)
        res = response['output']
        print(res)