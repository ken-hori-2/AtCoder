
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

from WebAPI.DateTime.WhatTimeIsItNow import SetTime
# set_time = SetTime()
# dt_now = set_time.run()


class Langchain4Judge():

    def __init__(self, dt_now):
        """
        モデルもどこか一か所にまとめる
        """
        self.llm_4o=ChatOpenAI(
            model="gpt-4o",
            # model="gpt-3.5-turbo",
            temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
        )
        self.llm_3p5t=ChatOpenAI(
            # model="gpt-4o",
            model="gpt-3.5-turbo",
            temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
        )

        self.userinterface = UserInterfaceModel()
        # self.set_time = SetTime()
        self.dt_now = dt_now

    def run(self, credentials_file):

        
        # dt_now = self.set_time.run()
        print("現在時刻：", self.dt_now)

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
            self.llm_3p5t,
            # self.llm_4o,
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
                # func=LLMMathChain.from_llm(llm=llm_4o).run,
                # coroutine=LLMMathChain.from_llm(llm=llm_4o).arun,
                func=LLMMathChain.from_llm(llm=self.llm_3p5t).run,
                coroutine=LLMMathChain.from_llm(llm=self.llm_3p5t).arun,
            ),


            # # 天気 （本厚木だとエラーになるので、ここはコメントアウトして、Searchを使うようにする）
            # OpenWeatherMapQueryRun(),
            # # こっちの天気でもできる
            # # Tool(
            # #     name="Open-Meteo-API",
            # #     description="Useful for when you want to get weather information from the OpenMeteo API. The input should be a question in natural language that this API can answer.",
            # #     func=chain_open_meteo.run,
            # # ),


            # WikipediaQueryRun(),
            
            
            # # Outlookを優先させたいので、一旦Google Calendarはコメントアウト
            # # Tool(
            # #     name = "Calendar",
            # #     func = calendar_tool.run,
            # #     description="Useful for keeping track of appointments."
            # # ),
            
            # Tool(
            #     name="News-API",
            #     description="Use this when you want to get information about the top headlines of current news stories. The input should be a question in natural language that this API can answer.",
            #     func=chain_news.run,
            # ),
            # # Tool(
            # #     name="Podcast-API",
            # #     description="Use the Listen Notes Podcast API to search all podcasts or episodes. The input should be a question in natural language that this API can answer.",
            # #     func=chain.run,
            # # )


            # なぜか省略事実引数ならいける（通常の引数だとエラー）
            RouteSearchQueryRun(),
            # RouteSearchQueryRun(dt_now_arg = dt_now), # RouteSearchQueryRun(),
            
            MusicPlaybackQueryRun(),

            # LocalizationQueryRun(), # 2024/05/30 一旦コメントアウト

            RestaurantSearchQueryRun(),

            # HumanInputRun(), # ユーザーに入力を求める

            # なぜか省略事実引数ならいける（通常の引数だとエラー）
            ScheduleQueryRun() # dt_now_arg = self.dt_now) # ScheduleQueryRun() # 2024/05/28 追加

        ]
        # agent が使用する memory の作成
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # テンプレート
        # agent_kwargs = {
        #     "suffix": 
        #     """
        #     あなたはユーザーの入力に応答するスペシャリストです。
        #     人間の要求に答えてください。
        #     その際に適切なツールを使いこなして回答してください。
        #     さらに、あなたは過去のやりとりに基づいて人間と会話をしています。

        #     開始!ここからの会話は全て日本語で行われる。

        #     ### 解答例
        #     Human: やあ！
        #     GPT(AI) Answer: こんにちは！
            
        #     ### 以前のチャット履歴
        #     {chat_history}

        #     ###
        #     Human: {input}
        #     {agent_scratchpad}
        #     """,

            
        # }
        agent_kwargs = {
            "suffix": 
            """
            あなたはユーザーの入力に応答するAIです。
            人間の要求に答えてください。
            その際に適切なツールを使いこなして回答してください。
            さらに、あなたは過去のやりとりに基づいて人間と会話をしています。



            開始!ここからの会話は全て日本語で行われる。

            ### 解答例
            [Human]: やあ！
            [AI]: こんにちは！
            Final Answer:
            
            ### 以前のチャット履歴
            {chat_history}

            ###
            Human: {input}
            {agent_scratchpad}
            """,
        }
        # 解答形式は以下のようにしなければならない。 # 2024/05/28 変更点

        # 最終回答：ここに文字列のみで最終回答をしてください。


        # agentと書いているが、実際はagent_executor
        agent = initialize_agent( # 非推奨
        # agent = create_react_agent(
            tools,

            # self.llm_3p5t,
            self.llm_4o,

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
    
    def output(self): # , response):
        from langchain import PromptTemplate, LLMChain
        # Memory: メモリ上に会話を記録する設定
        memory_key = "chat_history"
        memory = ConversationBufferMemory(memory_key=memory_key, ai_prefix="")

        # Prompts: プロンプトを作成。会話履歴もinput_variablesとして指定する
        template = """
        You are an AI who responds to user Input.
        Please provide an answer to the human's question.
        Additonaly, you are having a conversation with a human based on past interactions.

        From now on, you must communicate in Japanese.
        If there are multiple search results, output up to two results.

        ### 解答例
        Human: やあ！
        GPT(AI) Answer: こんにちは！

        ### 以前のチャット履歴
        {chat_history}

        ### 
        Human:{input}
        """
        # templateに追加してもいいかも
        # "あなたはリスト形式などではなく、また、カギかっこなどのなく、一文当たり短い箇条書きにして回答しなければならない"
        # "You must respond in a short bulleted list per sentence, not in list form, no brackets, etc."


        prompt = PromptTemplate(template=template, input_variables=["chat_history", "input"])
        # Chains: プロンプト&モデル&メモリをチェーンに登録
        llm_chain = LLMChain(

            # llm=llm,
            # llm = self.llm_3p5t,
            llm = self.llm_4o,

            prompt=prompt,
            memory=memory,
            verbose=True,
        )
        # 実行①
        # user_input = "次の文をリスト形式ではなく、[]や\{\}のない多仇の文字列にしてください。\n" + response # "What is the Japanese word for mountain？"
        
        
        # 現在は使っていない
        # user_input = f"あなたは'PLAYBACK'または'OTHER'のどちらかで回答しなければならない。{response['output']}の文からは楽曲再生、停止などの操作を実行してると思いますか？\n楽曲操作を実行している場合は、'PLAYBACK'、楽曲操作以外を実行している場合は'OTHER'と回答してください。"
        """
        LLM 2個目
        """


        """
        2024/05/09
        """
        # 一旦コメントアウト（PLAYBACK or OTHERは ルールベースで決める）
        # final_response = llm_chain.predict(input=user_input)
        # print(final_response)
        
        # return final_response, llm_chain

        return llm_chain # PLAYBACK or OTHERは ルールベースで決める