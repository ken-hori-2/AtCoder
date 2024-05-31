
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


"""
モデルもどこか一か所にまとめる
"""
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


userinterface = UserInterfaceModel()

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
            llm_3p5t,
            # llm_4o,
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
            ),
            # 天気
            # OpenWeatherMapQueryRun(),
            # こっちの天気でもできる
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
            RouteSearchQueryRun(),
            MusicPlaybackQueryRun(),

            # LocalizationQueryRun(), # 2024/05/30 一旦コメントアウト
            RestaurantSearchQueryRun(),

            # HumanInputRun(), # ユーザーに入力を求める

            ScheduleQueryRun() # 2024/05/28 追加

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

            max_iterations=5 # 10 # これがあるとagentのイタレーションに制限をかけられる
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
            # llm = llm_3p5t,
            llm = llm_4o,

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

    

    
    # from SuggestToolOutlook.Within5min import CheckScheduleTime
    from SuggestToolTimeAction.GeneratePrompt import GeneratePromptbyTool
    from SuggestToolTimeAction.RecommendToolbyTimeAction import RecommendTool
    from SuggestToolTimeAction.TriggerByEdgeAI import Trigger
    import requests
    import json
    import spotipy
    import spotipy.util as util
    """
    EdgeAIによるトリガーの定義
    """
    trigger = Trigger()

    # これらはデモアプリ起動のために使う
    # dt_now = datetime.datetime(2024, 5, 24, 8, 30)


    dt_now = datetime.datetime(2024, 5, 24, 7, 55) # 天気(walk:route, stable:music)
    dt_now = datetime.datetime(2024, 5, 24, 8, 15) # 出勤(walk:route, stable:music)
    dt_now = datetime.datetime(2024, 5, 24, 10, 55) # 定例
    dt_now = datetime.datetime(2024, 5, 24, 11, 55) # 昼食(walk:restaurant, stable:music)





    
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

        if isTrigger:
            print("デモアプリ起動。")

            
            
            
            
            
            # 2024/05/29 次回TODO
            # ユースケース
            # ある時刻でアプリ起動 → センシング開始 → 行動に応じて機能が変わる（ユーザーの傾向をもとにしている）
            # 2周目、検出された行動が変われば機能も変わることを示す

            print("***** センシング中 *****")
            # UserActionState = trigger.run()
            UserActionState = "WALKING" # テスト用
            # UserActionState = "STABLE" # テスト用
            print("DNN検出結果：", UserActionState)
            print("***** センシング終了 *****")

            
            # 一旦コメントアウト(APIの利用上限回数にならないようにするため)
            """
            # Spotifyの停止用ツールを作る（アップテンポな曲用、リラックスな曲用などに分けてもいいかも）
            # 今だけ（一時停止）
            id = None
            username = os.environ['UserName']
            scope = 'user-read-playback-state,playlist-read-private,user-modify-playback-state,playlist-modify-public'
            client_id = os.environ['SPOTIFY_USER_ID'] # Client_ID'] # ここに自分の client ID'
            client_secret = os.environ['SPOTIFY_TOKEN'] # Client_Secret'] # ここに自分の client seret'
            redirect_uri = 'http://localhost:8888/callback'
            token = util.prompt_for_user_token(username, scope,     client_id, client_secret, redirect_uri)
            header = {'Authorization': 'Bearer {}'.format(token)}
            res = requests.get("https://api.spotify.com/v1/me/player/devices", headers=header)
            devices = res.json()
            try:
                device_id = devices["devices"][0]['id']
            except:
                device_id = None
                print("デバイスIDが検出されませんでした。")
            try:
                playlist_id = id
            except:
                print("Playlist ID がありません")
            param = {'device_id':device_id,
                    'context_uri':'spotify:playlist:%s' % playlist_id}
            "一時停止"
            res = requests.put("https://api.spotify.com/v1/me/player/pause", data=json.dumps(param), headers = header)
            print(res) # 204なら成功
            # 今だけ（一時停止）
            """




            

            # recommend_tool = RecommendTool(UserActionState)
            # UserTrend = recommend_tool.getUserTrends()
            # suggested_tool = recommend_tool.getToolAnswer(check_schedule)
            # # suggested_tool, UserTrend = recommend_tool.getToolAnswer(check_schedule)
            
            dt_now_for_time_action = datetime.timedelta(hours=dt_now.hour, minutes=dt_now.minute) # 経路案内 # datetime.timedelta(hours=17, minutes=58) # 経路案内
            print("\n\n【テスト】現在時刻：", dt_now_for_time_action)
            # UserActionState = "WALKING"
            recommend_tool_time_action = RecommendTool(dt_now_for_time_action, UserActionState)
            recommend_tool_time_action.getUserTrends()
            suggested_tool = recommend_tool_time_action.getToolAnswer()

            if suggested_tool:
                print("\n--------------------------------------------------")
                print(suggested_tool)
                print("--------------------------------------------------")

                generate_prompt = GeneratePromptbyTool(suggested_tool)
                isExecuteTool = generate_prompt.getJudgeResult()
                if isExecuteTool:
                    # A
                    prompt_answer = generate_prompt.getGeneratedPrompt() # 機能からプロンプト生成する場合

                    # B
                    # # prompt_answer = suggested_tool + "を実行して。" # プロンプトに機能をそのまま入力する場合
                    # prompt_answer = suggested_tool + "して。" # プロンプトに機能をそのまま入力する場合

        
            if prompt_answer:
                print("プロンプト：", prompt_answer)
            
                # # A
                #     # PreInfo = f"行動状態は{UserActionState}です。ユーザーに関する情報が足りない場合は予定を参照して。"
                # # PreInfo = "ユーザーに関する情報が足りない場合は予定を参照して。"
                # PreInfo = "楽曲再生以外を実行する際のみ、ユーザーに関する情報が足りない場合は予定を参照して。"
                #     # PreInfo = "ユーザーに関する情報が足りない場合は予定を参照して。" + "楽曲再生以外を実行する場合はまず楽曲再生を停止して。"
                #     # PreInfo = f"""
                #     #         ユーザーの傾向は「{UserTrend}」です。 # ループになる
                #     #         ユーザーに関する情報が足りない場合は予定を参照して。
                #     #         """
                #     # text = prompt_answer + PreInfo
                # text = "現在時刻は" + str(dt_now) + "です。" + prompt_answer + PreInfo
                
                # # B
                # # PreInfo = "予定をもとに以下の情報を教えて。\n"
                # # text = "現在時刻は" + str(dt_now) + "です。" + PreInfo + prompt_answer

                # C
                text = "現在時刻は" + str(dt_now) + "です。" + prompt_answer
                
                if text:
                    print(" >> Waiting for response from Agent...")
                    """
                    Output
                    """
                    print("\n\n******************** [User Input] ********************\n", text)
                    # try:
                    response = agent.invoke(text) # できた(その後エラーはあるが)
                    # text_to_speach(final_response)

                    print("\n\n******************** [AI Answer] ********************\n")
                    # model.text_to_speach(response['output'])
                    userinterface.text_to_speach(response['output'])
                    # except:
                    #     print("\n##################################################\nERROR! ERROR! ERROR!\n##################################################")
                    #     print("もう一度入力してください。")




        
        # # 予定表 # 2024/05/28 追加
        # # dt_now = datetime.datetime.now() # 現在時刻
        
        # ##### 予定確認デモ
        # dt_now = datetime.datetime(2024, 5, 24, 10, 50)
        # text = "現在時刻は" + str(dt_now) + "です。" + "次の予定を教えて。" + "何分後にどこに向かえばいい？" # "今日の予定は何ですか？
        # Input = text
        # ##### 経路案内デモ
        # dt_now = datetime.datetime(2024, 5, 24, 8, 30)
        # # text = "家から会社までの経路教えて。予定から調べて教えて。"
        # text = "現在時刻は" + str(dt_now) + "です。" + "今日の予定から経路情報を簡潔に教えて。"
        # Input = text
        # ##### プロンプト生成デモ
        # dt_now = datetime.datetime(2024, 5, 24, 8, 30)
        # text = "現在時刻は" + str(dt_now) + "です。" \
        #        + "ユーザーに関する情報が足りない場合は予定を参照して。" \
        #        + "Prompt：最寄りの駅から目的地までの最短経路を教えて。" # プロンプト生成(3.5-turbo)
        #      # + "Prompt：経路検索を行いたいです。出発地と目的地を指定して、最適なルートと所要時間を教えてください。" # プロンプト生成(4o)
        # Input = text
        # ##### 予定が5分以内にあるか（これでもいいが、常時起動は実行金額が高くなりすぎる）
        # dt_now = datetime.datetime(2024, 5, 24, 10, 55)
        # text = "現在時刻は" + str(dt_now) + "です。" + "次の予定は5分以内に始まりますか？あるかないかで答えて。"\
        #        + "ユーザーに関する情報が足りない場合は予定を参照して。" 
        
        # # テスト
        # dt_now = datetime.datetime(2024, 5, 24, 8, 30)
        # text = "現在時刻は" + str(dt_now) + "です。" + "最適な経路を教えて。"\
        #         + "ユーザーに関する情報が足りない場合は予定を参照して。"  # プロンプト生成(4o)
        # Input = text
        # ##### 音声入力デモ
        # # 音声認識関数の呼び出し（現在時刻のみ事前に入力する）
        # # text += userinterface.recognize_speech() # 音声認識をする場合




        # if text:
        #     print(" >> Waiting for response from Agent...")
        #     """
        #     Output
        #     """
        #     print("\n\n******************** [User Input] ********************\n", text)
        #     try:
        #         response = agent.invoke(text) # できた(その後エラーはあるが)
        #         # text_to_speach(final_response)

        #         print("\n\n******************** [AI Answer] ********************\n")
        #         # model.text_to_speach(response['output'])
        #         userinterface.text_to_speach(response['output'])
        #     except:
        #         print("\n##################################################\nERROR! ERROR! ERROR!\n##################################################")
        #         print("もう一度入力してください。")