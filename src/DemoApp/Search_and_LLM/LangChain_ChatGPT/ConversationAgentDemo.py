
# C:\Users\0107409377\Desktop\code\AtCoder\src\DemoApp\Search_and_LLM\LangChain-ChatGPT\main_Langchain.py　のコピー、JUDGE_func.py用にClass化した編集バージョン
# このファイルを直接実行する場合のみ必要
from dotenv import load_dotenv
# load_dotenv()
load_dotenv('WebAPI\\Secret\\.env') # たぶんload_dotenv()のみでいい
# このファイルを直接実行する場合のみ必要

import os
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '../LangChain_ChatGPT'))
sys.path.append(os.path.join(os.path.dirname(__file__), '.')) # '.\\')) # './'))


# sys.path.append(os.path.join(os.path.dirname(__file__), '../LangChain_ChatGPT/WebAPI'))
# print(sys.path)
# sys.path.append(os.path.join(os.path.dirname(__file__), 'C:\\Users\\0107409377\\Desktop\\code\\AtCoder\\src\\DemoApp\\Search_and_LLM\\LangChain_ChatGPT\\WebAPI'))

# sys.path.append(os.path.join(os.path.dirname(__file__), './Search_and_LLM'))


from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
# from langchain.llms import OpenAI
from langchain_community.llms import OpenAI
# from langchain import LLMMathChain, SerpAPIWrapper
from langchain_openai import ChatOpenAI # 新しいやり方
# Memory
from langchain.memory import ConversationBufferMemory
from langchain.agents import load_tools


"""
mainファイルで一回読み込んでいるのでいらない
"""
# from dotenv import load_dotenv
# # .envファイルの内容を読み込見込む
# load_dotenv()
# agent の使用する LLM
# llm=ChatOpenAI(
#     temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
# ) # チャット特化型モデル
llm=ChatOpenAI(
    model="gpt-4o",
    # model="gpt-3.5-turbo",
    temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
) # チャット特化型モデル
# llm = OpenAI(temperature=0)

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

# from WebAPI.Spotify.spotify_api import MusicPlaybackQueryRun
from WebAPI.Spotify.spotify_api_Recommend_version import MusicPlaybackQueryRun

from WebAPI.RestaurantSearch.hotpepper_api import RestaurantSearchQueryRun
from WebAPI.Localization.place_api import LocalizationQueryRun
from WebAPI.Schedule.OutlookSchedule_api import ScheduleQueryRun # 2024/05/28 追加

from WebAPI.Calendar.GCalTool import GoogleCalendarTool


from langchain.chains.api.base import APIChain
from langchain.chains.api import news_docs, open_meteo_docs, podcast_docs, tmdb_docs

##################
# Pyttsx3を初期化 #
##################
import pyttsx3
engine = pyttsx3.init()
# 読み上げの速度を設定する
rate = engine.getProperty('rate')
engine.setProperty('rate', rate)
#volume デフォルト値は1.0、設定は0.0~1.0
volume = engine.getProperty('volume')
engine.setProperty('volume',1.0)
# Kyokoさんに喋ってもらう(日本語)
engine.setProperty('voice', "com.apple.ttsbundle.Kyoko-premium")


# 2024/05/28 追加
from UIModel_copy import UserInterfaceModel # ユーザーとのやり取りをするモデル
userinterface = UserInterfaceModel()
# from langchain_community.tools.human.tool import HumanInputRun



class Langchain4Judge():
    def text_to_speach(self, response):
        # ストリーミングされたテキストを処理する
        fullResponse = ""
        RealTimeResponce = ""

        # 随時レスポンスを音声ガイダンス
        for chunk in response:
            text = chunk

            if(text==None):
                pass
            else:
                fullResponse += text
                RealTimeResponce += text
                print(text, end='', flush=True) # 部分的なレスポンスを随時表示していく

                target_char = ["。", "！", "？", "\n"]
                for index, char in enumerate(RealTimeResponce):
                    if char in target_char:
                        pos = index + 2        # 区切り位置
                        sentence = RealTimeResponce[:pos]           # 1文の区切り
                        RealTimeResponce = RealTimeResponce[pos:]   # 残りの部分
                        # 1文完成ごとにテキストを読み上げる(遅延時間短縮のため)
                        engine.say(sentence)
                        engine.runAndWait()
                        break
                    else:
                        pass


    def run(self, credentials_file):
        """
        天気用のツール（二つ目なので現在使っていない）
        """
        # chain_open_meteo = APIChain.from_llm_and_api_docs(
        #     llm,
        #     open_meteo_docs.OPEN_METEO_DOCS,
        #     limit_to_domains=["https://api.open-meteo.com/"],
        # )
        """
        NEWS用のツール 2024/05/05
        """
        news_api_key = os.environ["NEWS_API_KEY"] # kwargs["news_api_key"]
        chain_news = APIChain.from_llm_and_api_docs(
            llm,
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
                func=LLMMathChain.from_llm(llm=llm).run,
                coroutine=LLMMathChain.from_llm(llm=llm).arun,
            ),
            # 天気
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
            RouteSearchQueryRun(),
            MusicPlaybackQueryRun(),
            # LocalizationQueryRun(), # エラー回避
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

            
            
            必要なら人間に追加の入力を求めてください。



            開始!ここからの会話は全て日本語で行われる。

            ### 解答例
            [Human]: やあ！
            [AI]: こんにちは！
            
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
            llm,
            # agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, # 複数引数の場合は新しいagentにする必要がある
            # handle_parsing_errors=True, # パースエラーを例外処理で回避しているだけかも（上のモデルとセット）
            # agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, # パースエラー回避
            memory=memory,
            agent_kwargs=agent_kwargs,
            verbose=True,
            handle_parsing_errors=True, # パースエラーを例外処理で回避

            max_iterations=10 # これがあるとagentのイタレーションに制限をかけられる
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
            llm=llm,
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
    model = Langchain4Judge()
    # ここのパスが重要 (DM4L.pyはDemoApp/にあるので、その直下のSearch_and_LLM/から指定)
    credentials_file = "WebAPI\\Secret\\credentials.json"
    agent = model.run(credentials_file)
    
    import datetime
    from TriggerHeadGesture import Trigger
    """
    EdgeAIによるトリガーの定義
    """
    trigger = Trigger()

    for i in range(5):
        print("***** センシング中 *****")
        # UserActionState = trigger.run()
        UserActionState = "HeadGesture"
        # UserActionState = "WALKING" # テスト用
        # UserActionState = "STABLE" # テスト用
        print("DNN検出結果：", UserActionState)
        print("***** センシング終了 *****")

        if "HeadGesture" in UserActionState:
            userinterface.text_to_speach("ジェスチャーを検知しました。\n")


            text = ""
            question = ""
            """
            デモするユースケースに応じて手動で時刻を設定する
            """
            # # dt_now = datetime.datetime.now()
            # # dt_now = datetime.datetime(2024, 5, 24, 8, 30)

            # 部定例デモ
            # dt_now = datetime.datetime(2024, 6, 3, 8, 30)

            dt_now = datetime.datetime(2024, 6, 12, 19, 5) # ジム(run:up tempo, walk:slow tempo, stable:stop)   # 行動検出と連動モード
            # dt_now = datetime.datetime(2024, 6, 12, 12, 5) # 昼食（ゆっくり休みたい）                          # リラックスモード
            dt_now = datetime.datetime(2024, 6, 12, 8, 00) # 出勤（気分上げたい）                                # アップテンポモード
            """
            デモするユースケースに応じて手動で時刻を設定する
            """

            for i in range(5):
                # ##### 音声入力デモ
                # # 音声認識関数の呼び出し（現在時刻のみ事前に入力する）
                # # text += userinterface.recognize_speech() # 音声認識をする場合
                # userinterface.text_to_speach("Please Speak in 3 seconds\n")
                print("入力してください。\n")
                # text = userinterface.recognize_speech() # 音声認識をする場合
                text = "楽曲再生して。" # "アップテンポな曲を再生して。"
                
                # text = "現在の日本の総理大臣は？"
                # text = "今日の東京の天気は？"
                # text = "次の予定を教えて。" + "何分後にどこに向かえばいい？"
                # text = "今日の経路情報を簡潔に教えて。" # 駅名を指定してもOK
                # text = "今日のおすすめのレストラン3つ教えて"

                # text = "ランニングに合う楽曲を再生して。"

                # # 記憶保持の確認
                # if i == 0:
                #     text = "私の名前は「けん」です。25歳です。"
                # elif i == 1:
                #     text = "私の名前を教えて。"
                # elif i == 2:
                #     text = "私の年齢を教えて。"
                # elif i == 3:
                #     break

                if text:

                    if ("終わり" in text) or ("終了" in text) or ("いらない" in text):
                        userinterface.text_to_speach("会話を終了します。\n")
                        break
                    else:
                        print("\n\n******************** [User Input] ********************\n", text)
                        question = "現在時刻は" + str(dt_now) + "です。" + text\
                        + "ユーザーに関する情報が足りない場合は予定を参照して。"  # プロンプト生成(4o)
                    
                        print(" >> Waiting for response from Agent...")
                        # try:
                        response = agent.invoke(question) # できた(その後エラーはあるが)
                        print("\n\n******************** [AI Answer] ********************\n")
                        userinterface.text_to_speach(response['output'])
                        # except:
                        #     print("\n##################################################\nERROR! ERROR! ERROR!\n##################################################")
                        #     print("もう一度入力してください。")
                    

                    # break # test
        # break # test