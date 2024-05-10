
# C:\Users\0107409377\Desktop\code\AtCoder\src\DemoApp\Search_and_LLM\LangChain-ChatGPT\main_Langchain.py　のコピー、JUDGE_func.py用にClass化した編集バージョン

import os
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '../LangChain_ChatGPT'))
sys.path.append(os.path.join(os.path.dirname(__file__), '.')) # '.\\')) # './'))


# sys.path.append(os.path.join(os.path.dirname(__file__), '../LangChain_ChatGPT/WebAPI'))
print(sys.path)
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
llm=ChatOpenAI(
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
from WebAPI.Spotify.spotify_api import MusicPlaybackQueryRun
from WebAPI.RestaurantSearch.hotpepper_api import RestaurantSearchQueryRun
from WebAPI.Localization.place_api import LocalizationQueryRun

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
        chain_open_meteo = APIChain.from_llm_and_api_docs(
            llm,
            open_meteo_docs.OPEN_METEO_DOCS,
            limit_to_domains=["https://api.open-meteo.com/"],
        )
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
                description="useful for when you need to answer questions about current events"
            ),
            Tool(
                name="Calculator",
                description="Useful for when you need to answer questions about math.",
                func=LLMMathChain.from_llm(llm=llm).run,
                coroutine=LLMMathChain.from_llm(llm=llm).arun,
            ),
            # 天気
            OpenWeatherMapQueryRun(),
            # こっちの天気でもできる
            Tool(
                name="Open-Meteo-API",
                description="Useful for when you want to get weather information from the OpenMeteo API. The input should be a question in natural language that this API can answer.",
                func=chain_open_meteo.run,
            ),
            WikipediaQueryRun(),
            Tool(
                name = "Calendar",
                func = calendar_tool.run,
                description="Useful for keeping track of appointments."
            ),
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
            LocalizationQueryRun(),
            RestaurantSearchQueryRun(),

        ]
        # agent が使用する memory の作成
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # テンプレート
        agent_kwargs = {
            "suffix": 
            """
            あなたはユーザーの入力に応答するスペシャリストです。
            人間の要求に答えてください。
            その際に適切なツールを使いこなして回答してください。
            さらに、あなたは過去のやりとりに基づいて人間と会話をしています。

            開始!ここからの会話は全て日本語で行われる。

            ### 解答例
            Human: やあ！
            GPT(AI) Answer: こんにちは！
            
            ### 以前のチャット履歴
            {chat_history}

            ###
            Human: {input}
            {agent_scratchpad}
            """,

            
        }
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
        )

        return agent
    
    def output(self, response):
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
        user_input = f"あなたは'PLAYBACK'または'OTHER'のどちらかで回答しなければならない。{response['output']}の文からは楽曲再生、停止などの操作を実行してると思いますか？\n楽曲操作を実行している場合は、'PLAYBACK'、楽曲操作以外を実行している場合は'OTHER'と回答してください。"
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
    agent = model.run()

    state = '走っている'
    Input = f"Action Detectionは{state}です。この行動にあったプレイリストをSpotifyAPIを使って再生もしくは一時停止してください。" # テンプレート化する
    response = agent.invoke(Input) # できた(その後エラーはあるが)
    # text_to_speach(final_response)

    final_response, llm_chain = model.output(response)
    if 'PLAYBACK' in final_response:
        print("\nMUSIC PLAYBACK!!!!! -> ガイダンス再生はしません。")
    else:
        print("\nOTHER!!!!!")
        """LLM 3個目"""
        state = '運動していません' # stableと認識
        question = f"Action Detectionは{state}です。この行動にあったプレイリストをSpotifyAPIを使って再生もしくは一時停止してください。" # テンプレート化する
        playback_response = agent.invoke(question) # できた(その後エラーはあるが)
        
        """LLM 4個目"""
        # templateに追加してもいいかも
        user_input = f"次の文をリスト形式ではなく、カギかっこなどのなく、一文当たり短い箇条書きにしてください。\n {response['output']}" # カギかっこなどのない、ただの文字列のみで、
        final_response = llm_chain.predict(input=user_input)
        # final_response = agent.invoke(user_input) # こっちだと「inputchat_historyoutput」と出力されてしまう
        # print(final_response)
        model.text_to_speach(final_response)
        # text_to_speach(response['output'])