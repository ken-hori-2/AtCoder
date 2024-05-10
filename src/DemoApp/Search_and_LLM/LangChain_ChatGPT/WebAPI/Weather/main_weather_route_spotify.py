import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(os.path.join(os.path.dirname(__file__), '..\Calendar'))
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
from dotenv import load_dotenv
# .envファイルの内容を読み込見込む
load_dotenv()


# # Serpapiのラッパーをインスタンス化
# search = SerpAPIWrapper()

# agent の使用する LLM
llm=ChatOpenAI(
    temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
) # チャット特化型モデル


# llm = OpenAI(temperature=0)




# from langchain.agents.conversational_chat.base import ConversationalChatAgent

# ConversationalChatAgent._validate_tools = lambda *_, **__: ...






# Tool
from langchain_google_community import GoogleSearchAPIWrapper # 新しいやり方
# agent が使用するGoogleSearchAPIWrapperのツールを作成
search = GoogleSearchAPIWrapper()
from langchain.chains.llm_math.base import LLMMathChain
# import weather_tool
# import weather_simple
# import zfinance
# import weather_api
from weather_api import OpenWeatherMapQueryRun
from Wikipedia.wikipedia_api import WikipediaQueryRun
from langchain.agents import load_tools, AgentExecutor, Tool, create_react_agent # 新しいやり方
# from google_calendar_api import GoogleCalendarTool

from RouteSearch.route_api import RouteSearchQueryRun

# sys.path.append("../Spotify")
from Spotify.spotify_api import MusicPlaybackhQueryRun


# 一旦コメントアウト
# credentials_file = "C:/Users/0107409377/Desktop/code/AtCoder/src/DemoApp/Search_and_LLM/LangChain-ChatGPT/Calendar/credentials.json" # "credentials.json"
# from Calendar.GCalTool import GoogleCalendarTool
# # ツールを作成
# calendar_tool = GoogleCalendarTool(credentials_file, llm=ChatOpenAI(temperature=0), memory=None)


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

def text_to_speach(response):
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


chain = APIChain.from_llm_and_api_docs(
    llm,
    open_meteo_docs.OPEN_METEO_DOCS,
    limit_to_domains=["https://api.open-meteo.com/"],
)






"""
NEWS用のツール 2024/05/05
"""
# news
news_api_key = os.environ["NEWS_API_KEY"] # kwargs["news_api_key"]
chain = APIChain.from_llm_and_api_docs(
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

# ツールを定義
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    # Tool(
    #     name="Music Search",
    #     # 自作関数を指定可能
    #     func=lambda x: "'All I Want For Christmas Is You' by Mariah Carey.", #Mock Function
    #     # func=zfinance.music,
    #     description="A Music search engine. Use this more than the normal search if the question is about Music, like 'who is the singer of yesterday?' or 'what is the most popular song in 2022?'",
    # ),
    
    Tool(
        name="Calculator",
        description="Useful for when you need to answer questions about math.",
        func=LLMMathChain.from_llm(llm=llm).run,
        coroutine=LLMMathChain.from_llm(llm=llm).arun,
    ),
    # """
    # C:/Users/0107409377/.pyenv/pyenv-win/versions/3.12.0/Lib/site-packages/langchain/agents/load_tools.py
    # を呼び出して使うものをそのまま持ってきた
    # """
    # weather_tool.weather_api, 
    # Tool(
    #     name="Weather",
    #     func=weather_simple.get_current_weather,
    #     description="It is useful when you want to know weather information."
    # ),
    # # Tool(
    # #     name="Current Stock Price",
    # #     func=zfinance.current_stock_price,
    # #     description="株式銘柄（米）の最新株価取得する関数"
    # # ),
    # zfinance.current_stock_price,
    # # Tool(
    # #     name="Stock Performance",
    # #     func=zfinance.stock_performance,
    # #     description="過去days間で変動した株価[%]"
    # # )
    # # zfinance.stock_performance,
    # weather_api.OpenWeatherMapQueryRun.run,
    
    # 天気
    OpenWeatherMapQueryRun(),
    # こっちの天気でもできる
    Tool(
        name="Open-Meteo-API",
        description="Useful for when you want to get weather information from the OpenMeteo API. The input should be a question in natural language that this API can answer.",
        func=chain.run,
    ),
    WikipediaQueryRun(),
    # GoogleCalendarTool(),
    # GoogleCalendarTool(credentials_file, llm=ChatOpenAI(temperature=0), memory=None),
    # Tool(
    #     name = "Calendar",
    #     func = calendar_tool.run,
    #     description="Useful for keeping track of appointments."
    # ),

    Tool(
        name="News-API",
        description="Use this when you want to get information about the top headlines of current news stories. The input should be a question in natural language that this API can answer.",
        func=chain.run,
    ),
    # Tool(
    #     name="Podcast-API",
    #     description="Use the Listen Notes Podcast API to search all podcasts or episodes. The input should be a question in natural language that this API can answer.",
    #     func=chain.run,
    # )
    RouteSearchQueryRun(),

    MusicPlaybackhQueryRun(),

    
]

# add_tools = load_tools(["llm-math"], llm=llm)
# tools.append(add_tools)

# agent = initialize_agent(
#         tools, OpenAI(temperature=0), 
#         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
#         verbose=True
# )

# agent が使用する memory の作成
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent_kwargs = {
    "suffix": """開始!ここからの会話は全て日本語で行われる。

    以前のチャット履歴
    {chat_history}

    新しいインプット: {input}
    {agent_scratchpad}""",
}
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
# agent.invoke("クリスマスに関連する歌で最も有名な曲は？")
# agent.invoke("東京都の人口は、何人ですか？また、日本の総人口の何パーセントを占めていますか？")
# agent.invoke("123×123を計算して。")
# agent.invoke("今日の東京の天気は？")
# agent.invoke("今日の大阪の天気は？")
state = 'running'
state = '止まっている'
state = '走っている'
state = '歩いている'
question = (f"Action Detectionは{state}です。この行動にあったプレイリストをSpotifyAPIを使って再生もしくは一時停止してください。") # テンプレート化する

"""
Schedule
"""
# agent.invoke("今日のカレンダーの予定を5つ教えて。無かったら無いと言ってください。")
# agent.invoke("今日の予定を教えて。") # 現在のGCalToolではseachをうまく返せない
""" -> z_calendar.pyのやり方でうまくできないか？ -> できた"""
# agent.invoke("今日の12時にBBQの予定を入れて。")
# agent.invoke("明日の12時~15時に従妹とBBQの予定を入れて") # できた(その後エラーはあるが)

"""
乗換案内
"""
# question = "本厚木駅から東京駅までの経路を教えて。また、どの車両に乗るのがいいかも合わせて教えて。"
# question = "本厚木駅から東京駅までの経路を教えて。箇条書きで簡潔にして。"
# question = "本厚木から東京までの経路を教えて。" # 箇条書きで簡潔にして。"
# question = "坂城駅から東京駅までの経路を教えて。JR北陸新幹線を使いたいです。" # 箇条書きで簡潔にして。"
# question = "上田駅から東京駅までの経路を教えて。" # 新幹線を使いたいです。"
# question = "上田駅から東京駅までの速い経路を教えて。" # 直接新幹線が必要か指示されていなくても早さ重視なら新幹線を使うようなパラメータ設定にすることを指示するように記述⇒うまくできた
# question = "上田駅から東京駅までの新幹線を使った経路を教えて。"
# question = "上田駅から東京駅までの乗換回数の少ない経路を教えて。" # 安い経路を教えて。" # 表示の優先度の確認



# agent.invoke("今日のニュースは？")
# agent.invoke("ラジオ聞かせて。")

# agent.invoke("ソニーの最近の事業について教えてください。") # wikipediaとsearchのいい例
# agent.invoke("徳川家康とはどんな人ですか？")
# agent.invoke("Googleの直近の株価が知りたい")

"""
Output
# """
# response = agent.invoke(question) # できた(その後エラーはあるが)
# print(response['output'])
# print(type(response['output']))
# text_to_speach(response['output'])



















































"2. 以前のやり方（単発の要求）*****" # Toolsを使えるが、会話を引き継げていない
# from langchain.agents import AgentExecutor, ZeroShotAgent, Tool # 少し古い
# from langchain.prompts.chat import (
#     # メッセージテンプレート
#     ChatPromptTemplate,
#     # System メッセージテンプレート
#     SystemMessagePromptTemplate,
#     # assistant メッセージテンプレート
#     AIMessagePromptTemplate,
#     # user メッセージテンプレート
#     HumanMessagePromptTemplate,
# )
# from langchain.chains import LLMChain
# # Memory
# from langchain.memory import ConversationBufferMemory
# # Agent のprompt
# # ZeroShotAgentは非推奨だが、一旦そのままにする
# # memory_2 = ConversationBufferMemory()
# # agent が使用する memory の作成
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# prefix = """Anser the following questions in Japanese. You have access to the following tools:""" # 極力日本語でお願いするのではなく、日本語縛りにしたが結果はあまり変化ない
# suffix = """Begin! Remember to speak Japanese when giving your final answer. Use lots of "Args"""
# prompt = ZeroShotAgent.create_prompt( # prompt = create_react_agent(
#     tools,
#     prefix=prefix,
#     suffix=suffix,
#     input_variables=[]
# )
# # ChatOpenAI に食わせるmessageのリスト
# messages = [
#     SystemMessagePromptTemplate(prompt=prompt), # ZeroShotAgentで作ったpromptをSystemに
#     HumanMessagePromptTemplate.from_template("{input}\n\nThis was your previous work "
#                                              f"(but I haven't seen any of it! I only see what "
#                                              "you return as final answer):\n{agent_scratchpad}")
# ]
# # ChatPromptのテンプレート
# chat_prompt = ChatPromptTemplate.from_messages(messages)
# # Chainの作成
# # llm_chain = langchain.chains.LLMChain
# llm_chain =  LLMChain(llm=ChatOpenAI(temperature=0), prompt=chat_prompt)
# # tool を与えてagent に
# tool_names = [tool.name for tool in tools]
# # ZeroShotAgentは非推奨だが、一旦そのままにする
# agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
# # agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)
# # verbose=True で途中経過を出力
# agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True) # パースエラーの対処？を追加
"2. 以前のやり方（単発の要求）*****"

# agent_chain.invoke("クリスマスに関連する歌で最も有名な曲は？")
