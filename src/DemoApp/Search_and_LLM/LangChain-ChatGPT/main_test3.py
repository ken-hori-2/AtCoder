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






# Tool
from langchain_google_community import GoogleSearchAPIWrapper # 新しいやり方
# agent が使用するGoogleSearchAPIWrapperのツールを作成
search = GoogleSearchAPIWrapper()
from langchain.chains.llm_math.base import LLMMathChain
# import weather_tool
import weather_simple
import zfinance
# import weather_api
from weather_api import OpenWeatherMapQueryRun
from wikipedia_api import WikipediaQueryRun
from langchain.agents import load_tools, AgentExecutor, Tool, create_react_agent # 新しいやり方
# from google_calendar_api import GoogleCalendarTool
credentials_file = "credentials.json"
from GCalTool import GoogleCalendarTool

# ツールを定義
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Music Search",
        # 自作関数を指定可能
        func=lambda x: "'All I Want For Christmas Is You' by Mariah Carey.", #Mock Function
        # func=zfinance.music,
        description="A Music search engine. Use this more than the normal search if the question is about Music, like 'who is the singer of yesterday?' or 'what is the most popular song in 2022?'",
    ),
    # Tool(
    #     name="Podcast-API",
    #     description="Use the Listen Notes Podcast API to search all podcasts or episodes. The input should be a question in natural language that this API can answer.",
    #     func=chain.run,
    # ),
    # Tool(
    #     name="Calculations",
    #     # 組み込み済みAPIも指定可能??
    #     func=load_tools(["llm-math"], llm=llm),
    #     description="It is useful when performing calculations.",
    # )
    # load_tools(["llm-math"], llm=llm)

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
    
    OpenWeatherMapQueryRun(),
    WikipediaQueryRun(),
    GoogleCalendarTool(),

    
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
    # handle_parsing_errors=True, # パースエラーを例外処理で回避しているだけかも（上のモデルとセット）
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, # パースエラー回避
    memory=memory,
    agent_kwargs=agent_kwargs,
    verbose=True,
    
)
# agent.invoke("クリスマスに関連する歌で最も有名な曲は？")
# agent.invoke("東京都の人口は、何人ですか？また、日本の総人口の何パーセントを占めていますか？")
# agent.invoke("123×123を計算して。")
# agent.invoke("今日の東京の天気は？")
agent.invoke("今日のカレンダーの予定を5つ教えて。無かったら無いと言ってください。")

# agent.invoke("ソニーの最近の事業について教えてください。") # wikipediaとsearchのいい例

# agent.invoke("徳川家康とはどんな人ですか？")

# agent.invoke("Googleの直近の株価が知りたい")



















































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
