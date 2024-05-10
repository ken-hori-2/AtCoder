# Agentの読み込み
from langchain.agents import AgentExecutor, ZeroShotAgent, Tool # 少し古い
from langchain.agents import load_tools, AgentExecutor, Tool, create_react_agent # 新しいやり方

# Google Search API
# from langchain.utilities import GoogleSearchAPIWrapper # 少し古い
# from langchain_community import GoogleSearchAPIWrapper # 間違い
# from langchain_community.utilities import GoogleSearchAPIWrapper # 新しいやり方
from langchain_google_community import GoogleSearchAPIWrapper # 新しいやり方

# ChatOpenAI GPT 3.5
# from langchain.chat_models import ChatOpenAI # 少し古いかも
from langchain_openai import ChatOpenAI # 新しいやり方
# from langchain import LLMChain # 少し古いかも
# import langchain
from langchain.chains import LLMChain

from langchain.prompts.chat import (
    # 汎用のメッセージテンプレート(将来の拡張に備えているのかも)
    ChatPromptTemplate,
    # System メッセージテンプレート
    SystemMessagePromptTemplate,
    # assistant メッセージテンプレート
    AIMessagePromptTemplate,
    # user メッセージテンプレート
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    # それぞれ GPT-3.5-turbo API の assistant, user, system role に対応
    AIMessage,
    HumanMessage,
    SystemMessage
)

# # env に読み込ませるAPIキーの類
# import key

# # 環境変数にAPIキーを設定
# import os
# os.environ["OPENAI_API_KEY"] = key.OPEN_API_KEY
# os.environ["GOOGLE_CSE_ID"] = key.GOOGLE_CSE_ID
# os.environ["GOOGLE_API_KEY"] = key.GOOGLE_API_KEY

import os
from dotenv import load_dotenv
# .envファイルの内容を読み込見込む
load_dotenv()
# os.environを用いて環境変数を表示させます
# print(os.environ['OpenAI_API_KEY'])
# key = 
# os.environ['OpenAI_API_KEY']
os.environ["OPENAI_API_KEY"]
# ⭐️ここにOpenAIから取得したキーを設定します。⭐️
# OPENAI_API_KEY = 
# os.environ["OPENAI_API_KEY"] = key
# GOOGLE_API_KEY = 
# os.environ['GoogleMap_API_KEY']
os.environ["GOOGLE_API_KEY"]
# GOOGLE_CSE_ID = 
# os.environ['Google_Custom_ID']
os.environ["GOOGLE_CSE_ID"]

# GoogleSearchAPIWrapperのツールを作成
search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    )
]

# Agent用 prefix, suffix
prefix = """Anser the following questions as best you can, but speaking Japanese. You have access to the following tools:"""
suffix = """Begin! Remember to speak Japanese when giving your final answer. Use lots of "Args"""

# Agent のprompt
# ZeroShotAgentは非推奨だが、一旦そのままにする
prompt = ZeroShotAgent.create_prompt(
# prompt = create_react_agent(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=[]
)

# ChatOpenAI に食わせるmessageのリスト
messages = [
    SystemMessagePromptTemplate(prompt=prompt), # ZeroShotAgentで作ったpromptをSystemに
    HumanMessagePromptTemplate.from_template("{input}\n\nThis was your previous work "
                                             f"(but I haven't seen any of it! I only see what "
                                             "you return as final answer):\n{agent_scratchpad}")
]

# ChatPromptのテンプレート
chat_prompt = ChatPromptTemplate.from_messages(messages)

# Chainの作成
# llm_chain = langchain.chains.LLMChain
llm_chain =  LLMChain(llm=ChatOpenAI(temperature=0), prompt=chat_prompt)

# tool を与えてagent に
tool_names = [tool.name for tool in tools]
# ZeroShotAgentは非推奨だが、一旦そのままにする
agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
# agent = create_react_agent(llm_chain=llm_chain, allowed_tools=tool_names)

# agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True) # パースエラーの対処？を追加

# runコマンドは古いので、invokeを使う
# agent_executor.run("中小企業診断士の試験科目について教えて")
# agent_executor.run("今日の東京の天気を教えて。")

# Action : Serach
# agent_executor.invoke("今日の東京の天気を教えて。") # agent_executor.invoke({"input":"今日の東京の天気を教えて。"})
# agent_executor.invoke("LangChainについて説明してください。")
# agent_executor.invoke("ソニーについて具体的に説明してください。")
# agent_executor.invoke("本厚木駅周辺のおいしい店三つ教えて。")
agent_executor.invoke("人の目はなぜ錯覚することがあるんですか？")

# Action : LLM Only
# agent_executor.invoke("AIについて説明してください。")


# agent_executor.invoke("2024年5月1日10:00の玉川学園前駅から大森駅までの経路案内を教えて。何分の電車に乗ればいいですか。")