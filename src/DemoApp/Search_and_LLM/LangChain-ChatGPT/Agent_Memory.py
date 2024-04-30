"以前のやり方"
from langchain.agents import AgentExecutor, ZeroShotAgent, Tool # 少し古い
from langchain.chains import LLMChain
"以前のやり方"

# Google Search API
# from langchain.utilities import GoogleSearchAPIWrapper
from langchain_google_community import GoogleSearchAPIWrapper # 新しいやり方
# Memory
from langchain.memory import ConversationBufferMemory
# ChatOpenAI GPT 3.5
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI # 新しいやり方

from langchain.chains import ConversationChain
from langchain.prompts.chat import (
    # メッセージテンプレート
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
# agentと agentが使用するtool
from langchain.agents import Tool, initialize_agent

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
os.environ["OPENAI_API_KEY"]
os.environ["GOOGLE_API_KEY"]
os.environ["GOOGLE_CSE_ID"]
# LANGCHAIN_HANDLER 設定
# os.environ["LANGCHAIN_HANDLER"] = "langchain"
# from langchain_core.tracers.context import tracing_v2_enabled
# # with tracing_v2_enabled():
#     # LangChain code will automatically be traced

# agent が使用するGoogleSearchAPIWrapperのツールを作成
search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    )
]
# agent が使用する memory の作成
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Agent用 prefix, suffix
# prefix = """Anser the following questions as best you can, but speaking Japanese. You have access to the following tools:"""
prefix = """Anser the following questions in Japanese. You have access to the following tools:""" # 極力日本語でお願いするのではなく、日本語縛りにしたが結果はあまり変化ない
suffix = """Begin! Remember to speak Japanese when giving your final answer. Use lots of "Args"""
# prefix = """日本語で回答してください。 あなたはtoolsを使えます。"""
# suffix = """final answer では日本語で話すことを意識してください。 "Args"をなるべく多く使ってください。"""

# agent の使用する LLM
llm=ChatOpenAI(temperature=0)

# tool, memory, llm を設定して agent を作成
"1. 今回のやり方 *****"
agent_chain = initialize_agent( # おそらくこのinitializeが英語になる原因
                tools, 
                llm, 
                agent="chat-conversational-react-description", # 英語になることが多い
                    # agent="zero-shot-react-description", # 以前のやり方
                    # agent="chat-zero-shot-react-description", # 以前のやり方
                    # handle_parsing_errors=True, # 以前のやり方 (ZeroShotAgentとセットで必要)
                verbose=True, 
                memory=memory, 
                prefix=prefix, 
                suffix=suffix
                )
    # agent_chain = ZeroShotAgent(
    #                 tools, 
    #                 llm, 
    #                 # agent="chat-conversational-react-description", # 英語になることが多い
    #                 # agent=ZeroShotAgent(), # 以前のやり方
    #                 verbose=True, 
    #                 memory=memory, 
    #                 prefix=prefix, 
    #                 suffix=suffix
    # )
"1. 今回のやり方 *****"

"2. 以前のやり方（単発の要求）*****" # Toolsを使えるが、会話を引き継げていない
# # Agent のprompt
# # ZeroShotAgentは非推奨だが、一旦そのままにする
# prompt = ZeroShotAgent.create_prompt(
# # prompt = create_react_agent(
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
# agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True) # パースエラーの対処？を追加
"2. 以前のやり方（単発の要求）*****"
"3. やり方その3（Toolsを使わない）*****" # おそらくLLMのみ
# # llm = ChatOpenAI(temperature=0)
# # Memory の作成と参照の獲得
# memory = ConversationBufferMemory() # これがないと3.のやり方はエラーになる
# conversation = ConversationChain(
#     llm=llm, 
#     verbose=True, 
#     memory=memory
# )
"3. やり方その3（Toolsを使わない）*****"

# 会話ループ
user = ""
while user != "exit":
    user = input("何か質問してください。\n >")
    print(user)
    # 入力はagent経由
    # ai = agent_chain.invoke(input=user) # run
    "1. *****"
    ai = agent_chain.invoke(user)
    print(ai["output"])
    "1. *****"
    "3. *****"
    # ai = conversation.predict(input=user)
    "3. *****"
    # print(ai)
    # print("type:", type(ai))