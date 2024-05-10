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
from langchain.agents import Tool, initialize_agent, AgentType
import os
from dotenv import load_dotenv
# .envファイルの内容を読み込見込む
load_dotenv()




"""
main
"""
# from . 
import weather_tool
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_functions

# agent が使用するGoogleSearchAPIWrapperのツールを作成
search = GoogleSearchAPIWrapper()
# tools = [
#     Tool(
#         name = "Search",
#         func=search.run,
#         description="useful for when you need to answer questions about current events"
#     )
# ]
tools = [
    weather_tool.weather_api, 
    # Tool(
    #     name = "Search",
    #     func=search.run,
    #     description="useful for when you need to answer questions about current events"
    # )
    # GoogleSearchAPIWrapper()
]
# tools = load_tools(["serpapi", "llm-math"], llm=llm)

# llm_with_tools = ChatOpenAI(temperature=0, model='gpt-3.5-turbo').bind(functions=[format_tool_to_openai_function(t) for t in tools])




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
                # agent="chat-conversational-react-description", # 英語になることが多い
                # agent=ZeroShotAgent(llm_chain=llm_chain, tools=tools),
                # agent="zero-shot-react-description", # 以前のやり方
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    # agent="chat-zero-shot-react-description", # 以前のやり方
                    # handle_parsing_errors=True, # 以前のやり方 (ZeroShotAgentとセットで必要)
                verbose=True, 
                # memory=memory, 
                # prefix=prefix, 
                # suffix=suffix
                )
"1. 今回のやり方 *****"

# "2. 以前のやり方（単発の要求）*****" # Toolsを使えるが、会話を引き継げていない
# # Agent のprompt
# # ZeroShotAgentは非推奨だが、一旦そのままにする

# prompt = ZeroShotAgent.create_prompt(
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
# # llm_chain =  LLMChain(llm=ChatOpenAI(temperature=0), prompt=chat_prompt)
# # llm_chain =  LLMChain(llm=llm_with_tools, prompt=chat_prompt)
# llm_chain =  LLMChain(llm=llm, prompt=chat_prompt)

# # tool を与えてagent に
# tool_names = [tool.name for tool in tools]
# # ZeroShotAgentは非推奨だが、一旦そのままにする
# agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
# # agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)
# # verbose=True で途中経過を出力
# agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True) # パースエラーの対処？を追加
# "2. 以前のやり方（単発の要求）*****"


# 会話ループ
user = ""
user = input("何か質問してください。\n >")
print(user)
# 入力はagent経由
ai = agent_chain.invoke(user)
print(ai) # ["output"])

# conversationHistory = []
# text = ""

# for i in range(3):
#     # print(f"text[{i}]:{text}")
#     # if i == 0:
#     #     text = "今日の東京の天気は？"
#     # if i == 1:
#     #     text =  "傘は必要ですか？"
#     # if i == 2:
#     #     text =  "何時から降る予報ですか？"
#     if i == 0:
#         text = "資格とは何ですか。"
#     if i == 1:
#         # text =  "取得のメリットとデメリットを教えて。"
#         text =  "メリットとデメリットを教えて。"
#     if i == 2:
#         text =  "最近の技術者向けのおすすめは何ですか。"
#     print(f"text[{i}]:{text}")
        
#     if text:
#         # print(" >> Waiting for response from ChatGPT...")
#         user_action = {"role": "user", "content": text}
#         conversationHistory.append(user_action)
#         ai = agent_chain.invoke(input=conversationHistory)
#         # ai = agent_chain.invoke(input=text) # Historyを渡す必要はないかも⇒会話が引き継がれない
#         user_action = {"role": "user", "content": ai}
#         print("******************** responce ********************\n", ai["output"])
#         print("******************** responce ********************\n")
#         conversationHistory.append(user_action)