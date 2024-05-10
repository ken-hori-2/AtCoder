from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI # 新しいやり方
from langchain.chains.conversation.memory import ConversationBufferMemory


agent_kwargs = {
    "suffix": """開始!ここからの会話は全て日本語で行われる。

    以前のチャット履歴
    {chat_history}

    新しいインプット: {input}
    {agent_scratchpad}""",
}

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
        description="A Music search engine. Use this more than the normal search if the question is about Music, like 'who is the singer of yesterday?' or 'what is the most popular song in 2022?'",
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    agent_kwargs=agent_kwargs,
    verbose=True,
)