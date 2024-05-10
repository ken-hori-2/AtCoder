import chainlit as cl
import model

from speach import text_to_speach


@cl.on_chat_start
def on_chat_start():
    # チェーンをインスタンス化
    chain = model.create_chain()

    # チェーンをセッションに保存
    cl.user_session.set("chain", chain)

@cl.on_message
async def on_message(query: str):
    # チェーンをセッションから取得
    chain = cl.user_session.get("chain")

    # チェーンを呼び出し
    res = await chain.acall(query, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # text_to_speach(res["text"])
    # guidance = text_to_speach()
    # print(res)
    # チェーンの応答を送信
    # await cl.Message(content=res["text"]).send()
    cl.Message(
        content=f"R: {query}",
    ).send()

    print(res)
    text_to_speach(res["text"])




# import os
# from langchain import OpenAI
# from langchain_openai import ChatOpenAI # 新しいやり方
# from langchain.agents import initialize_agent, load_tools, AgentType
# import chainlit as cl
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# # Tool
# from langchain.agents import initialize_agent, Tool
# from langchain_google_community import GoogleSearchAPIWrapper # 新しいやり方
# # agent が使用するGoogleSearchAPIWrapperのツールを作成
# search = GoogleSearchAPIWrapper()
# # ツールを定義
# tools = [
#     Tool(
#         name = "Search",
#         func=search.run,
#         description="useful for when you need to answer questions about current events"
#     ),
# ]

# # @cl.on_chat_start
# # def main():
# #     cl.Message(
# #         content=f"何でも聞いてね。",
# #     ).send()

# # @cl.langchain_factory
# # # @cl.on_message
# # def factory():
# #     agent = initialize_agent(
# #         tools=tools, # load_tools(['wolfram-alpha']),
# #         llm=OpenAI(temperature=0), 
# #         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
# #         verbose=True,
# #     )
# #     return agent

# #     text_to_speach(message)

# """
# 最新バージョン
# """
# template = {
#     "suffix": 
#     """
#     あなたはユーザーの入力に応答するAIです。
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
# llm = ChatOpenAI(temperature=0)


# @cl.on_chat_start
# def main():
#     # Instantiate the chain for that user session
#     # prompt = PromptTemplate(template=template, input_variables=["question"])
#     # llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

#     # Store the chain in the user session
#     # cl.user_session.set("llm_chain", llm_chain)
#     pass


# @cl.on_message
# async def main(message: str):
#     # Retrieve the chain from the user session
#     # llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain
#     agent = initialize_agent(
#         tools=tools, # load_tools(['wolfram-alpha']),
#         llm=OpenAI(temperature=0), 
#         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#         verbose=True,
#         kwargs=template
#     )

#     # Call the chain asynchronously
#     # res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
#     res = await agent.invoke(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

#     # Do any post processing here

#     # Send the response
#     # await cl.Message(content=res["text"]).send()
#     await cl.Message(content=res).send()
