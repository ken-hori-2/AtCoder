import os
from dotenv import load_dotenv
load_dotenv()

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import SimpleMemory

# llm = OpenAI(model_name="gpt-3.5-turbo")
# prompt_1 = PromptTemplate(
#     input_variables=["adjective", "job", "time"],
#     template="{time}において、{adjective}{job}に一番オススメのプログラミング言語は?\nプログラミング言語：",
# )
# chain_1 = LLMChain(llm=llm, prompt=prompt_1, output_key="programming_language")

# prompt_2 = PromptTemplate(
#     input_variables=["programming_language", "time"],
#     template="現在は{time}だとします。{programming_language}を学ぶためにやるべきことを3ステップで100文字で教えて。",
# )
# chain_2 = LLMChain(llm=llm, prompt=prompt_2, output_key="learning_step")

# overall_chain = SequentialChain(
#     chains=[chain_1, chain_2], 
#     input_variables=["adjective", "job"],
#     output_variables=["programming_language", "learning_step"],
#     verbose=True,
#     memory=SimpleMemory(memories={"time": "2030年"}),
# )
# output = overall_chain({
#     "adjective": "ベテランの",
#     "job": "エンジニア",
# })
# print(output)

"""
3. Buffer Memoryの使い方
"""
# from langchain.llms import OpenAI
# from langchain.memory import ConversationBufferMemory

# memory = ConversationBufferMemory()

# memory.save_context(
#     {"input": "AIとは何？"},
#     {"output": "AIとは、人工知能のことです。"},
# )
# print(memory.load_memory_variables({}))

from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

# memory = ConversationBufferMemory(return_messages=True)

# memory.save_context(
#     {"input": "AIとは何？"},
#     {"output": "AIとは、人工知能のことです。"},
# )
# print(memory.load_memory_variables({}))

# from langchain.llms import OpenAI
# from langchain.chains import ConversationChain


# llm = OpenAI(model_name="gpt-3.5-turbo")
# conversation = ConversationChain(
#     llm=llm, 
#     verbose=True, 
#     memory=ConversationBufferMemory()
# )

# conversation("AIとは何？")



"""
WARNING
"""
# LangChainDeprecationWarning: The class `LLMChain` was deprecated in LangChain 0.1.17 and will be removed in 0.3.0. Use RunnableSequence, e.g., `prompt | llm` instead.


from langchain.memory import ConversationBufferMemory
# from langchain import OpenAI, LLMChain, PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
llm=ChatOpenAI(
    # model="gpt-4o",
    model="gpt-3.5-turbo",
    temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
) # チャット特化型モデル
# memory_key = "chat_history"
memory = ConversationBufferMemory(memory_key="chat_history", ai_prefix="") # , return_messages=True)

template = """あなたは人間と話すチャットボットです。ユーザーの要求に答えてください。

### Chat History
{chat_history}
### Input
Human: {input}
AI: 
"""

# ### 以前のチャット履歴
# {chat_history}
# Human: {input}
# Chatbot:


prompt = PromptTemplate(
    input_variables=["chat_history", "input"], 
    template=template,
    # memory=memory
)
# llm_chain = LLMChain(
#     llm=llm, # OpenAI(), # model_name="gpt-3.5-turbo"),  # ChatOpenAIにすることで解決
#     prompt=prompt, 
#     verbose=True, 
#     memory=memory,
# )

# llm_chain.predict(human_input="AIとは何？")


from langchain_core.output_parsers import StrOutputParser

# chain = prompt | llm | StrOutputParser() # こっちの方が新しいが、メモリーを渡せない？？ # 以下のように配列などで渡す必要がある？
chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=True,
    )

# this
InputText = "AIとは何？"
# res = chain.invoke({"chat_history": "", "input": InputText}) # 新しいchainのやり方(chat_historyは毎回渡さないとだめかも)
# print(res)
res = chain.invoke({"input": InputText})
print(res['text'])
InputText = "具体例を挙げて。"
res = chain.invoke({"input": InputText})
print(res['text'])


##########
# chat history の中身
# → 結局配列の格納して全部渡しているだけ

# 一回目の出力
"""
'AI（人工知能）とは、コンピューターシステムが人間の知能を模倣する技術のことです。AIは様々な分野で活用されており、自動運転車、音声認識、画像認識、自然言語処理などの技術がAIの一部です。AIは機械学習やディープラーニングなどの技術を使って、データからパターンを学習し、問題を解決する能力を持っています。'
"""
# 二回目の入力時に渡された会話履歴
"""
'chat_history': 'Human: AIとは何？\n: AI（人工知能）とは、コンピューターシステムが人間の知能を模倣する技術のことです。AIは様々な分野で活用されており、自動運転車、音声認識、画像認識、自然言語処理などの技術がAIの一部です。AIは機械学習やディープラーニングなどの技術を使って、データからパターンを学習し、問題を解決する能力を持っています。'
"""
##########