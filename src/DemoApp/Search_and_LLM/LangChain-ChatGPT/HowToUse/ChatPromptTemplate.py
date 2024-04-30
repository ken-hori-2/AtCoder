from dotenv import load_dotenv
# .envファイルの内容を読み込見込む
load_dotenv()

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI


# system_template="あなたは、質問者からの質問を{language}で回答するAIです。"
# human_template="質問者：{question}"
# system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
# human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
# prompt_message_list = chat_prompt.format_prompt(language="日本語", question="ITコンサルについて30文字で教えて。").to_messages()

# print(prompt_message_list)

# chat = ChatOpenAI(model_name="gpt-3.5-turbo")
# print(chat(prompt_message_list))


# from langchain.output_parsers import CommaSeparatedListOutputParser
# from langchain.prompts import PromptTemplate
# # from langchain.llms import OpenAI
# from langchain_community.llms import OpenAI
# from langchain_community.chat_models import ChatOpenAI

# output_parser = CommaSeparatedListOutputParser()
# format_instructions = output_parser.get_format_instructions()
# prompt = PromptTemplate(
#     template=" {subject}に関する5つのリスト.\n{format_instructions}",
#     input_variables=["subject"],
#     partial_variables={"format_instructions": format_instructions}
# )

# llm = OpenAI(model_name="gpt-3.5-turbo")
# _input = prompt.format(subject="Programming Language")
# output = llm(_input)
# output_parser.parse(output)




# from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory
# # from langchain.chat_models import ChatOpenAI
# from langchain_openai import ChatOpenAI
# from langchain.chains import LLMChain
# # chain = ConversationChain(llm=ChatOpenAI(), memory=ConversationBufferMemory(return_messages=True))
# from langchain.prompts import (
#     ChatPromptTemplate, 
#     MessagesPlaceholder, 
#     SystemMessagePromptTemplate, 
#     HumanMessagePromptTemplate
# )

# # 人格を与えることで、そっけないレスポンスを軽減できる
# prompt = ChatPromptTemplate.from_messages([
#     SystemMessagePromptTemplate.from_template("あなたは優しくフレンドリーなAIです。「ねー」や「そだねー」のように、くだけた口調で話します。特にサッカーの話題は大好きなので、！を多用して話します。"),
#     MessagesPlaceholder(variable_name="history"),
#     HumanMessagePromptTemplate.from_template("{input}"),
# ])
# prompt_message_list = prompt.format_prompt(language="日本語").to_messages()
# chain = LLMChain(
#                 llm=ChatOpenAI(), 
#                 # prompt=prompt, 
#                 prompt=prompt_message_list, 
#                 memory=ConversationBufferMemory(return_messages=True)
#                 )

# response = chain.invoke("去年のワールドカップは本当に面白かったですね。")
# # response = chain.run("去年のワールドカップは本当に面白かったですね。")
# print(response['text'])
# # # response = response.split(",")
# # # conversationHistory = []
# # # conversationHistory.append(response)
# # print(type(response['text']))
# # # text_to_speach(conversationHistory)
# # text_to_speach(response['text'])




from langchain import HuggingFacePipeline, PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
# Memory: メモリ上に会話を記録する設定
memory_key = "chat_history"
memory = ConversationBufferMemory(memory_key=memory_key, ai_prefix="")
# agent の使用する LLM
llm=ChatOpenAI(
    temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
) # チャット特化型モデル


# Prompts: プロンプトを作成。会話履歴もinput_variablesとして指定する
# template = """
# You are an AI who responds to user Input.
# Please provide an answer to the human's question.
# Additonaly, you are having a conversation with a human based on past interactions.

# ### Answer Sample
# Human: Hi!
# AI: Hi, nice to meet you.

# ### Past Interactions
# {chat_history}

# ### 
# Human:{input}


# From now on, you must communicate in Japanese.

# """
template = """
You are an AI who responds to user Input.
Please provide an answer to the human's question.
Additonaly, you are having a conversation with a human based on past interactions.

From now on, you must communicate in Japanese.

### 解答例
Human: やあ！
AI: こんにちは！

### 以前のチャット履歴
{chat_history}

### 
Human:{input}
"""
prompt = PromptTemplate(template=template, input_variables=["chat_history", "input"])
# Chains: プロンプト&モデル&メモリをチェーンに登録
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True,
)
# 実行①
user_input = "What is the Japanese word for mountain？"
# response = llm_chain.predict(input=user_input)
final_response = llm_chain.predict(input=user_input)
print(final_response)
# 履歴表示
memory.load_memory_variables({})