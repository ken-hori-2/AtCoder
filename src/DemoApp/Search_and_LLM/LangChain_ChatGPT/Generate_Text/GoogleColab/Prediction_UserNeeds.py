import os
from dotenv import load_dotenv
load_dotenv()

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import SimpleMemory
from langchain.memory import ConversationBufferMemory
"""
WARNING
"""
# LangChainDeprecationWarning: The class `LLMChain` was deprecated in LangChain 0.1.17 and will be removed in 0.3.0. Use RunnableSequence, e.g., `prompt | llm` instead.

from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
llm=ChatOpenAI(
    model="gpt-4o",
    # model="gpt-3.5-turbo",
    temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
) # チャット特化型モデル
# memory_key = "chat_history"
memory = ConversationBufferMemory(memory_key="chat_history", ai_prefix="") # , return_messages=True)

"""
Input Textに全部含める場合
"""
# template = """あなたは人間と話すチャットボットです。ユーザーの要求に答えてください。

# ### Chat History
# {chat_history}
# ### Input
# Human: {input}
# AI: 
# """

# prompt = PromptTemplate(
#     input_variables=["chat_history", "input"], 
#     template=template,
# )
# from langchain_core.output_parsers import StrOutputParser
# # chain = prompt | llm | StrOutputParser() # こっちの方が新しいが、メモリーを渡せない？？ # 以下のように配列などで渡す必要がある？
# chain = LLMChain(
#         llm=llm,
#         prompt=prompt,
#         memory=memory,
#         verbose=True,
#     )

# # > Finished chain.
# # ### ユーザーの傾向
# # 1. **時間帯ごとの行動パターン**:
# #    - **朝 (7:30 - 9:00)**: 天気情報、経路検索、楽曲再生、会議情報といった情報収集や準備活動が多い。
# #    - **昼 (11:00 - 12:30)**: 会議情報やレストラン検索、楽曲再生といった活動が見られる。
# #    - **午後 (15:00 - 17:30)**: 会議情報や経路検索といった業務関連の活動が多い。
# #    - **夕方から夜 (18:00 - 19:30)**: 楽曲再生が多く、リラックスや運動中に音楽を聴く傾向がある。
# # 2. **行動状態**:
# #    - **STABLE (安定時)**: 情報収集や楽曲再生が多い。
# #    - **WALKING (歩行時)**: 経路検索や会議情報の確認が多い。
# #    - **RUNNING (ランニング時)**: 楽曲再生が多い。


# # 2024/05/21
# # とりあえずベクターストアに入れなくてもいいかも
# UserAction = "STABLE"
# # UserAction = "WALKING"
# # UserAction = "RUNNING"
# InputText = """
#             以下はユーザーの1日の行動と使った機能です。
#             このユーザーの傾向を述べてください。(例: 1. 時間帯ごとの行動パターン: , 2. 行動状態: )
#             その後、現在が11時30分、ユーザーの行動状態が{UserAction}の場合どの機能を提案するか教えてください。その際、各機能の提案する確率も教えてください。

#             7:30, STABLE, 天気情報
#             8:00, WALKING, 経路検索
#             8:30, STABLE, 楽曲再生
#             9:00, STABLE, 会議情報
#             11:00, WALKING, 会議情報
#             12:00, STABLE, レストラン検索
#             12:30, STABLE, 楽曲再生
#             15:00, WALKING, 会議情報
#             17:30, STABLE, 経路検索
#             18:00, STABLE, 楽曲再生
#             19:00, RUNNING, 楽曲再生
#             19:30, RUNNING, 楽曲再生
#             """
# res = chain.invoke({"input": InputText})
# print(res['text'])



"""
テンプレート化する場合
"""
# テンプレート化
prompt_1 = PromptTemplate(
    # input_variables=["input", "time", "UserAction"], # , "chat_history"],
    input_variables=["time", "UserAction"],
    # template="{adjective}{job}に一番オススメのプログラミング言語は?\nプログラミング言語：",
    template = """
               あなたは人間と話すチャットボットです。ユーザーの要求に答えてください。

               以下はユーザーの1日の行動と使った機能です。
               このユーザーの傾向を述べてください。(例: 1. 時間帯ごとの行動パターン: , 2. 行動状態: )
               その後、現在が{time}、ユーザーの行動状態が{UserAction}の場合どの機能を提案するか教えてください。
               その際、各機能の提案する確率と最終的な提案(Final Answer:)も教えてください。

               7:30, STABLE, 天気情報
               8:00, WALKING, 経路検索
               8:30, STABLE, 楽曲再生
               9:00, STABLE, 会議情報
               11:00, WALKING, 会議情報
               12:00, STABLE, レストラン検索
               12:30, STABLE, 楽曲再生
               15:00, WALKING, 会議情報
               17:30, STABLE, 経路検索
               18:00, STABLE, 楽曲再生
               19:00, RUNNING, 楽曲再生
               19:30, RUNNING, 楽曲再生



               あなたが提案できる機能は
               "会議情報", "楽曲再生", "経路検索", "リアルタイム情報検索", "レストラン検索", "ニュース情報", "天気情報"
               です。
               """
            #    ### 出力形式
            #    Answer: 提案機能
            #    """

            #    """
            #    あなたはユーザーの行動からユーザーが何をしているのかを推測してサポートする専門家です。\
            #    ユーザーの予定と行動の情報が文字列で入力されるので、その情報からユーザーが何をしているか推測し、何を求めているか考えてこたえなければならない。\
            #    その際、あなたが提供できる機能の中からユーザーの潜在的ニーズを満たす提案を1つ選んで回答しなければならない。\
            #    あなたが提案できる機能は
            #    ["会議情報", "楽曲再生", "経路検索", "リアルタイム情報検索", "レストラン検索", "ニュース情報", "天気情報"]
            #    です。\
            #    また、日本語で回答しなければならない。\
            #    ### 出力形式は以下\n\
            #    Question: {question}
            #    Answer: 提案機能
            #    """

            #    ### Chat History
            #    {chat_history}
            #    ### Input
            #    Human: {input}
            #    AI: 
            #    """
)

# chain_1 = LLMChain(llm=llm, prompt=prompt_1)
chain_1 = LLMChain(llm=llm, prompt=prompt_1, output_key="response") # あくまで辞書型のなんていう要素に出力が格納されるかの変数

# prompt_2 = PromptTemplate(
#     input_variables=["programming_language"],
#     template="{programming_language}を学ぶためにやるべきことを3ステップで100文字で教えて。",
# )
# chain_2 = LLMChain(llm=llm, prompt=prompt_2, output_key="learning_step")

# overall_chain = SequentialChain(
#     chains=[chain_1, chain_2], 
#     input_variables=["input", "time", "UserAction"],
#     output_variables=["programming_language", "learning_step"],
#     verbose=True,
# )


# """
# 一旦コメントアウト
# ユーザーの過去の行動を入力せずにやってみる(2024/05/23)
overall_chain = SequentialChain(
    chains=[chain_1], # , chain_2], 
    # input_variables=["input", "time", "UserAction"],
    input_variables=["time", "UserAction"],
    output_variables=["response"], # あくまで辞書型のなんていう要素に出力が格納されるかの変数
    # output_variables=["programming_language", "learning_step"],
    verbose=True,
)
# time = "11時30分"
# UserAction = "STABLE"
output = overall_chain({
    # "time" : "11時30分",
    # "time" : "12時05分",
    "time" : "9時10分",
    # "UserAction" : "STABLE",
    "UserAction" : "WALKING",
    # "UserAction" : "RUNNING",
})
# print(output['text'])
# print(output)
print(output['response'])
# """






# ###################
# # 2024/05/22 追記 #
# ###################
# # 予定を与えずに行動がどれくらい続いているかを教える
# prompt_2 = PromptTemplate(
#     input_variables=["date", "time", "UserAction", "conti"],
#     # template = """
#     #            あなたは人間と話すチャットボットです。ユーザーの要求に答えてください。
#     #            以下はユーザーのある一定時間の行動です。  
#     #            今日は{date}で、現在{time}時です。{UserAction}の状態が{conti}の間続いています。
#     #            ユーザーの現在状況を推測し、ユーザーのニーズに合った機能を提案して。
#     #            その際、各機能の提案する確率と最終的な提案(Final Answer:)も教えてください。
#     #            あなたが提案できる機能は
#     #            "会議情報", "楽曲再生", "経路検索", "リアルタイム情報検索", "レストラン検索", "ニュース情報", "天気情報"
#     #            です。
#     #            """

#     # 現在{time}時です。
#     template = """
#                あなたは人間と話すチャットボットです。ユーザーの要求に答えてください。
#                以下はユーザーのある一定時間の間の行動です。
#                現在{time}時です。{UserAction}の状態が{conti}の間続いています。
#                ユーザーの現在状況を推測し、ユーザーのニーズに合った機能を提案して。
#                その際、各機能の提案する確率と最終的な提案(Final Answer:)も教えてください。
#                あなたが提案できる機能は
#                "会議情報", "楽曲再生", "経路検索", "リアルタイム情報検索", "レストラン検索", "ニュース情報", "天気情報"
#                です。
#                """
# )
# chain_2 = LLMChain(llm=llm, prompt=prompt_2, output_key="response")
# overall_chain = SequentialChain(
#     chains=[chain_2], # , chain_1], 
#     # input_variables=["input", "time", "UserAction"],
#     input_variables=["date", "time", "UserAction", "conti"],
#     output_variables=["response"], # あくまで辞書型のなんていう要素に出力が格納されるかの変数
#     verbose=True,
# )
# output = overall_chain({
#     "date" : "平日",
#     "date" : "休日",
#     "time" : "11時30分",
#     # "time" : "12時05分",
#     # "time" : "9時10分",
#     # "UserAction" : "STABLE",
#     "UserAction" : "WALKING",
#     # "UserAction" : "RUNNING",

#     # 継続度はあまり意味ないかも
#     "conti" : "しばらく",
#     # "conti" : "10分",
#     # "conti" : "10秒",

#     # "location1" : "会社",
#     # "location2" : "会社",
# })
# print(output['response'])
# ###################
# # 2024/05/22 追記 #
# ###################