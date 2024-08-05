from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# client = OpenAI()

# # text = "ユーザーの現在状態は「time:8:30にstatus:WALKING」をしています。ユーザーのニーズに最適な機能(activity:)を提案して。"
# text = "ユーザーの現在状態は「time:10:30, status:WALKING」です。ユーザーのニーズに最適な「activity」を提案して。"



# time = '17:30'
# time = '19:30'
# time = '8:30' # 00'
# # time = '9:10' # 30'
# # time = '10:50'
# time = '12:05'
# # time = '17:30'
# # time = '20:00'
# # time = '23:00'
# # time = '0:00'
# status = 'WALKING'
# # status = 'STABLE'
# # status = 'RUNNNING'

# # # text = f"inputは「time:{time}, status:{status}」です。inputに対して適切な「activity」に最も合うものを考え、提案して。" # さらにあなたが保持するToolの中で最も近いものを選定しTool名のみ答えて。"

# # # Tool_list = "Search, Calculator, open_weather_map, wikipedia, News-API, Route-Search, Music-Playback, Restaurant-Search, Schedule-Information, Do Nothing"

# # # text = f"inputがtime:{time}, status:{status}の場合、最適なactivity(activity:)と実行するかどうか(Yes/No)を予測して。" # + f"また、あなたが保持するToolが[{Tool_list}]の場合、どのToolがActivityに最も近いですか？" # か、ステップを踏みながら考え、理由と合わせて提案して。"
# # text = f"If input is time:{time}, status:{status}, predict the best activity (activity:)" # and which Tool would you suggest?" # whether it should be executed (execute:)." +f"If the list of Tools you hold is [{Tool_list}], which Tool would you suggest?"
# # text = f"If the input is time:{time}, status:{status}, predict the best activity (activity:) and generate a sentence describing the user's status. Then think about what you can suggest and answer."

# text = f"""
        
#         Here is what you know about the user.
#         [The current date and time is {time}, and The current user action state is {status}.]
#         Predict and generate only sentences that describe the user's situation in as much detail as possible. (e.g., The user is doing XXX and has needs like XX)
#         Also, add one most likely user request to the output.
#         """

# response = client.chat.completions.create(
# #   model="ft:gpt-3.5-turbo-1106:personal:demoapp-3p5t-model:9qYtaL0i", # Tool予測
# #   model="ft:gpt-3.5-turbo-1106:personal:demoapp-model-2:9qaUOYsT",    # Tool予測2
#   model="ft:gpt-3.5-turbo-1106:personal:generateprompt:9qcfcS6N",       # Prompt生成

#   # messages=[],
#   messages=[
#     {
#         "role": 
#             "system", 
#         "content": 
#             """You are the user's secretary. You are the expert who takes the user's potential needs and proposes solutions.
#             """
#             # You are the secretary who is close to the user and an expert in predicting the user's current situation.
#     },
#     # {"role": "user", "content": "The user's current state is “WALKING at 9:30”. Suggest the best function for the user's needs."} # ユーザーの現在状態は「8:30にWALKING」をしています。ユーザーのニーズに最適な機能を提案して。"} # 何を提案できますか？"}
#     {"role": "user", "content":text}
#   ],
#   temperature=0, # 1,
#   max_tokens=1024, # 256,
# #   top_p=1,
# #   frequency_penalty=0,
# #   presence_penalty=0
# )
# # system_fingerprint = response.system_fingerprin
# # for res in response.choices:
# #     print(res.message.content)

# print(response)




from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

time = '8:30' # 00'
time = '9:10' # 30'
# time = '10:50'
time = '12:05'
# time = '17:30'
# time = '19:30'
# time = '20:00'
# time = '23:00'
# time = '0:00'

status = 'WALKING'
# status = 'STABLE'
# status = 'RUNNNING'


# OpenAIのモデルのインスタンスを作成
llm_demo = ChatOpenAI(model_name="ft:gpt-3.5-turbo-1106:personal:generateprompt:9qcfcS6N", temperature=0) # prompt version (以前のやつ)
# プロンプトのテンプレート文章を定義
template = """
        Here is what you know about the user.
        [The current date and time is {time}, and The current user action state is {status}.]
        Predict and generate only sentences that describe the user's situation in as much detail as possible. (e.g., The user is doing XXX and has needs like XX)
        Also, add one most likely user request to the output.
"""
# # Flagも返させるように追加
# template = """
#         Here is what you know about the user.
#         [The current date and time is {time}, and The current user action state is {status}.]
#         Predict and generate only sentences that describe the user's situation in as much detail as possible. (e.g., The user is doing XXX and has needs like XX)
#         Also, add one most likely user request to the output.
        
#         Also, answer True if the user wants the Music Playback, and False otherwise.
# """

# llm = ChatOpenAI(model_name="ft:gpt-3.5-turbo-1106:personal:genpromptdetails:9r0E1Bvv", temperature=0) # より詳細なデータにしたバージョン
# template = """
#             Here is what you know about the user.
#             [The current date and time is {time}, and The current user action state is {status}.]
#             Predict and generate only sentences that describe the user's situation in as much detail as possible. (e.g., The user is doing XXX and has needs like XX)
#            """

# テンプレート文章にあるチェック対象の単語を変数化
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are the user's secretary. You are the expert who takes the user's potential needs and proposes solutions."),
    ("user", template)
])
print(f"input: The current date and time is {time}, and The current user action state is {status}.\n************************************************************")

# チャットメッセージを文字列に変換するための出力解析インスタンスを作成
output_parser = StrOutputParser()

# OpenAIのAPIにこのプロンプトを送信するためのチェーンを作成
chain1 = prompt | llm_demo | output_parser

# それぞれ単独で実行する場合
# チェーンを実行し、結果を表示
res = chain1.invoke({"time": time, "status": status})
print(res)


# 2024/8/2
print("**********")
# # response = chain.invoke({"time": time, "status": status})
# # # 回答を解析してテキスト部分とフラグ部分を分離
# lines = response.strip().split('\n')
# reason = lines[0]  # 最初の行を理由として取得
# flag_text = lines[-1].strip().lower()  # 最後の行をフラグとして取得

# # # フラグをbool型に変換
# # # is_positive = flag_text == "true"
# is_positive = "true" in flag_text

# print("lines[0]:", reason)
# print("lines[-1]:", flag_text)
# print("Playback:", is_positive)


# カスタムchainでまとめる場合
# from langchain.prompts import PromptTemplate
# llm3p5t = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) # 通常のモデル
# prompt2 = PromptTemplate(
#     input_variables=["text"],
#     # template="このテキストが楽曲再生ならTrueのみを、そうでないならFalseのみを返して。{text}"
#     # template="Return only True if this text is a song playback, otherwise return only False. {text}" # song playback
#     template="Return only True if this text is a music playback, otherwise return only False. {text}" # music playback
# )
# # どっちでも適切に回答してくれる（Fine-Tuningしていないモデルの方が安いのでそっちを使う）
# # chain2 = prompt2 | llm3p5t | output_parser
# chain2 = prompt2 | llm_demo | output_parser
# # is_playback = chain2.invoke(res) # f'{res}が楽曲再生ならTrueのみを、そうでないならFalseのみを返して。')
# # print(is_playback)

# from langchain.chains.base import Chain

# class CustomChain(Chain):
#     def __init__(self, chain1, chain2):
#         self.chain1 = chain1
#         self.chain2 = chain2
    
#     @property
#     def input_keys(self):
#         # チェーン1の入力キーを返す
#         return ["text"]

#     @property
#     def output_keys(self):
#         # チェーン2の出力キーを返す
#         return ["output"]

#     def _call(self, inputs):
#         # チェーン1を実行
#         response1 = self.chain1.invoke(inputs)
        
#         # # チェーン1の出力を解析して理由を抽出
#         # lines = response1.strip().split('\n')
#         # reason = lines[0]  # 最初の行を理由として取得
        
#         # # チェーン2を実行
#         # response2 = self.chain2.invoke({"reason": reason})
#         return response1
#         # チェーン2を実行
#         response2 = self.chain2.invoke({"text":response1}) # {"reason": reason})
        
#         # return response2
#         return {"output": response2}

# # カスタムチェーンを作成
# custom_chain = CustomChain(chain1, chain2)

# # テキストを入力として渡す
# # response = custom_chain.run({"text": text})
# final_response = custom_chain.invoke({"time": time, "status": status})

# print("カスタムチェーンの出力:", final_response)