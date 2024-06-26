from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
# llm = ChatOpenAI()
llm=ChatOpenAI(
    # model="gpt-4o",
    model="gpt-3.5-turbo",
    temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
) # チャット特化型モデル
# print(llm.invoke("Qiitaとはなんですか"))


# promptの生成
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    # ("system", "あなたはQiitaの専門家です。Qiitaに関すること以外には分からないと返答してください。"),
    ("system", 
    #    "You are an expert in supporting users by inferring what they are doing from their actions.\
    #     You must guess what the user is doing from multiple pieces of behavioral information, because the user's list of actions is entered as a string of characters.\
    #     You also have to think about what they are asking for and respond to them.\
    #     In doing so, you must respond with one suggestion that satisfies the user's potential needs from among the functions you can offer.\
    #     The functions you can suggest are 'schedule confirmation and addition', 'song playback', 'route search', 'web search', 'restaurant search', 'news information', and 'weather information'.\
    #     You must also answer in Japanese.\
    # """
    """
    あなたはユーザーの行動からユーザーが何をしているのかを推測してサポートする専門家です。\
    ユーザーの予定と行動の情報が文字列で入力されるので、その情報からユーザーが何をしているか推測し、何を求めているか考えてこたえなければならない。\
    その際、あなたが提供できる機能の中からユーザーの潜在的ニーズを満たす提案を1つ選んで回答しなければならない。\
    あなたが提案できる機能は
    ["会議情報", "楽曲再生", "経路検索", "ウェブ検索", "レストラン検索", "ニュース情報", "天気情報"]
    です。\
    また、日本語で回答しなければならない。\
    ### 出力形式は以下\n\
    Question: {question}
    Answer: 提案機能
    """),
    
    ("user", "{input}")

    # Suggested Functions : "ここに機能名を出力"\

])


from langchain_core.output_parsers import StrOutputParser
# chain = prompt | llm | StrOutputParser()
# res = chain.invoke({"input": "Qiitaとは何ですか"})

chain = prompt | llm | StrOutputParser()
# res = chain.invoke({"input": "Running was detected 10 times, walking 10 times, and running 10 times."})


"""
2024/05/20 [Next Action]
ここ(事前情報)のテキストを生成させる
"""

# # 事前情報
# pre_Info = """
#            朝方の出勤中や、夕方の帰宅中には電車等の公共交通機関や車(乗り物)を使った移動をしたいので、経路検索を使う傾向が高いです。\
#            昼間は仕事で会議などの予定の確認のために予定管理を使う傾向にあります。\
#            帰宅した後の夜の時間はジムに行き、トレーニングをする際は楽曲再生を使うことが多いです。\
#            """ 

# # Action Detection
# pre_Info += """
#             ユーザーの現在の行動が、["STABLE"]の場合は、座っているか、会議中か、乗り物に乗って移動していることが多いです。
#             ユーザーの現在の行動が、["WALKING"]の場合は徒歩で移動中のことが多いです。
#             ユーザーの現在の行動が、["RUNNING"]の場合は、急いでどこかに移動しているか、トレーニング中のことが多いです。
#             """

# 次の予定の5分以内の時間でもユーザーの現在の行動が"STABLE"の時は会議が長引いている可能性が高いです。

# VectorStoreを使った方法
pre_Info = """
           このユーザーは朝にニュース閲覧と天気情報をチェックし、その後経路検索を行います。午後には天気情報を再度チェックし、楽曲再生や会議情報の確認を行います。午後には会議情報をチェックし、楽曲再生やレストラン検索を行います。
           """

time = '7:00:00' # 提案機能：経路検索
time = '7:30:00' # 提案機能：経路検索
time = '8:00:00' # 提案機能：経路検索
time = '9:00:00' # 提案機能：経路検索
# time = '9:30:00' # 提案機能：経路検索
# time = '10:00:00' # 提案機能：経路検索
# time = '11:00:00' # 提案機能：経路検索
time = '11:30:00' # 提案機能：経路検索
# time = '12:00:00' # 提案機能：レストラン検索


"""
1. GPT-4o機能を提案してもらう
"""
UserAction = "STABLE"
# UserAction = "WALKING"
# UserAction = "RUNNING"

# 予定情報あり
# InputText = f"事前情報は{schedule}で、現在時刻は{time}、ユーザーの現在の行動は{UserAction}です。ユーザーのニーズに合った機能を提案して。提案する機能を選定した理由も教えて。また、機能それぞれの採択される確率も出力して。"
# 時間と行動の情報のみ
InputText = f"現在時刻は{time}、ユーザーの現在の行動は{UserAction}です。ユーザーのニーズに合った機能を提案して。提案する機能を選定した理由も教えて。また、機能それぞれの採択される確率も出力して。"
InputText = f"現在時刻は{time}、{pre_Info} ユーザーの現在の行動は{UserAction}です。ユーザーのニーズに合った機能を提案して。提案する機能を選定した理由も教えて。また、機能それぞれの採択される確率も出力して。"
# 時間と行動の情報に加えてユーザーの傾向の情報を与える
# InputText = f"現在時刻は{time}、ユーザーの現在の行動は{UserAction}です。よく使う機能は{pre_Info}です。ユーザーのニーズに合った機能を提案して。提案する機能を選定した理由も教えて。また、機能それぞれの採択される確率も出力して。"


print("Input Text : ", InputText)
# res = chain.invoke({"input": f"事前情報は{res}です。よく使う機能は{pre_Info}です。提案する機能を選定した理由も教えてください。"})
# res = chain.invoke({"input": f"事前情報は{schedule}で、現在時刻は{time}です。ユーザーのニーズに合った機能を提案して。提案する機能を選定した理由も教えて。"})
res = chain.invoke({"input": InputText})
print(res)


# date = "平日"
# # date = "休日"
# time = "12:00"
# # state = "Stable"
# # state = "RUNNING"
# state = "WALLKING"
# conti = "しばらく"
# location1 = "会社"
# location2 = "会社"

# # time = "07:00"
# # state = "ダッシュ" # RUNNING" ランニングにすると、急いでいると判断されない。→よく運動する時間夕方や夜の時間帯なら"RUNNNING"、出勤する朝方なら"ダッシュ"と表現する
# # conti = "しばらく"
# # # location = "station"
# # location1 = "家"
# # location2 = "駅"
# # # 別途ユーザーの日々の行動の要約をインプットさせたLLMを作る？→そこに現在時刻を入力して照らし合わせてもらう＝今は何をしている可能性が高い時刻か
# # schedule = "その時刻は出勤していることが多い" # 別途LLMにその時刻は普段何をしていることが多いか教えてもらう
# # # 他にもユーザーは朝はよく急いでいるとかの情報があってもいいかも


# """
# 2.
# """
# # res = chain.invoke({"input": f"事前情報は{pre_Info}です。今日は{date}で、現在{time}時です。{state}の状態が{conti}の間続いています。行動の始まりの場所は{location1}で、現在位置は{location2}付近です。"}) # 過去の傾向だと「{schedule}」をしている可能性が高いことを踏まえて回答してください。"})
# # res = chain.invoke({"input": f"事前情報は{pre_Info}です。今日は{date}で、現在{time}時です。{state}の状態が{conti}の間続いています。ユーザーのニーズに合った機能を提案して。提案する機能を選定した理由も教えて。また、機能それぞれの採択される確率も出力して。"})
# # 事前情報なし
# res = chain.invoke({"input": f"今日は{date}で、現在{time}時です。{state}の状態が{conti}の間続いています。ユーザーのニーズに合った機能を提案して。提案する機能を選定した理由も教えて。また、機能それぞれの採択される確率も出力して。"})
# print(res)