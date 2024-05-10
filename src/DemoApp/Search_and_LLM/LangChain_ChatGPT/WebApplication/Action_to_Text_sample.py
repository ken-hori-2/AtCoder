from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
llm = ChatOpenAI()
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
    "あなたはユーザーの行動からユーザーが何をしているのかを推測してサポートする専門家です。\
    ユーザーの行動の羅列が文字列で入力されるので、複数の行動情報から何をしているか推測しなければなりません。\
    また、何を求めているか考えてこたえなければならない。\
    その際、あなたが提供できる機能の中からユーザーの潜在的ニーズを満たす提案を1つ選んで回答しなければならない。\
    あなたが提案できる機能は「予定管理、楽曲再生、経路検索、ウェブ検索、レストラン検索、ニュース情報、天気情報」です。\
    また、日本語で回答しなければならない。\
    ###\
    出力例\
    提案する機能: ここに機能名を出力\
    "),
    
    ("user", "{input}")
])
# prompt.invoke({"input": "Qiitaとは何ですか？"})

# # この質問にはちゃんと答えてくれるが、
# response_from_prompt = prompt.invoke({"input": "Qiitaとは何ですか?"})
# print(llm.invoke(response_from_prompt))

# # こちらには答えてくれない
# response_from_prompt = prompt.invoke({"input": "明日の天気は何ですか?"})
# print(llm.invoke(response_from_prompt))

# chain = prompt | llm 
# res = chain.invoke({"input": "Qiitaとは何ですか？"}) # これだけでOK
# print(res)

from langchain_core.output_parsers import StrOutputParser
# chain = prompt | llm | StrOutputParser()
# res = chain.invoke({"input": "Qiitaとは何ですか"})

chain = prompt | llm | StrOutputParser()
# res = chain.invoke({"input": "Running was detected 10 times, walking 10 times, and running 10 times."})

time = "07:00"
state = "ダッシュ" # RUNNING" ランニングにすると、急いでいると判断されない。→よく運動する時間夕方や夜の時間帯なら"RUNNNING"、出勤する朝方なら"ダッシュ"と表現する
conti = "しばらく"
# location = "station"
location1 = "家"
location2 = "駅"
# 別途ユーザーの日々の行動の要約をインプットさせたLLMを作る？→そこに現在時刻を入力して照らし合わせてもらう＝今は何をしている可能性が高い時刻か
schedule = "その時刻は出勤していることが多い" # 別途LLMにその時刻は普段何をしていることが多いか教えてもらう
# 他にもユーザーは朝はよく急いでいるとかの情報があってもいいかも

# 以下のように今日の予定は～～というように設定してもいいかも（Outlook等で設定するから）
# pre_Info = "平日の朝は'天気情報', '経路検索', '楽曲再生'をよく使います。\
#             平日の昼は'予定管理', '楽曲再生', ニュース情報'をよく使います。\
#             平日の夜は'楽曲再生'をよく使います。\
#             休日は'レストラン検索','予定管理',楽曲再生'をよく使います。\
#             ランニング中や通勤中は楽曲再生をよく使います。"

# pre_Info = "ランニング中や通勤中は楽曲再生を使います。\
#             ジムに行く際はレストラン検索を使います。\
#             出勤する際や帰宅する際の移動中は経路検索を使います。\
#             昼間の会議などの合間には予定の確認のために予定管理を使います。"
pre_Info = "7:00~9:00の出勤中や、18:00~20:00ごろの帰宅中には電車や公共交通機関を使った移動をしたいので、経路検索を使います。\
            9:30~17:30には予定の確認のために予定管理を使います。\
            20:00以降にジムに行き、トレーニングをする際は楽曲再生を使います。\
            "


date = "平日"
# date = "休日"
time = "12:00"
state = "Stable"
conti = "しばらく"
location1 = "会社"
location2 = "会社"




from sample2 import Outlook
# schedule = Outlook() # .run()

# outlook_schedule_result = schedule.run()
import datetime
schedule = Outlook()
res = schedule.run()
# 一旦ダミーデータ
res_dummy = "\
    - 件名：ランニング\
    場所：屋外\
    開始時刻：2024/05/08 10:30\
    終了時刻：2024/05/08 11:00\
    \
    - 件名：通勤\
    場所：公共交通機関\
    開始時刻：2024/05/08 11:00\
    終了時刻：2024/05/08 12:30\
    \
    - 件名：Eさんの資料のレビュー\
    場所：会議室TeamsE\
    開始時刻：2024/05/08 12:00\
    \
    - 件名：Fチーム定例会\
    場所：会議室TeamsF\
    開始時刻：2024/05/08 14:00\
    \
    - 件名：トレーニング\
    場所：ジム\
    開始時刻：2024/05/08 15:00"

    # 今日の予定の件数: 8
    # 件名： 出勤
    # 場所： 電車
    # 開始時刻： 2024/05/09 07:00
    # 終了時刻： 2024/05/09 08:00
    # ----
    # 件名： 定例会議1
    # 場所： A1会議室
    # 開始時刻： 2024/05/09 08:30
    # 終了時刻： 2024/05/09 09:00
    # ----
    # 件名： 昼食
    # 場所：
    # 開始時刻： 2024/05/09 12:00
    # 終了時刻： 2024/05/09 13:00
    # ----
    # 件名： [AUD外販] 顧客定例
    # 場所： teams
    # 開始時刻： 2024/05/09 13:00
    # 終了時刻： 2024/05/09 13:55
    # ----
    # 件名： [BLANC 外販]検証OJT
    # 場所： Microsoft Teams Meeting
    # 開始時刻： 2024/05/09 14:00
    # 終了時刻： 2024/05/09 15:00
    # ----
    # 件名： 顧客定例
    # 場所： B1会議室
    # 開始時刻： 2024/05/09 17:00
    # 終了時刻： 2024/05/09 17:30
    # ----
    # 件名： 帰宅
    # 場所： 電車
    # 開始時刻： 2024/05/09 19:00
    # 終了時刻： 2024/05/09 20:00
    # ----
    # 件名： トレーニング
    # 場所： ジム
    # 開始時刻： 2024/05/09 20:00
    # 終了時刻： 2024/05/09 21:00
    # ----

# res = res_dummy # 上記のダミーデータを使う場合

conversationHistory = []
user_action = {"role": "user", "content": res} # text}
conversationHistory.append(user_action)
time = datetime.datetime.now()
time = time.strftime('%H:%M:%S')

time = '7:00:00' # 提案機能：経路検索
time = '7:30:00' # 提案機能：経路検索
time = '8:00:00' # 提案機能：経路検索
time = '9:00:00' # 提案機能：経路検索
time = '9:30:00' # 提案機能：経路検索
time = '10:00:00' # 提案機能：経路検索
time = '11:00:00' # 提案機能：経路検索

pre_Info_2 = "現在時刻は" + time + "です。\n"
# LLMに入力
text = pre_Info_2 + "この後の予定は何ですか？何分後にどこに向かえばいいですか？"
text = pre_Info_2 + "今日の予定のうち、現在は何をする時間ですか？"
# text = pre_Info + "この後の予定は何ですか？"





# ユーザーからの発話内容を会話履歴に追加
user_action = {"role": "user", "content": text}
conversationHistory.append(user_action)


# APIリクエストを作成する
print("[ChatGPT]") #応答内容をコンソール出力
res = schedule.chat(conversationHistory)

"""
1.
"""
res = chain.invoke({"input": f"事前情報は{res}です。よく使う機能は{pre_Info}です。提案する機能を選定した理由も教えてください。"})
print(res)

# # res = chain.invoke({f"input": "ユーザーは普段{time}時頃は何をすることが多い傾向ですか？"})
# # res = chain.invoke({"input": f"{time}時頃に{state}の状態が{conti}の間続いています。現在位置は{location}付近です。過去の傾向だと「{res}」をしている可能性が高いことを踏まえて回答してください。"})
# # res = chain.invoke({"input": f"現在は{time}時で、{state}の状態が{conti}の間続いています。過去の傾向だと「{schedule}」をしている可能性が高いことを踏まえて回答してください。"})
# # res = chain.invoke({"input": f"現在は{time}時で、{state}の状態が{conti}の間続いています。現在位置は{location}付近です。"})






"""
2.
"""
# res = chain.invoke({"input": f"事前情報は{pre_Info}です。今日は{date}で、現在{time}時です。{state}の状態が{conti}の間続いています。行動の始まりの場所は{location1}で、現在位置は{location2}付近です。"}) # 過去の傾向だと「{schedule}」をしている可能性が高いことを踏まえて回答してください。"})
# print(res)

# # chain = prompt | llm
# # response = chain.invoke({"input": "Qiitaとは何ですか"})
# # response.content

# print("test")
# # from langchain.llms import OpenAI
# # from langchain.callbacks import get_openai_callback

# # llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2)

# # with get_openai_callback() as cb:
# #     result = llm("Tell me a joke")
# #     print(cb)
