



# Decision Making For Langchain(LLM)




import sys
import pprint
import os
from dotenv import load_dotenv
# .envファイルの内容を読み込見込む
load_dotenv('Search_and_LLM\\LangChain_ChatGPT\\WebAPI\\Secret\\.env')
# os.environを用いて環境変数を表示させます
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '\\Search_and_LLM\\LangChain_ChatGPT\\WebAPI'))
# sys.path.append(os.path.join(os.path.dirname(__file__), '.\\Search_and_LLM\\LangChain_ChatGPT\\WebAPI'))
# print(sys.path)
# sys.path.append(os.path.join(os.path.dirname(__file__), 'Search_and_LLM\\LangChain_ChatGPT\\WebAPI'))
from Search_and_LLM.LangChain_ChatGPT.JudgeModel import Langchain4Judge
from Schedule import Outlook
from TriggerByEdgeAI import Trigger
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import datetime



# DNN検出(ユーザーの瞬間的な行動) × DNN検出された時間帯(現在時刻) × Outlookの予定(ユーザーのタスク) を組み合わせてLLMに機能を決定してもらうコード



# 現在時刻
dt_now_str = datetime.datetime.now()
time = dt_now_str.strftime('%H%M') # xx時xx分:0815
time_real_version = dt_now_str.strftime('%Y/%m/%d %H:%M')

"""
ダミーデータ
"""
date = dt_now_str.strftime('%Y/%m/%d')

# time = "1055" # 出社中デモ(会議直前)
# time = "900" # 通勤中デモ
# time_real_version = f'{date} 10:55' # 出社中デモ(会議直前)
# time_real_version = f'{date} 9:00' # 通勤中デモ

# ここだけ変える
real_time = '10:55' # 出社中デモ(会議直前)
real_time = '9:00' # 通勤中デモ
# real_time = '17:30' # 通勤中デモ
# real_time = '12:00' # 昼食中デモ

time = real_time.replace(':', '')
time_real_version = f'{date} {real_time}'

# 通勤中 & 始業前 -> 「STABLEなら電車内か職場についてリラックス中」と仮定
# 通勤中 & 始業前 -> 「RUNNING, WALKINGなら移動中で経路情報を得たい」と仮定
print("現在時刻：", time_real_version)
print("現在時刻(計算用)：", time)


class DecisionMaking():

    def __init__(self, schedule, transit_go_start, transit_go_end, transit_back_start, transit_back_end, working_start, working_end, exercise_start, exercise_end, lunch_start,lunch_end, home_location, office_location):
        self.transit_go_start = transit_go_start
        self.transit_go_end = transit_go_end
        self.transit_back_start = transit_back_start
        self.transit_back_end = transit_back_end
        self.working_start = working_start
        self.working_end = working_end
        self.exercise_start = exercise_start
        self.exersise_end = exercise_end
        self.lunch_start = lunch_start
        self.lunch_end = lunch_end
        self.home_location = home_location
        self.office_location = office_location
    
    def TravelByFootOrTrainApp(self, Action_List, trigger, agent, model): # Route guidance & music playback application
        """
        ここでargs = ActionDetectionResult

        しばらく検知モード→ある一定期間センシング
        →今はずっとだが、いずれは予定の前後数分間とか
        """
        print("***** センシング中 *****")
        # args = trigger.run()
        args = "RUNNING" # テスト用
        print("DNN検出結果：", args)
        print("***** センシング終了 *****")

        if args in Action_List:
            if ("WALKING" in args) or ("RUNNING" in args): # 動いている場合
                print("*****\n経路案内\n*****")

                final_response = 'OTHER' # PLAYBACK or OTHERは ルールベースで決める
                
                if self.transit_go_judge:
                    # とりあえず通勤のルートを設定
                    # Input = "本厚木駅から東京駅までの経路を教えてください" # 何分後に何番線の電車に乗ればいいですか？"
                    # departure = '本厚木駅'
                    # destination = '品川駅'
                    departure = self.home_location
                    destination = self.office_location
                else:
                    # とりあえず帰宅のルートを設定
                    # Input = "本厚木駅から東京駅までの経路を教えてください" # 。何分後に何番線の電車に乗ればいいですか？"
                    # departure = '品川駅'
                    # destination = '本厚木駅'
                    departure = self.office_location
                    destination = self.home_location
                
                print(f"出発地：{departure}, 目的地：{destination}")
                
                Input = f"{departure}から{destination}までの経路を教えて。" # ください" # 。何分後に何番線の電車に乗ればいいですか？"
                
                # 天気情報も追加
                # Input += f"{departure}と{destination}付近の天気情報も教えて。" # 一度に二個だとエラーになる（WeatherAPI側の引数を複数にするなど変更が必要）→一個でもエラーになる
                # Input += f"{destination}の天気も教えて。" # 付近というワードがダメかも。断定する必要がある？→これでもなさそう
                # 英語入力にする必要がある(OpenWeatherMapのToolのdescriptionに翻訳するようにしたら解決)
                # Input += f"{destination}の天気を教えて。" # "品川付近の天気を教えて。"(付近をつけると日本語入力されてエラーになる)
                # 一度に経路案内と天気を入力すると、LLMが混乱して適切な引数を入力できなくなる
                # なので、一回につき一つの命令をしてあげて、あとでマージする
                """
                LangchainにAPIを決定してもらう
                """
                print(f"Input Text は 「{Input}」 です。")
                response = agent.invoke(Input)


                # # 追加で天気情報も取得する場合
                # Input_additional = f"{destination}の天気を教えて。"
                # response_additional = agent.invoke(Input_additional)
                # response['output'] += response_additional['output']
                # 他にも追加してもいいかも
            
            else: # STABLEの場合
                
                # if "STABLE" in args: # これいらない
                    
                print("*****\n楽曲再生\n*****")
                final_response = 'PLAYBACK' # PLAYBACK or OTHERは ルールベースで決める
                # state = args
                
                # とりあえず電車内では気分を上げる曲を流す（通勤中で憂鬱と仮定）
                state = "RUNNING" # 使いまわしのため、RUNNINGを引数に指定
                """
                LangchainにAPIを決定してもらう
                """
                state = '走っている'
                Input = f"Action Detectionは{state}です。この行動にあったプレイリストをSpotifyAPIを使って再生もしくは一時停止してください。" # テンプレート化する
                print(f"Input Text は 「{Input}」 です。")
                response = agent.invoke(Input)
        else:
            print('アクションリスト内に一致するアクションはありません')

        
        
        """
        2024/05/09
        """
        # # 一旦コメントアウト（PLAYBACK or OTHERは ルールベースで決める）
        # # final_response, llm_chain = model.output(response) # LLMに考えさせる -> final_responseは上記のif文でいい
        # llm_chain = model.output() # response)

        

        if 'PLAYBACK' in final_response:
            print("\nMUSIC PLAYBACK!!!!! -> ガイダンス再生はしません。")
        else:
            llm_chain = model.output()

            print("\nOTHER!!!!!")
            """LLM 3個目"""
            state = '運動していません' # stableと認識
            question = f"Action Detectionは{state}です。この行動にあったプレイリストをSpotifyAPIを使って再生もしくは一時停止してください。" # テンプレート化する
            playback_response = agent.invoke(question)
            
            """LLM 4個目"""
            if ('{' in response['output']) or ('}' in response['output']): # カギかっこなどが文字列に含まれる場合 # ただし、全角文字のカッコには対応できない
                # templateに追加してもいいかも
                # user_input = f"次の文をリスト形式ではなく、カギかっこなどのなく、一文当たり短い箇条書きにしてください。\n {response['output']}" # カギかっこなどのない、ただの文字列のみで、再度生成して出力
                user_input = f"次の文をカギかっこなどのなく、一文当たり短い箇条書きにしてください。\n {response['output']}"
                final_response = llm_chain.predict(input=user_input)
                model.text_to_speach(final_response)
                # text_to_speach(response['output'])
            else:
                print("{{ または }} は含まれていないため、出力形式修正用のLLMは呼び出しません。")
                model.text_to_speach(response['output'])


                # print("##### response #####")
                # print(response)
    

    def WorkScheduleApp(self, Action_List, trigger, agent, model,          isLunchTime):
        text = "今日は" +  time_real_version + "です。"\
                + "本日" + time_real_version + "の予定として以下の情報を提供します。\
                "
                # あとで予定を聞くので、そのタイミングでリマインドしてください。"

        print("ここに出社中に必要な機能を追加する：次の会議時間の数分前にSTAND-UP検出 or 歩き始めた＝移動中でガイダンス")

        print("***** センシング中 *****")
        # args = trigger.run()
        args = "RUNNING" # テスト用

        print("DNN検出結果：", args)
        print("***** センシング終了 *****")



        margin = 5
        select_items = schedule.ScheduleItem() # Schedule.pyからスケジュールアイテムを受け取る
        
        """
        # 今回のやり方
        """
        MTG_Section, meeting_contents = schedule.MTG_ScheduleItem(select_items, self.working_start, self.working_end)
        # text += meeting_contents # 下に移動(Trueの時にtextに追加する)
        self.isMtgStart = schedule.isMtgStartWithin5min(MTG_Section, margin, time)
        print("is Meeting Start Within 5 min : ", self.isMtgStart)
        
        # if self.working_start - margin <= int(time) <= self.working_start + margin: # 次の会議の5分前になった場合
        if self.isMtgStart: # self.next_work_is_coming_up:

            print(f"直近{margin}分以内に始まる会議があります。")

            if args in Action_List:
                # if "STANDUP" in args: # 立ち上がったら（これも追加する）
                if ("WALKING" in args) or ("RUNNING" in args): # 次の会議場所に向かっている
                    print("*****\n次の予定ガイダンス\n*****")
                    final_response = 'OTHER'

                    """ 追加 """
                    text += meeting_contents
                    """ 追加 """

                    # time = dt_now_str.strftime('%H%M') # xx時xx分:0815
                    pre_Info = "\n現在時刻は" + time_real_version + "です。\n" # dt_now + "です。"
                    text += pre_Info + "直近の予定を教えて。何分後にどこに向かえばいい？" # text += pre_Info + "この後の予定は何ですか？何分後にどこに向かえばいいですか？" # 今日の15:00の予定は何ですか？どこに向かえばいいですか？
                    text += "直近に必要な情報のみ簡潔に教えてください。"

                    # llm_chain = model.output() # response)
                    # user_input = text
                    # response = llm_chain.predict(input=user_input)
                    # model.text_to_speach(response)
                    """ or """
                    # こっちのほうがよさそう
                    Input = text
                    print(f"Input Text は 「{Input}」 です。")
                    response = agent.invoke(Input)
                    model.text_to_speach(response['output'])
                    
                            # """LLM 4個目"""
                            # if ('{' in response['output']) or ('}' in response['output']): # カギかっこなどが文字列に含まれる場合 # ただし、全角文字のカッコには対応できない
                            #     # templateに追加してもいいかも
                            #     # user_input = f"次の文をリスト形式ではなく、カギかっこなどのなく、一文当たり短い箇条書きにしてください。\n {response['output']}" # カギかっこなどのない、ただの文字列のみで、再度生成して出力
                            #     user_input = f"次の文をカギかっこなどのなく、一文当たり短い箇条書きにしてください。\n {response['output']}"
                            #     final_response = llm_chain.predict(input=user_input)
                            #     model.text_to_speach(final_response)
                            #     # text_to_speach(response['output'])
                            # else:
                            #     print("{{ または }} は含まれていないため、出力形式修正用のLLMは呼び出しません。")
                            #     model.text_to_speach(response['output'])

                else:
                    print("STABLEです。まだ会議中だと判断したため、処理を終了します。")
        else:
            print(f"直近{margin}分以内に始まる会議はありません。")

            # 出社の時間帯で昼食中の場合（会議の方が優先としているので、まずは直近5分以内に会議がないか判定する）
            if isLunchTime:
                ret = self.RestaurantForLunchApp(Action_List, trigger, agent)
    

    def RestaurantForLunchApp(self, Action_List, trigger, agent):

        print("昼食の時間です。")


        print("***** センシング中 *****")
        # args = trigger.run()
        args = "Chewing" # Bite
        print("DNN検出：", args)

        if args in Action_List:
            if ("Chewing" in args): # 咀嚼を検知

                print("昼食の時間です。レストラン検索をします。")


    
    def GymTrainingApp(self, Action_List, trigger, agent):
        print("***** センシング中 *****")
        args = trigger.run()
        print("DNN検出：", args)

        if args in Action_List:
            if ("WALKING" in args) or ("RUNNING" in args): # ジムで動いているなら楽曲再生
                print("*****\n楽曲再生\n*****")
                final_response = 'PLAYBACK' # PLAYBACK or OTHERは ルールベースで決める
                # とりあえずジムでは気分を上げる曲を流す（通勤中で憂鬱と仮定）
                # state = "RUNNING" # 使いまわしのため、RUNNINGを引数に指定
                """
                LangchainにAPIを決定してもらう
                """

                # とりあえずLLMに投げる（ジムの時は常にアップテンポの曲でいいならここに直接SpotifyAPIを記述してもいい）
                # LLMをかませるのは、Spotifyのツールを選択させるためと、ユーザーの行動に応じてプレイリストを自動で決定するため
                
                state = '走っている'
                Input = f"Action Detectionは{state}です。この行動にあったプレイリストをSpotifyAPIを使って再生もしくは一時停止してください。" # テンプレート化する
                print(f"Input Text は 「{Input}」 です。")
                response = agent.invoke(Input)
            else:
                print("ジムでSTABLE状態です。処理を終了します。")

    def run(self):
        """
        予定表から予定のある時間帯付近になったらこのプログラムを起動する
        今は、このプログラム1回起動につき、EdgeAI側のセンシング時間はずっとだが、同じ行動が検知され続けたら約15秒(100回カウント)で現在の行動結果を返してしまう（ユーザーの行動を検知できる時間）
        
        1分後とかインターバルを空けて再度このプログラムを実行してEdgeAIによるセンシングもする？
        """
        print("\n***** RESULTS*****")
        self.transit_go_judge = False
        self.transit_back_judge = False
        self.working_judge = False
        self.lunch_judge = False
        self.exercise_judge = False
        self.through = False

        # self.next_work_is_coming_up = False
        self.isMtgStart = False

        if self.transit_go_start <= int(time) < self.transit_go_end:
            print("通勤中")
            self.transit_go_judge = True
        elif self.transit_back_start <= int(time) < self.transit_back_end:
            print("帰宅中")
            self.transit_back_judge = True
            
        elif self.working_start <= int(time) < self.working_end:
            print("出社中")
            self.working_judge = True

            # # ここに具体的なタスクを抜き出す処理を追加
            # if 次の予定の前後五分以内：
            #     self.is_5minutes_around = True

            if self.lunch_start <= int(time) < self.lunch_end:
                print("昼食中")
                self.lunch_judge = True
        

        # 出社中に含まれる
        # elif self.lunch_start <= int(time) < self.lunch_end:
        #     print("昼食中")
        #     self.lunch_judge = True


        elif self.exercise_start <= int(time) < self.exersise_end:
            print("運動中")
            self.exercise_judge = True
        else:
            print("現在時刻に予定されているタスクはありません。\n処理を終了します。")
            self.through = True


        if not self.through:
            print("################################################################################\n処理を開始します...")
            print("################################################################################")

            Action_List = ['STABLE', 'WALKING', 'RUNNING']
        
            ##################################################################################################################
            """
            Langchainの定義
            """
            model = Langchain4Judge()
            """
            EdgeAIによるトリガーの定義
            """
            trigger = Trigger()

            # ここのパスが重要 (DM4L.pyはDemoApp/にあるので、その直下のSearch_and_LLM/から指定)
            credentials_file = "Search_and_LLM\\LangChain_ChatGPT\\WebAPI\\Secret\\credentials.json"

            agent = model.run(credentials_file) # ここでAPI叩いている
            ##################################################################################################################
            
            if self.transit_go_judge or self.transit_back_judge:
                
                # 経路案内&楽曲再生アプリ(徒歩、電車で移動中)
                ret = self.TravelByFootOrTrainApp(Action_List, trigger, agent, model) # modelはガイダンスのためだけに渡しているので、置き換える
            
            elif self.working_judge:

                # 予定管理アプリ
                ret = self.WorkScheduleApp(Action_List, trigger, agent, model,          self.lunch_judge) # modelはガイダンスのためだけに渡しているので、置き換える
            
            # elif self.lunch_judge:

            #     # 予定管理アプリ
            #     ret = self.RestaurantForLunchApp(Action_List, trigger, agent, model) # modelはガイダンスのためだけに渡しているので、置き換える

            elif self.exercise_judge:

                # ジムトレーニングアプリ
                ret = self.GymTrainingApp(Action_List, trigger, agent) # , model)


            else:
                print(f"現在の予定はありません。")



if __name__ == "__main__":

    schedule = Outlook()
    
    # あまりschedule.run()はやらないようにする（上限回数に達してAPI呼び出せなくなる）
    transit_go_start, transit_go_end, transit_back_start, transit_back_end, working_start, working_end, exercise_start, exercise_end, lunch_start, lunch_end, home_location, office_location = schedule.run()
    
    functional_decision = DecisionMaking(schedule, transit_go_start, transit_go_end, transit_back_start, transit_back_end, working_start, working_end, exercise_start, exercise_end, lunch_start, lunch_end, home_location, office_location)

    
    
    
    
    
    """
    予定表から予定のある時間帯付近になったらこのプログラムを起動する
    今は、このプログラム1回起動につき、EdgeAI側のセンシング時間はずっとだが、
    同じ行動が検知され続けたら約15秒(100回カウント)で現在の行動結果を返してしまう（ユーザーの行動を検知できる時間に制限がある）
    
    1分後とかインターバルを空けて再度このプログラムを実行してEdgeAIによるセンシングもする？
    （予定の変わり目の10分の間でやる？
      →デモ用には常時センシングでいいかも
      →ただ、LLMなどAPIをたたくのは少なめにしたいので、条件にあったらLLM起動する？）→済
    """
    for i in range(1): # 3): # 今は3回ユーザーにフィードバックしたら終了
        functional_decision.run()




    # パーサー用
    # print(type(response))
    # if not isinstance(response, dict):
    #     from langchain.output_parsers import StructuredOutputParser, ResponseSchema
    #     # OutputParserの準備
    #     response_schemas = [
    #         ResponseSchema(name="answer", description="ユーザーの質問に対する回答"),
    #         # ResponseSchema(name="source", description="ユーザーの質問への回答に使用されるソース。Webサイトである必要がある。")
    #     ]
    #     output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    #     # 辞書型にパース
    #     response = output_parser.parse(response)
    #     print("type:", type(response))
    #     print("response:", response)