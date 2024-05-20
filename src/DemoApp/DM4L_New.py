



# Decision Making For Langchain(LLM)




import sys
import pprint
import os
from dotenv import load_dotenv
# .envファイルの内容を読み込見込む
load_dotenv('Search_and_LLM\\LangChain_ChatGPT\\WebAPI\\Secret\\.env')

from Search_and_LLM.LangChain_ChatGPT.JudgeModel import Langchain4Judge
# from Schedule import Outlook
from Schedule_New import Outlook
from TriggerByEdgeAI import Trigger
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import datetime

# from UIModel import UserInterfaceModel # ユーザーとのやり取りをするモデル
"""
2024/05/16 次回Todo
modelから音声からテキスト変換する関数を呼ぶのではなく、置き換える
"""



# DNN検出(ユーザーの瞬間的な行動) × DNN検出された時間帯(現在時刻) × Outlookの予定(ユーザーのタスク) を組み合わせてLLMに機能を決定してもらうコード


# 通勤中 & 始業前 -> 「STABLEなら電車内か職場についてリラックス中」と仮定
# 通勤中 & 始業前 -> 「RUNNING, WALKINGなら移動中で経路情報を得たい」と仮定


# 本番環境
# dt_now = datetime.datetime.now()
# # 現在時刻を手動で設定
# dt_now = datetime.datetime(2024, 5, 16, 8, 58) # 通勤中
dt_now = datetime.datetime(2024, 5, 16, 10, 55) # Mtg
# dt_now = datetime.datetime(2024, 5, 16, 12, 00) # Lunch
# dt_now = datetime.datetime(2024, 5, 16, 13, 55) # Mtg
# # dt_now = datetime.datetime(2024, 5, 16, 13, 58) # Mtg
# dt_now = datetime.datetime(2024, 5, 16, 18, 00) # 帰宅中
# # dt_now = datetime.datetime(2024, 5, 16, 20, 00) # ジム
print("現在時刻：", dt_now)
margin = 5
margin = datetime.timedelta(minutes=margin)


class DecisionMaking():

    def __init__(self, schedule):
        self.schedule = schedule
    
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
                    departure = self.schedule.getHomeLocation()
                    destination = self.schedule.getOfficeLocation()
                else:
                    # とりあえず帰宅のルートを設定
                    departure = self.schedule.getOfficeLocation()
                    destination = self.schedule.getHomeLocation()
                
                print(f"出発地：{departure}, 目的地：{destination}")
                
                Input = f"{departure}から{destination}までの経路を教えて。" # ください" # 。何分後に何番線の電車に乗ればいいですか？"
                # Input = f"{departure}から{destination}までの経路を箇条書きで1文当たり簡潔に教えて。" # 乗換駅が含まれる場合は含めて教えて。" # 乗車駅と降車駅、# 乗換駅は明確に教えて。"
                
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
                user_input = f"次の文をカギかっこなどのなく、一文当たり短い箇条書きにしてください。\n {response['output']}"
                final_response = llm_chain.predict(input=user_input)
                model.text_to_speach(final_response)
                # text_to_speach(response['output'])
            else:
                print("{{ または }} は含まれていないため、出力形式修正用のLLMは呼び出しません。")
                model.text_to_speach(response['output'])
    

    def WorkScheduleApp(self, Action_List, trigger, agent, model,          isLunchTime):
        """
        注意！！！！！
        date(dt_now.strftime('%Y年%m月%d日%H時%M分')) と time(dt_now.strftime('%H時%M分')
        が一致する要素がないと、うまく予定を認識できずにカレンダーツールを起動してしまう
        %Y/%m/%d %H:%M じゃないと認識しにくい
        """
        text = "本日" +  str(dt_now) + "の会議の予定は以下です。"

        print("ここに出社中に必要な機能を追加する：次の会議時間の数分前にSTAND-UP検出 or 歩き始めた＝移動中でガイダンス")

        print("***** センシング中 *****")
        # args = trigger.run()
        args = "RUNNING" # テスト用

        print("DNN検出結果：", args)
        print("***** センシング終了 *****")

        """
        # 今回のやり方
        """
        self.isMtgStart = self.schedule.isMtgStartWithin5min(margin, dt_now)
        print("MtgStart:", self.isMtgStart)
        
        # 次の会議の5分前になった場合
        if self.isMtgStart:

            print(f"直近{margin}分以内に始まる会議があります。")

            if args in Action_List:
                # if "STANDUP" in args: # 立ち上がったら（これも追加する）
                if ("WALKING" in args) or ("RUNNING" in args): # 次の会議場所に向かっている
                    print("*****\n次の予定ガイダンス\n*****")
                    final_response = 'OTHER'

                    """ 追加 """
                    # text += meeting_contents
                    text += self.schedule.getMeetingContents()
                    """ 追加 """

                    time = dt_now.strftime('%Y/%m/%d %H:%M') # %Y年%m月%d日%H時%M分') # dt_now.strftime('%H時%M分') # xx時xx分:0815
                    pre_Info = "\n現在時刻は" + str(dt_now) + "です。\n" # dt_now + "です。"
                    
                    # 直近の会議の名前は教える
                    # text += pre_Info + "直近の予定は" + schedule.getNextMtg() + "です。何分後にどこに向かえばいいか教えて。"
                    text += pre_Info + "直近の予定は" + self.schedule.getNextMtg() + "です。何分後にどこに向かえばいいか教えて。"
                    text += "直近に必要な情報のみ簡潔に教えて。" # 時刻の計算は計算機で正確に算出しなければならない。"

                    Input = text
                    print(f"Input Text は 「{Input}」 です。")
                    response = agent.invoke(Input)
                    
                    # model.text_to_speach(response['output'])
                    
                    """LLM 追加"""
                    if ('{' in response['output']) or ('}' in response['output']): # カギかっこなどが文字列に含まれる場合 # ただし、全角文字のカッコには対応できない
                        llm_chain = model.output()
                        user_input = f"次の文をカギかっこなどのなく、一文当たり短い箇条書きにしてください。{response['output']}"
                        final_response = llm_chain.predict(input=user_input)
                        model.text_to_speach(final_response)
                        # text_to_speach(response['output'])
                    else:
                        print("{{ または }} は含まれていないため、出力形式修正用のLLMは呼び出しません。")
                        model.text_to_speach(response['output'])

                else:
                    print("STABLEです。まだ会議中だと判断したため、処理を終了します。")
        else:
            print(f"直近{margin}分以内に始まる会議はありません。")

            # 出社の時間帯で昼食中の場合（会議の方が優先としているので、まずは直近5分以内に会議がないか判定する）
            if isLunchTime:
                ret = self.RestaurantForLunchApp(Action_List, trigger, agent, model)
    

    def RestaurantForLunchApp(self, Action_List, trigger, agent, model):

        print("昼食の時間です。")


        print("***** センシング中 *****")
        # args = trigger.run()
        args = "Chewing" # Bite
        print("DNN検出：", args)

        # if args in Action_List: # まだ咀嚼検出はできていないのでコメントアウト
        if ("Chewing" in args): # 咀嚼を検知

                print("\n昼食の時間です。レストラン検索をします。")

                place = self.schedule.getOfficeLocation() # "本厚木駅"
                keyword = "お肉" # "ラーメン屋"
                # ジェスチャーでキーワードを決める？
                # 何食べたいか？(What Do You Want To Be ?)を音声入力で求めるのもありかも


                Input = f"{place}という場所周辺の{keyword}という条件に近いお店を3つ教えて。お店の評価も併せて教えて。" # お店を教えて。" # Hotpepper

                print(f"Input Text は 「{Input}」 です。")
                response = agent.invoke(Input)

                
                """LLM 追加"""
                if ('{' in response['output']) or ('}' in response['output']): # カギかっこなどが文字列に含まれる場合 # ただし、全角文字のカッコには対応できない
                    llm_chain = model.output()
                    user_input = f"次の文をカギかっこなどのなく、一文当たり短い箇条書きにして。最後に以上ですと言って。\n{response['output']}\n"
                    final_response = llm_chain.predict(input=user_input)
                    model.text_to_speach(final_response)
                    # text_to_speach(response['output'])
                else:
                    print("{{ または }} は含まれていないため、出力形式修正用のLLMは呼び出しません。")
                    model.text_to_speach(response['output'])

                # model.text_to_speach(response['output']) # text_to_speachしか使っていない


    
    def GymTrainingApp(self, Action_List, trigger, agent):
        print("***** センシング中 *****")
        # args = trigger.run()
        args = "RUNNING" # テスト用
        print("DNN検出：", args)

        if args in Action_List:
            if ("WALKING" in args) or ("RUNNING" in args): # ジムで動いているなら楽曲再生
                print("*****\n楽曲再生\n*****")
                final_response = 'PLAYBACK' # PLAYBACK or OTHERは ルールベースで決める
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
        self.isMtgStart = False
        isTransitingGo = self.schedule.isTransitingGo(margin, dt_now)
        # print("TransitingGo:", isTransitingGo)
        isTransitingBack = self.schedule.isTransitingBack(margin, dt_now)
        # print("TransitingBack:", isTransitingBack)
        isWorking = self.schedule.isWorking(margin, dt_now)
        # print("Working:", isWorking)
        isExercising = self.schedule.isExercising(margin, dt_now)
        # print("isExercising:", isExercising)

        if isTransitingGo:
            print("通勤中")
            self.transit_go_judge = True
        elif isTransitingBack:
            print("帰宅中")
            self.transit_back_judge = True
        elif isWorking:
            print("出社中")
            self.working_judge = True
            isLunchStart = self.schedule.isLunchStartWithin5min(margin, dt_now)
            print("LunchStart:", isLunchStart)
            if isLunchStart:
                print("昼食中")
                self.lunch_judge = True
        elif isExercising:
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
            elif self.exercise_judge:
                # ジムトレーニングアプリ
                ret = self.GymTrainingApp(Action_List, trigger, agent) # , model)
            else:
                print(f"現在の予定はありません。")
    




    def test(self, userInput):
        # デモ動画用
        ##################################################################################################################
        """
        Langchainの定義
        """
        model = Langchain4Judge()
        # """
        # EdgeAIによるトリガーの定義
        # """
        # trigger = Trigger()
        # ここのパスが重要 (DM4L.pyはDemoApp/にあるので、その直下のSearch_and_LLM/から指定)
        credentials_file = "Search_and_LLM\\LangChain_ChatGPT\\WebAPI\\Secret\\credentials.json"
        agent = model.run(credentials_file) # ここでAPI叩いている
        ##################################################################################################################
        # 予定管理アプリ
        text = "本日" +  str(dt_now) + "の会議の予定は以下です。"
        text += self.schedule.getMeetingContents()
        """ 追加 """

        # time = dt_now.strftime('%Y/%m/%d %H:%M') # %Y年%m月%d日%H時%M分') # dt_now.strftime('%H時%M分') # xx時xx分:0815
        pre_Info = "\n現在時刻は" + str(dt_now) + "です。\n" # dt_now + "です。"
        
        # 直近の会議の名前は教える
        # text += pre_Info + "直近の予定は" + self.schedule.getNextMtg() + "です。何分後にどこに向かえばいいか教えて。"
        # text += "直近に必要な情報のみ簡潔に教えて。" # 時刻の計算は計算機で正確に算出しなければならない。"
        text += userInput

        Input = text
        # print(f"Input Text は 「{Input}」 です。")
        print("\n\n******************** [AI Answer] ********************\n")
        response = agent.invoke(Input)

        # model.text_to_speach(response['output'])
        """LLM 追加"""
        if ('{' in response['output']) or ('}' in response['output']): # カギかっこなどが文字列に含まれる場合 # ただし、全角文字のカッコには対応できない
            llm_chain = model.output()
            user_input = f"次の文をカギかっこなどのなく、一文当たり短い箇条書きにして。最後に以上ですと言って。\n{response['output']}\n"
            final_response = llm_chain.predict(input=user_input)
            model.text_to_speach(final_response)
            # text_to_speach(response['output'])
        else:
            print("{{ または }} は含まれていないため、出力形式修正用のLLMは呼び出しません。")
            model.text_to_speach(response['output'])



if __name__ == "__main__":

    schedule = Outlook()
    
    # あまりschedule.run()はやらないようにする（上限回数に達してAPI呼び出せなくなる）
    # transit_go_start, transit_go_end, transit_back_start, transit_back_end, working_start, working_end, exercise_start, exercise_end, lunch_start, lunch_end, home_location, office_location = schedule.run()
    schedule.run()
    
    # functional_decision = DecisionMaking(schedule, transit_go_start, transit_go_end, transit_back_start, transit_back_end, working_start, working_end, exercise_start, exercise_end, lunch_start, lunch_end, home_location, office_location)
    functional_decision = DecisionMaking(schedule)

    
    
    
    
    
    # """
    # 予定表から予定のある時間帯付近になったらこのプログラムを起動する
    # 今は、このプログラム1回起動につき、EdgeAI側のセンシング時間はずっとだが、
    # 同じ行動が検知され続けたら約15秒(100回カウント)で現在の行動結果を返してしまう（ユーザーの行動を検知できる時間に制限がある）
    
    # 1分後とかインターバルを空けて再度このプログラムを実行してEdgeAIによるセンシングもする？
    # （予定の変わり目の10分の間でやる？
    #   →デモ用には常時センシングでいいかも
    #   →ただ、LLMなどAPIをたたくのは少なめにしたいので、条件にあったらLLM起動する？）→済
    # """
    
    # this
    for i in range(1): # 3): # 今は3回ユーザーにフィードバックしたら終了
        functional_decision.run()

    
    
    # デモ用
    # from UIModel import UserInterfaceModel # ユーザーとのやり取りをするモデル
    # userinterface = UserInterfaceModel()
    # # 音声認識関数の呼び出し
    # print(" >> Waiting for response from Agent...")
    # # userInput = userinterface.recognize_speech()
    # userInput = "この後の予定は何？何分後にどこに向かえばいいか教えて。"

    # print("\n\n******************** [User Input] ********************\n", userInput)
    # try:
    #     # response = 
    #     functional_decision.test(userInput)

    #     print("\n\n******************** [AI Answer] ********************\n")
    #     # userinterface.text_to_speach(response['output'])
    #     # print("\n******************** [AI Answer] ********************\n", response["output"])
    # except:
    #     print("\n##################################################\nERROR! ERROR! ERROR!\n##################################################")
    #     print("もう一度入力してください。")