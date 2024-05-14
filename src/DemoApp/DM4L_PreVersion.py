



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



# args = sys.argv[1]
# args = 
# args = "RUNNNING"
# print("実行時の引数：", args)
"""
ダミーデータ
"""
# args = "RUNNING" # or "WALKING" # 「RUNNING, WALKINGなら移動中で経路情報を得たい」と仮定　仕事中なら次の会議室に移動していると仮定
# args = "STABLE" # 「STABLEなら電車内か職場についてリラックス中」と仮定　仕事中なら、まだ会議中と仮定



# 現在時刻
dt_now_str = datetime.datetime.now()
time = dt_now_str.strftime('%H%M') # xx時xx分:0815
time_real_version = dt_now_str.strftime('%Y/%m/%d %H:%M')

"""
ダミーデータ
"""
# time = "815" # 8:15
# time = "1015"
# time = "1200"

# time = "900"
# time = "0957"
# time = "0955"
time = "1055"
time = "900"

# time_real_version = '2024/05/10/ 9:00' # '%Y/%m/%d %H:%M'
# time_real_version = '2024/05/10/ 9:57'
# time_real_version = '2024/05/10/ 9:55'
time_real_version = '2024/05/13/ 10:55'
time_real_version = '2024/05/13/ 9:00'

# 通勤中 & 始業前 -> 「STABLEなら電車内か職場についてリラックス中」と仮定
# 通勤中 & 始業前 -> 「RUNNING, WALKINGなら移動中で経路情報を得たい」と仮定
print("現在時刻：", time_real_version)
print("現在時刻(計算用)：", time)
# print("DNN検出：", args)


class DecisionMaking():

    def __init__(self, schedule, transit_go_start, transit_go_end, transit_back_start, transit_back_end, working_start, working_end, exercise_start, exercise_end, home_location, office_location):
        # self.time_zone = [[transit_go_start, transit_go_end], [transit_back_start, transit_back_end]]

        # self.time_zone = [[800,  930], [930,  1730], [1730, 1900], [1900, 2200]] # Outlookから引っ張ってくる
        # self.state = ["transit-go", "working", "transit-back", "training"]
        # self.sep =["start", "end"]
        # self.Section= pd.DataFrame(data=self.time_zone, index=self.state, columns=self.sep)
        # print(self.Section) # ["start"])
        # self.transit_go_start = self.Section.loc["transit-go"]["start"] # .loc[-1])
        # self.transit_go_end = self.Section.loc["transit-go"]["end"] # .loc[-1])
        # self.transit_back_start = self.Section.loc["transit-back"]["start"] # .loc[-1])
        # self.transit_back_end = self.Section.loc["transit-back"]["end"] # .loc[-1])
        self.transit_go_start = transit_go_start
        self.transit_go_end = transit_go_end
        self.transit_back_start = transit_back_start
        self.transit_back_end = transit_back_end
        self.working_start = working_start
        self.working_end = working_end
        self.exercise_start = exercise_start
        self.exersise_end = exercise_end
        self.home_location = home_location
        self.office_location = office_location



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

            # カレントディレクトリの取得のときに区切り文字を「/」に置き換え
            current_dir = os.getcwd().replace(os.sep,'\\\\')
            # credentials_file = f"{current_dir}\\Search_and_LLM\\LangChain_ChatGPT\\SecretSecret\\credentials.json" # "credentials.json"
            # credentials_file = "DemoApp\\Search_and_LLM\\LangChain_ChatGPT\\WebAPI\\Secret\\credentials.json"
            
            # ここのパスが重要 (DM4L.pyはDemoApp/にあるので、その直下のSearch_and_LLM/から指定)
            credentials_file = "Search_and_LLM\\LangChain_ChatGPT\\WebAPI\\Secret\\credentials.json"

            agent = model.run(credentials_file) # ここでAPI叩いている
            ##################################################################################################################
            
            if self.transit_go_judge or self.transit_back_judge:
                """
                ここでargs = ActionDetectionResult

                しばらく検知モード→ある一定期間センシング
                →今はずっとだが、いずれは予定の前後数分間とか
                """
                print("***** センシング中 *****")
                args = trigger.run()
                # args = "RUNNING" # テスト用
                print("DNN検出結果：", args)
                print("***** センシング終了 *****")


                # Action_List = ['STABLE', 'WALKING', 'RUNNING']

                # """
                # Langchainの定義
                # """
                # model = Langchain4Judge()
                # credentials_file = "credentials.json"
                # agent = model.run(credentials_file)

                if args in Action_List:
                    if ("WALKING" in args) or ("RUNNING" in args):
                        # if "WALKING" in args:
                        #     pass
                        # if "RUNNING" in args:
                        print("*****\n経路案内\n*****")

                        final_response = 'OTHER' # PLAYBACK or OTHERは ルールベースで決める
                        
                        if self.transit_go_judge:
                            # とりあえず通勤のルートを設定
                            # Input = "本厚木駅から東京駅までの経路を教えてください" # 。何分後に何番線の電車に乗ればいいですか？"
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
                        
                        Input = f"{departure}から{destination}までの経路を教えてください" # 。何分後に何番線の電車に乗ればいいですか？"
                        
                        """
                        LangchainにAPIを決定してもらう
                        """
                        print(f"Input Text は 「{Input}」 です。")
                        response = agent.invoke(Input)    
                    
                    else: # STABLEの場合
                        
                        if "STABLE" in args: # これいらない
                            
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
                # 一旦コメントアウト（PLAYBACK or OTHERは ルールベースで決める）
                # final_response, llm_chain = model.output(response) # LLMに考えさせる -> final_responseは上記のif文でいい
                llm_chain = model.output() # response)

                

                if 'PLAYBACK' in final_response:
                    print("\nMUSIC PLAYBACK!!!!! -> ガイダンス再生はしません。")
                else:
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
            
            elif self.working_judge:

                text = "今日は" +  time_real_version + "です。"\
                        + "本日" + time_real_version + "の予定として以下の情報を提供します。\
                        "
                        # あとで予定を聞くので、そのタイミングでリマインドしてください。"

                # # ここに具体的なタスクを抜き出す処理を追加
                # if self.is_5minutes_around: # 次の予定の前後五分以内





                print("ここに出社中に必要な機能を追加する：次の会議時間の数分前にSTAND-UP検出 or 歩き始めた＝移動中でガイダンス")

                print("***** センシング中 *****")
                args = trigger.run()
                # args = "RUNNING" # テスト用

                print("DNN検出結果：", args)
                print("***** センシング終了 *****")



                margin = 5
                # if self.working_start - margin <= int(time) <= self.working_start + margin: # 次の会議の5分前になった場合
                select_items = schedule.ScheduleItem() # Schedule.pyからスケジュールアイテムを受け取る
                
                
                
                """
                # A (これまでのやり方)
                for select_item in select_items:
                    
                    # A
                    # 「仕事」という大きな枠組みでデモする場合
                    if (int(select_item.Start.Format("%H%M")) - margin -40) <= int(time) <= (int(select_item.Start.Format("%H%M")) + margin):
                        # next_working_start = select_item.Start.Format("%H%M")
                        self.next_work_is_coming_up = True
                        print("\n直近の予定のみ入力する")

                        text += "\n件名：" + select_item.subject
                        # text += "件名：" + work[i] # 社外秘情報は伏せる
                        text += "\n場所：" + select_item.location
                        # text += "場所：" + "会議室" + room[i] # 社外秘情報は伏せる
                        text += "\n開始時刻：" + str(select_item.Start.Format("%Y/%m/%d %H:%M"))
                        text += "\n----"

                
                    B
                    # ここに具体的なタスクを抜き出す処理を追加（より現実的なシチュエーションでデモする場合）
                    if working_start < int(time) < working_end:
                        # next_working_start = select_item.Start.Format("%H%M")
                        # 社外秘情報は伏せる
                        if (not (mask_list[0] in select_item.subject)) and (not (mask_list[1] in select_item.subject)) and (not (mask_list[2] in select_item.subject)):
                            meeting_contents += "\n件名：" + select_item.subject
                            meeting_contents += "\n場所：" + select_item.location
                            meeting_contents += "\n開始時刻：" + str(select_item.Start.Format("%Y/%m/%d %H:%M"))
                            meeting_contents += "\n----"
                    # → 関数化
                """
                
                """
                # B (今回のやり方)
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
                            # text += pre_Info + "この後の予定は何ですか？何分後にどこに向かえばいいですか？" # 今日の15:00の予定は何ですか？どこに向かえばいいですか？
                            text += pre_Info + "直近の予定を教えて。何分後にどこに向かえばいい？"
                            text += "直近に必要な情報のみ簡潔に教えてください。"

                            llm_chain = model.output() # response)

                            user_input = text
                            response = llm_chain.predict(input=user_input)
                            model.text_to_speach(response)
                        else:
                            print("STABLEです。まだ会議中だと判断したため、処理を終了します。")
                else:
                    # print("直近五分以内の予定はありません。")
                    print(f"直近{margin}分以内に始まる会議はありません。")

            # else:
            #     print(f"現在の予定はありません。")
            
            
            
            elif self.exercise_judge:

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
                
                # """
                # 2024/05/09
                # """
                # # 一旦コメントアウト（PLAYBACK or OTHERは ルールベースで決める）
                # # final_response, llm_chain = model.output(response) # LLMに考えさせる -> final_responseは上記のif文でいい
                # llm_chain = model.output(response)
                # if 'PLAYBACK' in final_response:
                #     print("\nMUSIC PLAYBACK!!!!! -> ガイダンス再生はしません。")
                # else:
                #     print("\nOTHER!!!!!")
                #     """LLM 3個目"""
                #     state = '運動していません' # stableと認識
                #     question = f"Action Detectionは{state}です。この行動にあったプレイリストをSpotifyAPIを使って再生もしくは一時停止してください。" # テンプレート化する
                #     playback_response = agent.invoke(question)
                    
                #     """LLM 4個目"""
                #     # templateに追加してもいいかも
                #     user_input = f"次の文をリスト形式ではなく、カギかっこなどのなく、一文当たり短い箇条書きにしてください。\n {response['output']}" # カギかっこなどのない、ただの文字列のみで、
                #     final_response = llm_chain.predict(input=user_input)
                #     model.text_to_speach(final_response)
                #     # text_to_speach(response['output'])
            

            else:
                print(f"現在の予定はありません。")



if __name__ == "__main__":

    schedule = Outlook()
    
    # あまりschedule.run()はやらないようにする（上限回数に達してAPI呼び出せなくなる）
    transit_go_start, transit_go_end, transit_back_start, transit_back_end, working_start, working_end, exercise_start, exercise_end, home_location, office_location = schedule.run()
    
    functional_decision = DecisionMaking(schedule, transit_go_start, transit_go_end, transit_back_start, transit_back_end, working_start, working_end, exercise_start, exercise_end, home_location, office_location)

    
    
    
    
    
    """
    予定表から予定のある時間帯付近になったらこのプログラムを起動する
    今は、このプログラム1回起動につき、EdgeAI側のセンシング時間はずっとだが、
    同じ行動が検知され続けたら約15秒(100回カウント)で現在の行動結果を返してしまう（ユーザーの行動を検知できる時間に制限がある）
    
    1分後とかインターバルを空けて再度このプログラムを実行してEdgeAIによるセンシングもする？
    （予定の変わり目の10分の間でやる？
      →デモ用には常時センシングでいいかも
      →ただ、LLMなどAPIをたたくのは少なめにしたいので、条件にあったらLLM起動する？）
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