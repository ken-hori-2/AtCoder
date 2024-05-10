



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


from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import datetime



# DNN検出(ユーザーの瞬間的な行動) × DNN検出された時間帯(現在時刻) × Outlookの予定(ユーザーのタスク) を組み合わせてLLMに機能を決定してもらうコード



# args = sys.argv[1]
# args = "RUNNNING"
# print("実行時の引数：", args)
"""
ダミーデータ
"""
args = "RUNNING" # or "WALKING" # 「RUNNING, WALKINGなら移動中で経路情報を得たい」と仮定
args = "STABLE" # 「STABLEなら電車内か職場についてリラックス中」と仮定



# 現在時刻
dt_now_str = datetime.datetime.now()
time = dt_now_str.strftime('%H%M') # xx時xx分:0815
"""
ダミーデータ
"""
# time = "815" # 8:15
# time = "1015"
# time = "1200"
time = "900"

# 通勤中 & 始業前 -> 「STABLEなら電車内か職場についてリラックス中」と仮定
# 通勤中 & 始業前 -> 「RUNNING, WALKINGなら移動中で経路情報を得たい」と仮定
print("現在時刻：", time)
print("DNN検出：", args)


class DecisionMaking():

    def __init__(self, transit_go_start, transit_go_end, transit_back_start, transit_back_end, working_start, working_end, exercise_start, exercise_end, home_location, office_location):
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
        print("\n***** RESULTS*****")
        self.transit_go_judge = False
        self.transit_back_judge = False
        self.working_judge = False
        self.exercise_judge = False

        if self.transit_go_start <= int(time) < self.transit_go_end:
            print("通勤中")
            self.transit_go_judge = True
        elif self.transit_back_start <= int(time) < self.transit_back_end:
            print("帰宅中")
            self.transit_back_judge = True
            
        elif self.working_start <= int(time) < self.working_end:
            print("出社中")
            self.working_judge = True
        elif self.exercise_start <= int(time) < self.exersise_end:
            print("運動中")
            self.exercise_judge = True


        Action_List = ['STABLE', 'WALKING', 'RUNNING']
        """
        Langchainの定義
        """
        model = Langchain4Judge()





        # カレントディレクトリの取得のときに区切り文字を「/」に置き換え
        current_dir = os.getcwd().replace(os.sep,'\\\\')
        # credentials_file = f"{current_dir}\\Search_and_LLM\\LangChain_ChatGPT\\SecretSecret\\credentials.json" # "credentials.json"
        # credentials_file = "DemoApp\\Search_and_LLM\\LangChain_ChatGPT\\WebAPI\\Secret\\credentials.json"
        
        # ここのパスが重要 (DM4L.pyはDemoApp/にあるので、その直下のSearch_and_LLM/から指定)
        credentials_file = "Search_and_LLM\\LangChain_ChatGPT\\WebAPI\\Secret\\credentials.json"





        agent = model.run(credentials_file)
        
        if self.transit_go_judge or self.transit_back_judge:
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
            llm_chain = model.output(response)

            

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
        
        if self.working_judge:
            print("ここに出社中に必要な機能を追加する：次の会議時間の数分前にSTAND-UP検出でガイダンス")
        
        if self.exercise_judge:

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



if __name__ == "__main__":

    schedule = Outlook()
    transit_go_start, transit_go_end, transit_back_start, transit_back_end, working_start, working_end, exercise_start, exercise_end, home_location, office_location = schedule.run()
    
    functional_decision = DecisionMaking(transit_go_start, transit_go_end, transit_back_start, transit_back_end, working_start, working_end, exercise_start, exercise_end, home_location, office_location)

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