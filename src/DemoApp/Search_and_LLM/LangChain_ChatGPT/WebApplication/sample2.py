import win32com.client
import datetime
import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
import pyttsx3

from dotenv import load_dotenv
load_dotenv()

##################
# Pyttsx3を初期化 #
##################
engine = pyttsx3.init()
# 読み上げの速度を設定する
rate = engine.getProperty('rate')
engine.setProperty('rate', rate)
# #rate デフォルト値は200
# rate = engine.getProperty('rate')
# engine.setProperty('rate',200)

#volume デフォルト値は1.0、設定は0.0~1.0
volume = engine.getProperty('volume')
engine.setProperty('volume',1.0)


# Kyokoさんに喋ってもらう(日本語)
engine.setProperty('voice', "com.apple.ttsbundle.Kyoko-premium")

class Outlook():
    
    def run(self):
        dt_now_str = datetime.datetime.now()
        dt_now_str = dt_now_str.strftime('%Y年%m月%d日 %H:%M:%S')

        # dt_now_str= "2024年05月08日 07:40:20"
        print("時間:", dt_now_str)

        Outlook = win32com.client.Dispatch("Outlook.Application").GetNameSpace("MAPI")
        items = Outlook.GetDefaultFolder(9).Items
        # 定期的な予定の二番目以降の予定を検索に含める
        items.IncludeRecurrences = True
        # 開始時間でソート
        items.Sort("[Start]")
        select_items = [] # 指定した期間内の予定を入れるリスト
        dt_now = datetime.datetime.now()
        start_date = datetime.date(dt_now.year, dt_now.month, dt_now.day)
        end_date = datetime.date(dt_now.year, dt_now.month, dt_now.day + 1)
        strStart = start_date.strftime('%m/%d/%Y %H:%M %p')
        strEnd = end_date.strftime('%m/%d/%Y %H:%M %p')
        sFilter = f"[Start] >= '{strStart}' And [End] <= '{strEnd}'"
        # フィルターを適用し表示
        FilteredItems = items.Restrict(sFilter)
        for item in FilteredItems:
            if start_date <= item.start.date() <= end_date:
                select_items.append(item)
                
        print("今日の予定の件数:", len(select_items))
        # 抜き出した予定の詳細を表示

        # ダミーデータ
        i = 0
        work = ['A社と会議', 'Bさんと面談', 'Cの開発定例', 'D社との商談', 'Eさんの資料のレビュー', 'Fチーム定例会', 'Gチーム戦略会議', 'H部定例', 'I課定例', 'J退勤', 'Kジム']
        room = ['会議室A', '会議室B', 'TeamsC', '会議室D', 'TeamsE', 'TeamsF', '会議室G', '会議室H', 'TeamsI', '移動中J', 'ジムK']
        for select_item in select_items:
            print("件名：", select_item.subject)
            # print(f"件名：予定 {work[i]}")  # 社外秘情報は伏せる
            print("場所：", select_item.location)
            # print(f"場所：会議室 {room[i]}") # 社外秘情報は伏せるd
            print("開始時刻：", str(select_item.Start.Format("%Y/%m/%d %H:%M")))
            print("終了時刻：", str(select_item.End.Format("%Y/%m/%d %H:%M")))
            # print("本文：", select_item.body)
            
            print("----")
            i += 1 # ダミーデータ用


        # ダミーデータ
        i = 0
        text = "今日は" +  dt_now_str + "です。"\
            + "本日" + dt_now_str + "の予定として以下の情報を提供します。\
            あとで予定を聞くので、そのタイミングでリマインドしてください。"

        for select_item in select_items:
            text += "件名：" + select_item.subject
            # text += "件名：" + work[i] # 社外秘情報は伏せる
            text += "場所：" + select_item.location
            # text += "場所：" + "会議室" + room[i] # 社外秘情報は伏せる
            text += "開始時刻：" + str(select_item.Start.Format("%Y/%m/%d %H:%M"))

            text += "終了時刻：" + str(select_item.End.Format("%Y/%m/%d %H:%M"))

            text += "----"
            i += 1 # ダミーデータ用
            
        # user_action = {"role": "user", "content": text}
        # conversationHistory.append(user_action)


        

        return text

    def chat(self, conversationHistory):
        # APIリクエストを作成する
        response = openai.chat.completions.create(
            messages=conversationHistory,
            max_tokens=1024,
            n=1,
            stream=True,
            temperature=0.5,
            stop=None,
            presence_penalty=0.5,
            frequency_penalty=0.5,
            model="gpt-3.5-turbo"
        )

        # ストリーミングされたテキストを処理する
        fullResponse = ""
        RealTimeResponce = ""   
        for chunk in response:
            text = chunk.choices[0].delta.content # chunk['choices'][0]['delta'].get('content')

            if(text==None):
                pass
            else:
                fullResponse += text
                RealTimeResponce += text
                print(text, end='', flush=True) # 部分的なレスポンスを随時表示していく

                target_char = ["。", "！", "？", "\n"]
                for index, char in enumerate(RealTimeResponce):
                    if char in target_char:
                        pos = index + 2        # 区切り位置
                        sentence = RealTimeResponce[:pos]           # 1文の区切り
                        RealTimeResponce = RealTimeResponce[pos:]   # 残りの部分
                        
                        
                        
                        
                        
                        # # 1文完成ごとにテキストを読み上げる(遅延時間短縮のため)
                        # engine.say(sentence)
                        # engine.runAndWait()
                        break
                    else:
                        pass

        # APIからの完全なレスポンスを返す
        return fullResponse


if __name__ == "__main__":
    schedule = Outlook()
    res = schedule.run()




    conversationHistory = []
    user_action = {"role": "user", "content": res} # text}
    conversationHistory.append(user_action)

    time = datetime.datetime.now()
    time = time.strftime('%H:%M:%S')
    # time = '10:00:00'

    pre_Info = "現在時刻は" + time + "です。\n"
    # LLMに入力
    text = pre_Info + "この後の予定は何ですか？何分後にどこに向かえばいいですか？"
    # ユーザーからの発話内容を会話履歴に追加
    user_action = {"role": "user", "content": text}
    conversationHistory.append(user_action)

    # llm = OpenAI()
    # prompt = PromptTemplate(
    #     input_variables=["input"],
    #     template="{input}"
    # )
    # chain = LLMChain(llm=llm, prompt=prompt)
    # ans = chain(res)
    # APIリクエストを作成する
    print("[ChatGPT]") #応答内容をコンソール出力
    res = schedule.chat(conversationHistory)

    ans = res

    # ChatGPTからの応答内容を会話履歴に追加
    chatGPT_responce = {"role": "assistant", "content": res}
    conversationHistory.append(chatGPT_responce)

    # chain = prompt | llm | StrOutputParser()
    # print(chain(text))
    # print(chain.invoke(text))

    # print(res)
    # print(ans)