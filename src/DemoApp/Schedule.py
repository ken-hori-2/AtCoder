import win32com.client
import datetime





class Outlook():
    
    def run(self):
        dt_now_str = datetime.datetime.now()
        dt_now_str = dt_now_str.strftime('%Y年%m月%d日 %H:%M:%S')

        # dt_now_str= "2024年05月08日 07:40:20"
        # print("時間:", dt_now_str)

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

        # # ダミーデータ
        # i = 0
        # work = ['A社と会議', 'Bさんと面談', 'Cの開発定例', 'D社との商談', 'Eさんの資料のレビュー', 'Fチーム定例会', 'Gチーム戦略会議', 'H部定例', 'I課定例', 'J退勤', 'Kジム']
        # room = ['会議室A', '会議室B', 'TeamsC', '会議室D', 'TeamsE', 'TeamsF', '会議室G', '会議室H', 'TeamsI', '移動中J', 'ジムK']
        # for select_item in select_items:
        #     print("件名：", select_item.subject)
        #     # print(f"件名：予定 {work[i]}")  # 社外秘情報は伏せる
        #     print("場所：", select_item.location)
        #     # print(f"場所：会議室 {room[i]}") # 社外秘情報は伏せるd
        #     print("開始時刻：", str(select_item.Start.Format("%Y/%m/%d %H:%M")))
        #     print("終了時刻：", str(select_item.End.Format("%Y/%m/%d %H:%M")))
        #     # print("本文：", select_item.body)
            
        #     print("----")
        #     i += 1 # ダミーデータ用


        # ダミーデータ
        i = 0
        # text = "今日は" +  dt_now_str + "です。"\
        #     + "本日" + dt_now_str + "の予定として以下の情報を提供します。\
        #     あとで予定を聞くので、そのタイミングでリマインドしてください。"
        text = ""


        home_location = "玉川学園前" # 家の場所（出発地点）
        office_location = ""

        for select_item in select_items:
            text += "\n件名：" + select_item.subject
            # text += "件名：" + work[i] # 社外秘情報は伏せる
            text += "\n場所：" + select_item.location
            # text += "場所：" + "会議室" + room[i] # 社外秘情報は伏せる
            text += "\n開始時刻：" + str(select_item.Start.Format("%Y/%m/%d %H:%M"))

            text += "\n終了時刻：" + str(select_item.End.Format("%Y/%m/%d %H:%M"))

            text += "\n----"
            i += 1 # ダミーデータ用

            if select_item.subject in '出勤中': # 通勤中':
                # start = select_item.Start.Format("%Y/%m/%d %H:%M")
                # end = select_item.End.Format("%Y/%m/%d %H:%M")
                transit_go_start = select_item.Start.Format("%H%M")
                transit_go_end = select_item.End.Format("%H%M")
                print(select_item.subject)
                print(int(transit_go_start))
                print(int(transit_go_end))
            if select_item.subject in '出社中':
                working_start = select_item.Start.Format("%H%M")
                working_end = select_item.End.Format("%H%M")

                office_location = select_item.location

                print(select_item.subject)
                print(int(working_start))
                print(int(working_end))
                print(office_location)
            if select_item.subject in '帰宅中':
                transit_back_start = select_item.Start.Format("%H%M")
                transit_back_end = select_item.End.Format("%H%M")
                print(select_item.subject)
                print(int(transit_back_start))
                print(int(transit_back_end))
            if select_item.subject in '運動中':
                exercise_start = select_item.Start.Format("%H%M")
                exercise_end = select_item.End.Format("%H%M")
                print(select_item.subject)
                print(int(exercise_start))
                print(int(exercise_end))
        



        # print(select_items)

        # return text
        return int(transit_go_start), int(transit_go_end), int(transit_back_start), int(transit_back_end), int(working_start), int(working_end), int(exercise_start), int(exercise_end), home_location, office_location


if __name__ == "__main__":
    schedule = Outlook()
    transit_go_start, transit_go_end, transit_back_start, transit_back_end, working_start, working_end, exercise_start, exercise_end, home_location, office_location = schedule.run()
    # print(res)