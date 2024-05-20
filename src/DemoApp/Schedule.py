import win32com.client
import datetime

import pandas as pd





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
        
        self.select_items = [] # 指定した期間内の予定を入れるリスト

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
                self.select_items.append(item)
                
        print("今日の予定の件数:", len(self.select_items))
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


        # transit_go_start = ""
        # transit_go_end = ""
        # working_start = ""
        # working_end = ""
        # transit_back_start = ""
        # transit_back_end = ""
        # exercise_start = ""
        # exercise_end = ""
        
        home_location = "玉川学園前" # 家の場所（出発地点）
        office_location = ""

        for select_item in self.select_items:
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
            
            if select_item.subject in '昼食中':
                lunch_start = select_item.Start.Format("%H%M")
                lunch_end = select_item.End.Format("%H%M")
                print(select_item.subject)
                print(int(lunch_start))
                print(int(lunch_end))
            

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
        return int(transit_go_start), int(transit_go_end), int(transit_back_start), int(transit_back_end), int(working_start), int(working_end), int(exercise_start), int(exercise_end), int(lunch_start), int(lunch_end), home_location, office_location
    
    def ScheduleItem(self):

        return self.select_items
    
    def MTG_ScheduleItem(self, select_items, working_start, working_end):
        meeting_contents = ""
        mask_list = ["SoC", "BLANC", "昼食"] # 社外秘情報は伏せる
        
        
        time_zone = []
        meeting_list = [] # より具体的な内容
        sep = ["START"] # , "END"]

        for select_item in select_items:

            meeting_time = str(select_item.Start.Format("%H%M")) # %Y/%m/%d %H:%M"))
            
            # if working_start <= int(meeting_time) < working_end:
            # if working_start < int(time_now) < working_end:
            if working_start < int(meeting_time) < working_end:
                # next_working_start = select_item.Start.Format("%H%M")

                # 社外秘情報は伏せる
                if (not (mask_list[0] in select_item.subject)) and (not (mask_list[1] in select_item.subject)) and (not (mask_list[2] in select_item.subject)):

                    meeting_contents += "\n件名：" + select_item.subject
                    meeting_contents += "\n場所：" + select_item.location
                    meeting_contents += "\n開始時刻：" + str(select_item.Start.Format("%Y/%m/%d %H:%M"))
                    meeting_contents += "\n----"

                    # time_zone.append([select_item.Start.Format("%H%M"), select_item.End.Format("%H%M")])
                    time_zone.append(select_item.Start.Format("%H%M"))
                    meeting_list.append(select_item.subject)
        
        MTG_Section = pd.DataFrame(data=time_zone, index=meeting_list, columns=sep)

        # print(time_zone)
        # print(MTG_Section)

        return MTG_Section, meeting_contents
    
    def isMtgStartWithin5min(self, MTG_Section, margin, time):
        print(MTG_Section.index)
        num = len(MTG_Section)
        MTG_START_LIST = []
        ret_list = []
        isTrue = False

        for i in range(num):
            MTG_START_LIST.append(MTG_Section["START"].iloc[i])
        
        print(MTG_START_LIST) # .loc[i])

        print("#######################")
        for i in range(num):
            
            # 30分からスタートの場合、-40にするとおかしくなる
            
            print(int(MTG_START_LIST[i])-margin-40)
            print(int(MTG_START_LIST[i])+margin)
            if int(MTG_START_LIST[i]) -margin -40 <= int(time) <= int(MTG_START_LIST[i]) +margin:
                # print("TRUE!")
                ret_list.append(MTG_Section.index[i])
                isTrue = True
         
        print("#######################")
        return isTrue # , ret_list


if __name__ == "__main__":
    schedule = Outlook()
    transit_go_start, transit_go_end, transit_back_start, transit_back_end, working_start, working_end, exercise_start, exercise_end, lunch_start, lunch_end, home_location, office_location = schedule.run()
    # print(res)


    
    
    
    
    
    margin = 5
    time = "1100"
    select_items = schedule.ScheduleItem()
    MTG_Section, text = schedule.MTG_ScheduleItem(select_items, working_start, working_end)
    isMtgStart = schedule.isMtgStartWithin5min(MTG_Section, margin, time)
    print(isMtgStart)


    print(text)
    
    
    # print("\n直近の予定のみ入力する")
    # print(meeting_contents)
    # print(meeting_list)