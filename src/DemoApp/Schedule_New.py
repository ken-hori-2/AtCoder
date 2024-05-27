import win32com.client
import datetime

import pandas as pd





class Outlook():
    
    def run(self):
        # dt_now_str = datetime.datetime.now()
        # dt_now_str = dt_now_str.strftime('%Y年%m月%d日 %H:%M:%S')

        # # dt_now_str= "2024年05月08日 07:40:20"
        # # print("時間:", dt_now_str)

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

        i = 0
        text = ""
        # self.transit_go_start = ""
        # self.transit_go_end = ""
        # self.working_start = ""
        # self.working_end = ""
        # self.transit_back_start = ""
        # self.transit_back_end = ""
        # self.exercise_start = ""
        # self.exercise_end = ""
        
        self.home_location = "玉川学園前" # 家の場所（出発地点）
        self.office_location = ""

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
                self.transit_go_start = select_item.Start.Format("%Y/%m/%d %H:%M") # "%H%M")
                self.transit_go_end = select_item.End.Format("%Y/%m/%d %H:%M") # "%H%M")
                # print(select_item.subject)
                # # print(int(transit_go_start))
                # # print(int(transit_go_end))
                # print(self.transit_go_start)
                # print(self.transit_go_end)

                self.transit_go_start = datetime.datetime.strptime(self.transit_go_start, "%Y/%m/%d %H:%M") # 形式に沿って文字列をdatetime型に変換
                self.transit_go_end = datetime.datetime.strptime(self.transit_go_end, "%Y/%m/%d %H:%M") # 形式に沿って文字列をdatetime型に変換


            if select_item.subject in '出社中':
                self.working_start = select_item.Start.Format("%Y/%m/%d %H:%M") # "%H%M")
                self.working_end = select_item.End.Format("%Y/%m/%d %H:%M") # "%H%M")

                self.office_location = select_item.location

                # print(select_item.subject)
                # # print(int(working_start))
                # # print(int(working_end))
                # print(self.working_start)
                # print(self.working_end)
                # print(self.office_location)

                self.working_start = datetime.datetime.strptime(self.working_start, "%Y/%m/%d %H:%M") # 形式に沿って文字列をdatetime型に変換
                self.working_end = datetime.datetime.strptime(self.working_end, "%Y/%m/%d %H:%M") # 形式に沿って文字列をdatetime型に変換


                # # def MTG_ScheduleItem
                # meeting_contents = ""
                # # mask_list = ["SoC", "BLANC", "昼食"] # 社外秘情報は伏せる
                # mask_list = ["SoC", "BLANC", "昼食", "出社"] # 出社を追加
                # time_zone = []
                # meeting_list = [] # より具体的な内容
                # sep = ["START"] # , "END"]
                # meeting_time = select_item.Start.Format("%Y/%m/%d %H:%M")
                # meeting_time = datetime.datetime.strptime(meeting_time, "%Y/%m/%d %H:%M") # 形式に沿って文字列をdatetime型に変換
                # print("meeting_time = ", meeting_time)
                # # if working_start < meeting_time < working_end: # < としているのは、出社という件名を含まないため
                # if self.working_start <= meeting_time < self.working_end: # <= にするならmask_listに出社を追加
                #     # 社外秘情報は伏せる
                #     # mask_listに出社を追加
                #     if (not (mask_list[0] in select_item.subject)) and (not (mask_list[1] in select_item.subject)) and (not (mask_list[2] in select_item.subject)) and (not (mask_list[3] in select_item.subject)):

                #         meeting_contents += "\n件名：" + select_item.subject
                #         meeting_contents += "\n場所：" + select_item.location
                #         meeting_contents += "\n開始時刻：" + str(select_item.Start.Format("%Y/%m/%d %H:%M"))
                #         meeting_contents += "\n----"

                #         # time_zone.append([select_item.Start.Format("%H%M"), select_item.End.Format("%H%M")])
                #         time_zone.append(select_item.Start.Format("%Y/%m/%d %H:%M")) # "%H%M"))
                #         meeting_list.append(select_item.subject)
            
                # self.MTG_Section = pd.DataFrame(data=time_zone, index=meeting_list, columns=sep)

                # # return MTG_Section, meeting_contents
                self.MTG_ScheduleItem()
            
            if select_item.subject in '昼食中':
                self.lunch_start = select_item.Start.Format("%Y/%m/%d %H:%M") # "%H%M")
                self.lunch_end = select_item.End.Format("%Y/%m/%d %H:%M") # "%H%M")
                # print(select_item.subject)
                # # print(int(lunch_start))
                # # print(int(lunch_end))
                # print(self.lunch_start)
                # print(self.lunch_end)

                self.lunch_start = datetime.datetime.strptime(self.lunch_start, "%Y/%m/%d %H:%M") # 形式に沿って文字列をdatetime型に変換
                self.lunch_end = datetime.datetime.strptime(self.lunch_end, "%Y/%m/%d %H:%M") # 形式に沿って文字列をdatetime型に変換
            

            if select_item.subject in '帰宅中':
                self.transit_back_start = select_item.Start.Format("%Y/%m/%d %H:%M") # "%H%M")
                self.transit_back_end = select_item.End.Format("%Y/%m/%d %H:%M") # "%H%M")
                # print(select_item.subject)
                # # print(int(transit_back_start))
                # # print(int(transit_back_end))
                # print(self.transit_back_start)
                # print(self.transit_back_end)

                self.transit_back_start = datetime.datetime.strptime(self.transit_back_start, "%Y/%m/%d %H:%M") # 形式に沿って文字列をdatetime型に変換
                self.transit_back_end = datetime.datetime.strptime(self.transit_back_end, "%Y/%m/%d %H:%M") # 形式に沿って文字列をdatetime型に変換


            if select_item.subject in '運動中':
                self.exercise_start = select_item.Start.Format("%Y/%m/%d %H:%M") # "%H%M")
                self.exercise_end = select_item.End.Format("%Y/%m/%d %H:%M") # "%H%M")
                # print(select_item.subject)
                # # print(int(exercise_start))
                # # print(int(exercise_end))
                # print(self.exercise_start)
                # print(self.exercise_end)

                self.exercise_start = datetime.datetime.strptime(self.exercise_start, "%Y/%m/%d %H:%M") # 形式に沿って文字列をdatetime型に変換
                self.exercise_end = datetime.datetime.strptime(self.exercise_end, "%Y/%m/%d %H:%M") # 形式に沿って文字列をdatetime型に変換
    
    def ScheduleItem(self):

        return self.select_items
    
    def getHomeLocation(self):
        return self.home_location
    def getOfficeLocation(self):
        return self.office_location
    
    def getMeetingContents(self):
        return self.meeting_contents
    
    # def MTG_ScheduleItem(self, select_items): # , working_start, working_end):
    #     meeting_contents = ""
    #     # mask_list = ["SoC", "BLANC", "昼食"] # 社外秘情報は伏せる
    #     mask_list = ["SoC", "BLANC", "昼食", "出社"] # 出社を追加
        
        
    #     time_zone = []
    #     meeting_list = [] # より具体的な内容
    #     sep = ["START"] # , "END"]


    #     # self.working_start = datetime.datetime.strptime(self.working_start, "%Y/%m/%d %H:%M") # 形式に沿って文字列をdatetime型に変換
    #     # print("working start = ", self.working_start)
    #     # self.working_end = datetime.datetime.strptime(self.working_end, "%Y/%m/%d %H:%M") # 形式に沿って文字列をdatetime型に変換
    #     # print("working end = ", self.working_end)

    #     for select_item in select_items:

    #         # meeting_time = str(select_item.Start.Format("%Y/%m/%d %H:%M")) # "%H%M"))
    #         meeting_time = select_item.Start.Format("%Y/%m/%d %H:%M")
    #         meeting_time = datetime.datetime.strptime(meeting_time, "%Y/%m/%d %H:%M") # 形式に沿って文字列をdatetime型に変換
    #         print("meeting_time = ", meeting_time)

    #         # print("TEST:", type(meeting_time))
            
    #         # if working_start <= int(meeting_time) < working_end:
    #         # if working_start < int(time_now) < working_end:
    #         # if working_start < int(meeting_time) < working_end:
            
    #         # if working_start < meeting_time < working_end: # < としているのは、出社という件名を含まないため
    #         if self.working_start <= meeting_time < self.working_end: # <= にするならmask_listに出社を追加
    #             # next_working_start = select_item.Start.Format("%H%M")

    #             # 社外秘情報は伏せる
    #             # if (not (mask_list[0] in select_item.subject)) and (not (mask_list[1] in select_item.subject)) and (not (mask_list[2] in select_item.subject)):
    #             # mask_listに出社を追加
    #             if (not (mask_list[0] in select_item.subject)) and (not (mask_list[1] in select_item.subject)) and (not (mask_list[2] in select_item.subject)) and (not (mask_list[3] in select_item.subject)):

    #                 meeting_contents += "\n件名：" + select_item.subject
    #                 meeting_contents += "\n場所：" + select_item.location
    #                 meeting_contents += "\n開始時刻：" + str(select_item.Start.Format("%Y/%m/%d %H:%M"))
    #                 meeting_contents += "\n----"

    #                 # time_zone.append([select_item.Start.Format("%H%M"), select_item.End.Format("%H%M")])
    #                 time_zone.append(select_item.Start.Format("%Y/%m/%d %H:%M")) # "%H%M"))
    #                 meeting_list.append(select_item.subject)
        
    #     MTG_Section = pd.DataFrame(data=time_zone, index=meeting_list, columns=sep)

    #     return MTG_Section, meeting_contents

    def MTG_ScheduleItem(self): # , select_items):
        select_items_for_mtg = self.ScheduleItem()
        
        meeting_contents = ""
        # mask_list = ["SoC", "BLANC", "昼食"] # 社外秘情報は伏せる
        mask_list = ["SoC", "BLANC", "昼食", "出社", "AUD", "外販"] # 出社を追加
        time_zone = []
        meeting_list = [] # より具体的な内容
        sep = ["START"] # , "END"]

        for select_item in select_items_for_mtg:
            meeting_time = select_item.Start.Format("%Y/%m/%d %H:%M")
            meeting_time = datetime.datetime.strptime(meeting_time, "%Y/%m/%d %H:%M") # 形式に沿って文字列をdatetime型に変換
            
            # if working_start < meeting_time < working_end: # < としているのは、出社という件名を含まないため
            if self.working_start <= meeting_time < self.working_end: # <= にするならmask_listに出社を追加
                
                # 社外秘情報は伏せる
                # mask_listに出社を追加
                if (not (mask_list[0] in select_item.subject)) and (not (mask_list[1] in select_item.subject)) and (not (mask_list[2] in select_item.subject)) and (not (mask_list[3] in select_item.subject)):

                    meeting_contents += "\n件名：" + select_item.subject
                    meeting_contents += "\n場所：" + select_item.location
                    meeting_contents += "\n開始時刻：" + str(select_item.Start.Format("%Y/%m/%d %H:%M"))
                    meeting_contents += "\n----"

                    # time_zone.append([select_item.Start.Format("%H%M"), select_item.End.Format("%H%M")])
                    time_zone.append(select_item.Start.Format("%Y/%m/%d %H:%M")) # "%H%M"))
                    meeting_list.append(select_item.subject)

        print("#####")
        print("meeting : ", meeting_contents)
        print("#####")
    
        self.MTG_Section = pd.DataFrame(data=time_zone, index=meeting_list, columns=sep)
        self.meeting_contents = meeting_contents

        # return MTG_Section, meeting_contents
    
    # def isMtgStartWithin5min(self, MTG_Section, margin, time):
    def isMtgStartWithin5min(self, margin, time):

        # time : datetime形式
        # MTG_Section : datetime形式に変換
        # margin : timedelta形式


        # print(self.MTG_Section.index)
        num = len(self.MTG_Section)
        MTG_START_LIST = []
        ret_list = []
        isTrue = False
        
        # margin = datetime.timedelta(minutes=margin)
        # print("margin = ", margin)

        for i in range(num):
            # print(type(MTG_Section["START"].iloc[i]))
            # MTG_START_LIST.append(MTG_Section["START"].iloc[i])

            str = self.MTG_Section["START"].iloc[i]
            date_time = datetime.datetime.strptime(str, "%Y/%m/%d %H:%M") # STARTの中に格納されている形式に沿って文字列をdatetime型に変換
            MTG_START_LIST.append(date_time)
        
        # print(MTG_START_LIST) # .loc[i])

        # print("#######################")
        for i in range(num):
            # print(type(MTG_START_LIST[i]))

            # 30分からスタートの場合、-40にするとおかしくなる
            
            # print(int(MTG_START_LIST[i])-margin-40)
            # print(int(MTG_START_LIST[i])+margin)
            # if int(MTG_START_LIST[i]) -margin -40 <= int(time) <= int(MTG_START_LIST[i]) +margin:
            """ or """
            # print(MTG_START_LIST[i]-margin)
            # print(MTG_START_LIST[i]+margin)
            try:
                if MTG_START_LIST[i] -margin <= time <= MTG_START_LIST[i] +margin: # マージンなので、前後5分は含む
                    # print("TRUE!")
                    # ret_list.append(self.MTG_Section.index[i])

                    self.next_mtg = self.MTG_Section.index[i]

                    
                    # ret_list.append(MTG_START_LIST[i])
                    isTrue = True
            except:
                print("\n\n\n\n\n[Schedule_new.py] isMtgStartWithin5min() ... MTG_START_LIST[i] is Error !!!!!")
                print("予定表にないため、MTG_ScheduleItem()実行時に値が格納されていない\n\n\n\n\n")
         
        # print("#######################")
        self.ret_list = ret_list
        return isTrue # , ret_list
    
    def getNextMtg(self):
        return self.next_mtg # self.ret_list
    
    # def isLunchStartWithin5min(self, lunch_start, lunch_end, margin, time):
    def isLunchStartWithin5min(self, margin, time):

        # time : datetime形式
        # lunch_start, lunch_end : datetime形式に変換
        # margin : timedelta形式
        
        isTrue = False
        # margin = datetime.timedelta(minutes=margin)
        # print("margin = ", margin)

        # # self.lunch_start = datetime.datetime.strptime(self.lunch_start, "%Y/%m/%d %H:%M") # 形式に沿って文字列をdatetime型に変換
        # print("lunch start = ", self.lunch_start)
        # # self.lunch_end = datetime.datetime.strptime(self.lunch_end, "%Y/%m/%d %H:%M") # 形式に沿って文字列をdatetime型に変換
        # print("lunch end = ", self.lunch_end)

        # if lunch_start -margin <= time <= lunch_end +margin: # <=(終了時刻も含む)
        try:
            if self.lunch_start -margin <= time < self.lunch_end +margin: # <(終了時刻は含まない)
                isTrue = True
        except:
            print("\n\n\n\n\n[Schedule_new.py] isLunchStartWithin5min() ... self.lunch_start is Error !!!!!")
            print("予定表にないため、run()実行時に値が格納されていない\n\n\n\n\n")

        return isTrue # , ret_list
    
    # def isWorking(self, working_start, working_end, margin, time):
    def isWorking(self, margin, time):

        # time : datetime形式
        # working_start, working_end : datetime形式に変換
        # margin : timedelta形式
        
        isTrue = False
        # margin = datetime.timedelta(minutes=margin)
        # print("margin = ", margin)

        # # self.working_start = datetime.datetime.strptime(self.working_start, "%Y/%m/%d %H:%M") # 形式に沿って文字列をdatetime型に変換
        # print("working start = ", self.working_start)
        # # self.working_end = datetime.datetime.strptime(self.working_end, "%Y/%m/%d %H:%M") # 形式に沿って文字列をdatetime型に変換
        # print("working end = ", self.working_end)

        # if working_start -margin <= time <= working_end +margin: # <=(終了時刻も含む)
        try:
            if self.working_start -margin <= time < self.working_end +margin: # <(終了時刻は含まない)
                isTrue = True
        except:
            print("\n\n\n\n\n[Schedule_new.py] isWorking() ... self.working_start is Error !!!!!")
            print("予定表にないため、run()実行時に値が格納されていない\n\n\n\n\n")

        return isTrue # , ret_lists
    
    def isTransitingGo(self, margin, time):

        # time : datetime形式
        # transiting_start, transiting_end : datetime形式に変換
        # margin : timedelta形式
        
        isTrue = False
        # margin = datetime.timedelta(minutes=margin)
        # print("margin = ", margin)

        # # self.transit_go_start = datetime.datetime.strptime(self.transit_go_start, "%Y/%m/%d %H:%M") # 形式に沿って文字列をdatetime型に変換
        # print("transiting go start = ", self.transit_go_start)
        # # self.transit_go_end = datetime.datetime.strptime(self.transit_go_end, "%Y/%m/%d %H:%M") # 形式に沿って文字列をdatetime型に変換
        # print("transiting go end = ", self.transit_go_end)

        # if transiting_start -margin <= time <= transiting_end +margin: # <=(終了時刻も含む)
        try:
            if self.transit_go_start -margin <= time < self.transit_go_end +margin: # <(終了時刻は含まない)
                isTrue = True
        except:
            print("\n\n\n\n\n[Schedule_new.py] isTransitingGo() ... self.transit_go_start is Error !!!!!")
            print("予定表にないため、run()実行時に値が格納されていない\n\n\n\n\n")

        return isTrue # , ret_list
    
    def isTransitingBack(self, margin, time):

        # time : datetime形式
        # transiting_start, transiting_end : datetime形式に変換
        # margin : timedelta形式
        
        isTrue = False
        # margin = datetime.timedelta(minutes=margin)
        # print("margin = ", margin)

        # # self.transit_back_start = datetime.datetime.strptime(self.transit_back_start, "%Y/%m/%d %H:%M") # 形式に沿って文字列をdatetime型に変換
        # print("transiting back start = ", self.transit_back_start)
        # # self.transit_back_end = datetime.datetime.strptime(self.transit_back_end, "%Y/%m/%d %H:%M") # 形式に沿って文字列をdatetime型に変換
        # print("transiting back end = ", self.transit_back_end)

        # if transiting_start -margin <= time <= transiting_end +margin: # <=(終了時刻も含む)
        try:
            if self.transit_back_start -margin <= time < self.transit_back_end +margin: # <(終了時刻は含まない)
                isTrue = True
        except:
            print("\n\n\n\n\n[Schedule_new.py] isTransitingBack() ... self.transit_back_start is Error !!!!!")
            print("予定表にないため、run()実行時に値が格納されていない\n\n\n\n\n")

        return isTrue # , ret_list
    
    def isExercising(self, margin, time):

        # time : datetime形式
        # transiting_start, transiting_end : datetime形式に変換
        # margin : timedelta形式
        
        isTrue = False
        # margin = datetime.timedelta(minutes=margin)
        # print("margin = ", margin)

        # # self.exercise_start = datetime.datetime.strptime(self.exercise_start, "%Y/%m/%d %H:%M") # 形式に沿って文字列をdatetime型に変換
        # print("excercise start = ", self.exercise_start)
        # # self.exercise_end = datetime.datetime.strptime(self.exercise_end, "%Y/%m/%d %H:%M") # 形式に沿って文字列をdatetime型に変換
        # print("excercise end = ", self.exercise_end)

        # if transiting_start -margin <= time <= transiting_end +margin: # <=(終了時刻も含む)
        try:
            if self.exercise_start -margin <= time < self.exercise_end +margin: # <(終了時刻は含まない)
                isTrue = True
        except:
            print("\n\n\n\n\n[Schedule_new.py] isExercising() ... self.exercise_start is Error !!!!!")
            print("予定表にないため、run()実行時に値が格納されていない\n\n\n\n\n")

        return isTrue # , ret_list


if __name__ == "__main__":
    schedule = Outlook()
    # transit_go_start, transit_go_end, transit_back_start, transit_back_end, working_start, working_end, exercise_start, exercise_end, lunch_start, lunch_end, home_location, office_location = schedule.run()
    schedule.run()
    # print(res)


    
    
    
    
    
    # dt_now_str = datetime.datetime.now()
    # dt_now = datetime.datetime(2024, 5, 15, 12, 00) # 現在時刻を手動で設定
    dt_now = datetime.datetime(2024, 5, 15, 13, 58)
    print("現在時刻：", dt_now)
    margin = 5
    margin = datetime.timedelta(minutes=margin)

    time = dt_now

    # select_items = schedule.ScheduleItem()

    # 一旦run内にマージ
    # MTG_Section, text = schedule.MTG_ScheduleItem(select_items) # , working_start, working_end)

    # isMtgStart = schedule.isMtgStartWithin5min(MTG_Section, margin, time)
    isMtgStart = schedule.isMtgStartWithin5min(margin, time)
    print("MtgStart:", isMtgStart)

    # isLunchStart = schedule.isLunchStartWithin5min(lunch_start, lunch_end, margin, time)
    isLunchStart = schedule.isLunchStartWithin5min(margin, time)
    print("LunchStart:", isLunchStart)




    # isWorking = schedule.isWorking(working_start, working_end, margin, time) # 引数として渡さなくてselfでいい
    isWorking = schedule.isWorking(margin, time)
    print("Working:", isWorking)

    isTransitingGo = schedule.isTransitingGo(margin, time)
    print("TransitingGo:", isTransitingGo)

    isTransitingBack = schedule.isTransitingBack(margin, time)
    print("TransitingBack:", isTransitingBack)

    isExercising = schedule.isExercising(margin, time)
    print("isExercising:", isExercising)


    # print(text)
    
    
    # print("\n直近の予定のみ入力する")
    # print(meeting_contents)
    # print(meeting_list)


    print("\n直近の予定")
    # print(schedule.getNextMtg()) # isWithin5min~を実行しないとエラー（self.next_mtgに書き込まれない）