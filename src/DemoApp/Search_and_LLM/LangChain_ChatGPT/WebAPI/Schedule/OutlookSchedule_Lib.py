# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import strip

from langchain.tools.base import BaseTool

import win32com.client
import datetime

import pandas as pd

class OutlookSchedule(): # BaseTool): # BaseToolの記述がなくても動く
    
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

        """
        デモするユースケースに応じて手動で時刻を設定する
        """
        # 2024/05/28 変更点
        # dt_now = datetime.datetime.now() # 現在時刻
        # dt_now = datetime.datetime(2024, 5, 24, 8, 30) # 00)
        dt_now = datetime.datetime(2024, 6, 3, 8, 30)
        """
        デモするユースケースに応じて手動で時刻を設定する
        """

        start_date = datetime.date(dt_now.year, dt_now.month, dt_now.day)
        # end_date = datetime.date(dt_now.year, dt_now.month, dt_now.day + 1) # 月末だと、同じ月の次の日がないのでエラーになる

        dt_now += datetime.timedelta(days=1)
        end_date = datetime.date(dt_now.year, dt_now.month, dt_now.day)
        
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


                text += "\n出発駅：" + self.home_location



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

                self.MTG_ScheduleItem()

                text += "\n目的駅：" + self.office_location
            
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

        return text
    def ScheduleItem(self):

        return self.select_items
    
    def getHomeLocation(self):
        return self.home_location
    def getOfficeLocation(self):
        return self.office_location
    
    def getMeetingContents(self):
        return self.meeting_contents
    
    def MTG_ScheduleItem(self): # , select_items):
        select_items_for_mtg = self.ScheduleItem()
        
        meeting_contents = ""
        # mask_list = ["SoC", "BLANC", "昼食"] # 社外秘情報は伏せる
        # mask_list = ["SoC", "BLANC", "昼食", "出社", "AUD", "外販"] # 出社を追加
        mask_list = ["SoC", "BLANC", "昼食", "出社", "AUD", "外販"]

        time_zone = []
        meeting_list = [] # より具体的な内容
        sep = ["START"] # , "END"]

        for select_item in select_items_for_mtg:
            meeting_time = select_item.Start.Format("%Y/%m/%d %H:%M")
            meeting_time = datetime.datetime.strptime(meeting_time, "%Y/%m/%d %H:%M") # 形式に沿って文字列をdatetime型に変換
            
            # # if working_start < meeting_time < working_end: # < としているのは、出社という件名を含まないため
            # if self.working_start <= meeting_time < self.working_end: # <= にするならmask_listに出社を追加
                
            # 社外秘情報は伏せる
            # mask_listに出社を追加
            if (not (mask_list[0] in select_item.subject)) and (not (mask_list[1] in select_item.subject)) and (not (mask_list[2] in select_item.subject)) and (not (mask_list[3] in select_item.subject)) and (not (mask_list[4] in select_item.subject)) and (not (mask_list[5] in select_item.subject)):

                meeting_contents += "\n件名：" + select_item.subject
                meeting_contents += "\n場所：" + select_item.location
                meeting_contents += "\n開始時刻：" + str(select_item.Start.Format("%Y/%m/%d %H:%M"))
                meeting_contents += "\n終了時刻：" + str(select_item.End.Format("%Y/%m/%d %H:%M"))
                # 2024/05/28 追加
                if (select_item.subject in "出勤 通勤"):
                    meeting_contents += "\n出発地：" + self.home_location
                    meeting_contents += "\n目的地：" + self.office_location
                elif (select_item.subject in "帰宅"):
                    meeting_contents += "\n目的地：" + self.home_location
                    meeting_contents += "\n出発地：" + self.office_location
                # 2024/05/28 追加
                meeting_contents += "\n----"

                # time_zone.append([select_item.Start.Format("%H%M"), select_item.End.Format("%H%M")])
                time_zone.append(select_item.Start.Format("%Y/%m/%d %H:%M")) # "%H%M"))
                meeting_list.append(select_item.subject)
            
            
            # # 2024/05/28 追加
            # if (select_item.subject in "出勤 通勤"):
            #     meeting_contents += "\n出発地：" + self.home_location
            #     meeting_contents += "\n目的地：" + self.office_location
            # elif (select_item.subject in "帰宅"):
            #     meeting_contents += "\n目的地：" + self.home_location
            #     meeting_contents += "\n出発地：" + self.office_location
            # # elif (select_item.subject in "出社"):
            # # 2024/05/28 追加

        # print("#####")
        # print("meeting : ", meeting_contents)
        # print("#####")
    
        self.MTG_Section = pd.DataFrame(data=time_zone, index=meeting_list, columns=sep)
        self.meeting_contents = meeting_contents

        # return MTG_Section, meeting_contents
    
    
    def isMtgStartWithin5min(self, margin, time):
        num = len(self.MTG_Section)
        MTG_START_LIST = []
        ret_list = []
        isTrue = False

        for i in range(num):
            str = self.MTG_Section["START"].iloc[i]
            date_time = datetime.datetime.strptime(str, "%Y/%m/%d %H:%M") # STARTの中に格納されている形式に沿って文字列をdatetime型に変換
            MTG_START_LIST.append(date_time)
        

        for i in range(num):
            try:
                if MTG_START_LIST[i] -margin <= time <= MTG_START_LIST[i] +margin: # マージンなので、前後5分は含む
                    self.next_mtg = self.MTG_Section.index[i]
                    isTrue = True
            except:
                print("\n\n\n\n\n[Schedule_new.py] isMtgStartWithin5min() ... MTG_START_LIST[i] is Error !!!!!")
                print("予定表にないため、MTG_ScheduleItem()実行時に値が格納されていない\n\n\n\n\n")
        self.ret_list = ret_list
        return isTrue
    
    def getNextMtg(self):
        return self.next_mtg
    
    
    def isLunchStartWithin5min(self, margin, time):

        isTrue = False
        try:
            if self.lunch_start -margin <= time < self.lunch_end +margin: # <(終了時刻は含まない)
                isTrue = True
        except:
            print("\n\n\n\n\n[Schedule_new.py] isLunchStartWithin5min() ... self.lunch_start is Error !!!!!")
            print("予定表にないため、run()実行時に値が格納されていない\n\n\n\n\n")

        return isTrue
    
    def isWorking(self, margin, time):
        isTrue = False
        try:
            if self.working_start -margin <= time < self.working_end +margin: # <(終了時刻は含まない)
                isTrue = True
        except:
            print("\n\n\n\n\n[Schedule_new.py] isWorking() ... self.working_start is Error !!!!!")
            print("予定表にないため、run()実行時に値が格納されていない\n\n\n\n\n")

        return isTrue
    
    def isTransitingGo(self, margin, time):
        isTrue = False
        try:
            if self.transit_go_start -margin <= time < self.transit_go_end +margin: # <(終了時刻は含まない)
                isTrue = True
        except:
            print("\n\n\n\n\n[Schedule_new.py] isTransitingGo() ... self.transit_go_start is Error !!!!!")
            print("予定表にないため、run()実行時に値が格納されていない\n\n\n\n\n")

        return isTrue
    
    def isTransitingBack(self, margin, time):
        isTrue = False
        try:
            if self.transit_back_start -margin <= time < self.transit_back_end +margin: # <(終了時刻は含まない)
                isTrue = True
        except:
            print("\n\n\n\n\n[Schedule_new.py] isTransitingBack() ... self.transit_back_start is Error !!!!!")
            print("予定表にないため、run()実行時に値が格納されていない\n\n\n\n\n")

        return isTrue
    
    def isExercising(self, margin, time):
        isTrue = False
        try:
            if self.exercise_start -margin <= time < self.exercise_end +margin: # <(終了時刻は含まない)
                isTrue = True
        except:
            print("\n\n\n\n\n[Schedule_new.py] isExercising() ... self.exercise_start is Error !!!!!")
            print("予定表にないため、run()実行時に値が格納されていない\n\n\n\n\n")

        return isTrue

if __name__ == "__main__":
    
    outlook_schedule = OutlookSchedule()
    # result = 
    outlook_schedule.run()
    # print(result)
    print("-----")
    outlook_schedule.MTG_ScheduleItem()
    result = outlook_schedule.getMeetingContents()
    print(result)
