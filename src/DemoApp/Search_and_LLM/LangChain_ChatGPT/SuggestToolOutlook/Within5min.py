"""
トークン削減のためテスト用
"""
# 5分以内に次の予定が始まる場合（トリガー）

import win32com.client
import datetime

class CheckScheduleTime():
    def __init__(self, dt_now):
        Outlook = win32com.client.Dispatch("Outlook.Application").GetNameSpace("MAPI")
        items = Outlook.GetDefaultFolder(9).Items
        # 定期的な予定の二番目以降の予定を検索に含める
        items.IncludeRecurrences = True
        # 開始時間でソート
        items.Sort("[Start]")
        self.select_items = [] # 指定した期間内の予定を入れるリスト # select_items = []

        """
        時刻はどこか一か所にまとめる # 2024/05/29
        """
        # # dt_now = datetime.datetime.now() # 2024/05/28 一旦コメントアウト(現在時刻)
        # # dt_now = datetime.datetime(2024, 5, 24, 8, 00) # テスト用
        # self.dt_now = datetime.datetime(2024, 5, 24, 10, 50) # テスト用
        # self.dt_now = datetime.datetime(2024, 5, 24, 10, 55) # テスト用
        self.dt_now = dt_now

        start_date = datetime.date(self.dt_now.year, self.dt_now.month, self.dt_now.day)
        end_date = datetime.date(self.dt_now.year, self.dt_now.month, self.dt_now.day + 1)
        strStart = start_date.strftime('%m/%d/%Y %H:%M %p')
        strEnd = end_date.strftime('%m/%d/%Y %H:%M %p')
        sFilter = f"[Start] >= '{strStart}' And [End] <= '{strEnd}'"
        # フィルターを適用し表示
        FilteredItems = items.Restrict(sFilter)
        for item in FilteredItems:
            if start_date <= item.start.date() <= end_date:
                self.select_items.append(item) # select_items.append(item)
                
        print("今日の予定の件数:", len(self.select_items)) # print("今日の予定の件数:", len(select_items))

        self.meeting_contents = ""



    def isScheduleStartWithin5min(self, margin=None, current_time=None):
        
        margin = 5
        margin = datetime.timedelta(minutes=margin)
        isTrue = False
        for select_item in self.select_items: # for select_item in select_items:
            Contents_Start = select_item.Start.Format("%Y/%m/%d %H:%M")
            # print(Contents)
            date_time = datetime.datetime.strptime(Contents_Start, "%Y/%m/%d %H:%M")
            print(date_time)
            # try:
            if date_time -margin <= self.dt_now <= date_time +margin: # マージンなので、前後5分は含む
                isTrue = True
                print("True!!!!!")

            
            # if isTrue:
            #     self.meeting_contents = "\n件名：" + select_item.subject
                
            # except:
                # print("エラー")

        print(type(date_time))
        return isTrue
    
    # def getScheduleContents(self):
    #     return self.meeting_contents
    
    def getScheduleContents(self):
        # mask_list = ["SoC", "BLANC", "昼食", "出社", "AUD", "外販"] # 出社を追加
        mask_list = ["SoC", "BLANC", "出社", "AUD", "外販"] # 出社を追加
        mask_list_ok = ["出勤", "新人研修相談", "定例会議1", "顧客定例2", "ブレスト定例", "LLMデモンストレーション", "帰宅", "運動", "昼食"] # 入力していい情報のみ通す

        margin = 5
        margin = datetime.timedelta(minutes=margin)

        for select_item in self.select_items: # select_items_for_mtg:
            meeting_time_start = select_item.Start.Format("%Y/%m/%d %H:%M")
            meeting_time_start = datetime.datetime.strptime(meeting_time_start, "%Y/%m/%d %H:%M") # 形式に沿って文字列をdatetime型に変換
            meeting_time_end = select_item.End.Format("%Y/%m/%d %H:%M")
            meeting_time_end = datetime.datetime.strptime(meeting_time_end, "%Y/%m/%d %H:%M") # 形式に沿って文字列をdatetime型に変換
            
            # if working_start < meeting_time < working_end: # < としているのは、出社という件名を含まないため
            if meeting_time_start - margin <= self.dt_now < meeting_time_end + margin: # <= にするならmask_listに出社を追加
                
                """
                # 社外秘情報は伏せる
                """
                # # mask_listに出社を追加
                # if (not (mask_list[0] in select_item.subject)) and (not (mask_list[1] in select_item.subject)) and (not (mask_list[2] in select_item.subject)) and (not (mask_list[3] in select_item.subject)):
                # 入力していい情報のみ通す場合
                is_inputok = False
                for subject in mask_list_ok:
                    if subject in select_item.subject:
                        is_inputok = True
                if is_inputok:
                    self.meeting_contents += "\n件名：" + select_item.subject

        print("#####")
        print("meeting : ", self.meeting_contents)
        print("#####")

        return self.meeting_contents

if __name__ == "__main__":

    # dt_now = datetime.datetime(2024, 5, 24, 8, 30)
    # dt_now = datetime.datetime(2024, 5, 24, 8, 30)
    dt_now = datetime.datetime(2024, 5, 24, 10, 50)
    dt_now = datetime.datetime(2024, 5, 24, 10, 55)

    dt_now = datetime.datetime(2024, 7, 17, 10, 55)


    check_schedule = CheckScheduleTime(dt_now)
    is5min = check_schedule.isScheduleStartWithin5min()

    schedule_contents = check_schedule.getScheduleContents()
    print(schedule_contents)
    if schedule_contents:
        print("True")