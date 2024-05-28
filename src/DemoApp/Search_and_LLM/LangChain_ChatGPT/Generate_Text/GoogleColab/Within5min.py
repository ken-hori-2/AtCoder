"""
トークン削減のためテスト用
"""
# 5分以内に次の予定が始まる場合（トリガー）

import win32com.client
import datetime
Outlook = win32com.client.Dispatch("Outlook.Application").GetNameSpace("MAPI")
items = Outlook.GetDefaultFolder(9).Items
# 定期的な予定の二番目以降の予定を検索に含める
items.IncludeRecurrences = True
# 開始時間でソート
items.Sort("[Start]")
# self.select_items = [] # 指定した期間内の予定を入れるリスト
select_items = []

# 2024/05/28 一旦コメントアウト
# dt_now = datetime.datetime.now()
# dt_now = datetime.datetime(2024, 5, 24, 8, 00) # テスト用
dt_now = datetime.datetime(2024, 5, 24, 10, 50) # テスト用
dt_now = datetime.datetime(2024, 5, 24, 10, 55) # テスト用

start_date = datetime.date(dt_now.year, dt_now.month, dt_now.day)
end_date = datetime.date(dt_now.year, dt_now.month, dt_now.day + 1)
strStart = start_date.strftime('%m/%d/%Y %H:%M %p')
strEnd = end_date.strftime('%m/%d/%Y %H:%M %p')
sFilter = f"[Start] >= '{strStart}' And [End] <= '{strEnd}'"
# フィルターを適用し表示
FilteredItems = items.Restrict(sFilter)
for item in FilteredItems:
    if start_date <= item.start.date() <= end_date:
        # self.select_items.append(item)
        select_items.append(item)
        
# print("今日の予定の件数:", len(self.select_items))
print("今日の予定の件数:", len(select_items))
# for select_item in self.select_items:
margin = 5
margin = datetime.timedelta(minutes=margin)
for select_item in select_items:
    Contents_Start = select_item.Start.Format("%Y/%m/%d %H:%M")
    # print(Contents)
    date_time = datetime.datetime.strptime(Contents_Start, "%Y/%m/%d %H:%M")
    print(date_time)
    # try:
    if date_time -margin <= dt_now <= date_time +margin: # マージンなので、前後5分は含む
        isTrue = True
        print("True!!!!!")
    # except:
        # print("エラー")

print(type(date_time))