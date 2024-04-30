import win32com.client
import datetime

# 調べたい日付範囲を定義
# start_date = datetime.date(2024, 4, 1)
# end_date = datetime.date(2024, 4, 30)


dt_now = datetime.datetime.now()
# dt_now = dt_now.strftime('%Y年%m月%d日')
# print("時間:", dt_now)
# start_date = datetime.date(2024, 4, 15)
# end_date = datetime.date(2024, 4, 19+1)
start_date = datetime.date(dt_now.year, dt_now.month, dt_now.day)
end_date = datetime.date(dt_now.year, dt_now.month, dt_now.day + 1)

# Outlookの予定表へのインタフェースオブジェクトを取得
Outlook = win32com.client.Dispatch("Outlook.Application").GetNameSpace("MAPI")
CalendarItems = Outlook.GetDefaultFolder(9).Items

# 定期的な予定の二番目以降の予定を検索に含める
CalendarItems.IncludeRecurrences = True

# 開始時間でソート
CalendarItems.Sort("[Start]")

# "mm/dd/yyyy HH:MM AM"の形式に変換し、フィルター文字列を作成
strStart = start_date.strftime('%m/%d/%Y %H:%M %p')
strEnd = end_date.strftime('%m/%d/%Y %H:%M %p')
sFilter = f"[Start] >= '{strStart}' And [End] <= '{strEnd}'"

# フィルターを適用し表示
FilteredItems = CalendarItems.Restrict(sFilter)
for item in FilteredItems:
	print(str(item.Start.Format("%Y/%m/%d %H:%M")) + " : " + str(item.Subject))


# select_items = [] # 指定した期間内の予定を入れるリスト
# for item in CalendarItems:
#     if start_date <= item.start.date() <= end_date:
#         select_items.append(item)

# for item in select_items:
#     print("件名：", item.subject)
#     print("場所：", item.location)
#     print("開始時刻：", str(item.Start.Format("%Y/%m/%d %H:%M")))
#     # print("本文：", select_item.body)
#     print("----")