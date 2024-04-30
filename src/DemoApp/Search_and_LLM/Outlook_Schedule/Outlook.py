import win32com.client

# outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
# calender = outlook.GetDefaultFolder(9) # 「9」というのがOutlookの予定表のこと


import datetime

# items = calender.Items # このitemsが一つ一つの「予定」

Outlook = win32com.client.Dispatch("Outlook.Application").GetNameSpace("MAPI")
items = Outlook.GetDefaultFolder(9).Items
# 定期的な予定の二番目以降の予定を検索に含める
items.IncludeRecurrences = True
# 開始時間でソート
items.Sort("[Start]")







select_items = [] # 指定した期間内の予定を入れるリスト

# 予定を抜き出したい期間を指定
# start_date = datetime.date(2024, 4, 15) # 2020-10-19
# end_date = datetime.date(2024, 4, 19) # 2020-10-23
dt_now = datetime.datetime.now()
start_date = datetime.date(dt_now.year, dt_now.month, dt_now.day)
end_date = datetime.date(dt_now.year, dt_now.month, dt_now.day + 1)

# "mm/dd/yyyy HH:MM AM"の形式に変換し、フィルター文字列を作成
strStart = start_date.strftime('%m/%d/%Y %H:%M %p')
strEnd = end_date.strftime('%m/%d/%Y %H:%M %p')
sFilter = f"[Start] >= '{strStart}' And [End] <= '{strEnd}'"

# フィルターを適用し表示
FilteredItems = items.Restrict(sFilter)

for item in FilteredItems:
    if start_date <= item.start.date() <= end_date:
        select_items.append(item)
        # print("item:", item)



# # "mm/dd/yyyy HH:MM AM"の形式に変換し、フィルター文字列を作成
# strStart = start_date.strftime('%m/%d/%Y %H:%M %p')
# strEnd = end_date.strftime('%m/%d/%Y %H:%M %p')
# sFilter = f"[Start] >= '{strStart}' And [End] <= '{strEnd}'"

# # フィルターを適用し表示
# FilteredItems = items.Restrict(sFilter)
# for item in FilteredItems:
# 	print(str(item.Start.Format("%Y/%m/%d %H:%M")) + " : " + str(item.Subject))


print("今日の予定の件数:", len(select_items))
# 抜き出した予定の詳細を表示

# ダミーデータ
i = 0
work = ['A社と会議', 'Bさんと面談', 'Cの開発定例', 'D社との商談', 'Eさんの資料のレビュー', 'Fチーム定例会', 'Gチーム戦略会議', 'H部定例', 'I課定例']
room = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
for select_item in select_items:
    print("件名：", select_item.subject)
    # print(f"件名：予定 {work[i]}")  # 社外秘情報は伏せる
    print("場所：", select_item.location)
    # print(f"場所：会議室 {room[i]}") # 社外秘情報は伏せる

    print("開始時刻：", str(select_item.Start.Format("%Y/%m/%d %H:%M")))
    print("終了時刻：", str(select_item.End.Format("%Y/%m/%d %H:%M")))
    # print("本文：", select_item.body)
    
    print("----")
    i += 1 # ダミーデータ用
