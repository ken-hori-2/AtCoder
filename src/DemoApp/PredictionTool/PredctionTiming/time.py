from datetime import datetime, timedelta

# 開始時刻と終了時刻を設定
# start_time = datetime.strptime("09:10", "%H:%M")
# end_time = datetime.strptime("17:45", "%H:%M")
start_time = datetime.strptime("00:00", "%H:%M")
end_time = datetime.strptime("23:59", "%H:%M")

# 現在時刻を開始時刻に設定
current_time = start_time

# # 終了時刻に達するまで5分ごとに時刻を出力
# while current_time <= end_time:
#     print(current_time.strftime("%H:%M"))
#     current_time += timedelta(minutes=5)
# # ファイルに書き込む
# with open('times.txt', 'w') as file:
#     current_time = start_time
#     while current_time <= end_time:
#         file.write(current_time.strftime("%H:%M") + '\n')
#         current_time += timedelta(minutes=5)
# ファイルに書き込む
with open('times.txt', 'w') as file:
    current_time = start_time
    no = 0  # 行番号を初期化
    while current_time <= end_time:
        # 時刻と行番号をファイルに書き込む
        # file.write(f"{current_time.strftime('%H:%M')}, NotUse\n") # No.{no}\n")
        file.write(f"{current_time.strftime('%H:%M')}, {no}\n")
        current_time += timedelta(minutes=5)
        # no += 1  # 行番号をインクリメント