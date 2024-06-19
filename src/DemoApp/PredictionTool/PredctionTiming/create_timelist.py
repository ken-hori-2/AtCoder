# from datetime import datetime, timedelta

# # 全時刻のリストを生成（0:00から23:55まで5分刻み）
# all_times = [(datetime.min + timedelta(minutes=5) * i).strftime('%H:%M') for i in range(24 * 60 // 5)]

# # 全時刻のカウントを初期化（初期値は0）
# time_counts = {time: 0 for time in all_times}

# # times.txtから時刻を読み込む
# with open('times.txt', 'r', encoding='utf-8') as times_file:
#     times_list = [line.split(',')[0].strip() for line in times_file]

# # matching_times.txtから時刻を読み込む
# with open('matching_times.txt', 'r', encoding='utf-8') as matching_file:
#     matching_times = [line.split(',')[0].strip() for line in matching_file]


# print("times.txt -----")
# print(times_list)

# print("matching_times.txt -----")
# print(matching_times)
# # 一致する時刻のカウント
# for time in times_list:
#     if time in matching_times:
#         time_counts[time] += 1

# # カウント結果を新しいファイルに出力
# with open('time_counts.txt', 'w') as output_file:
#     for time in all_times:
#         output_file.write(f"{time}, count:{time_counts[time]}\n")

from datetime import datetime, timedelta

# 全時刻のリストを生成（0:00から23:55まで5分刻み）
all_times = [(datetime.min + timedelta(minutes=5) * i).strftime('%H:%M') for i in range(24 * 60 // 5)]

# 全時刻のカウントを初期化（初期値は0）
time_counts = {time: 0 for time in all_times}

# times.txtから時刻を読み込む
with open('times.txt', 'r', encoding='utf-8') as times_file:
    times_list = [line.split(',')[0].strip() for line in times_file]

# matching_times.txtから時刻を読み込む
with open('matching_times.txt', 'r', encoding='utf-8') as matching_file:
    matching_times = [line.split(',')[0].strip() for line in matching_file]

# 一致する時刻のカウント（重複を含む）
for time in times_list:
    time_counts[time] += matching_times.count(time)

print(times_list)
print("-----")
print(matching_times)

# カウント結果を新しいファイルに出力
with open('time_counts2.txt', 'w') as output_file:
    for time in all_times:
        # output_file.write(f"{time}, count:{time_counts.get(time, 0)}\n")
        output_file.write(f"{time}, {time_counts.get(time, 0)}\n")