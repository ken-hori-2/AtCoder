# # 一致する時刻のカウントを初期化
# matching_count = 0

# # times.txtから時刻を読み込む
# with open('times.txt', 'r') as times_file:
#     times_list = times_file.readlines()

# # matching_times.txtから時刻を読み込む
# with open('matching_times.txt', 'r') as matching_file:
#     matching_times_list = matching_file.readlines()

# # 一致する時刻をカウント
# for time in times_list:
#     # times.txtの行から時刻部分のみを抽出（例: "09:10, No.0\n" -> "09:10"）
#     time = time.split(',')[0].strip()
#     # matching_times.txtの各行と比較
#     for matching_time in matching_times_list:
#         matching_time = matching_time.strip()  # 余分な空白や改行を削除
#         if time == matching_time:
#             matching_count += 1

# # 一致する時刻の数を出力
# print(f"Matching times count: {matching_count}")

# 一致する時刻のカウントを初期化
matching_count = 0

# times.txtから時刻を読み込む
# with open(file_path, 'r', encoding='utf-8') as file:

with open('times.txt', 'r', encoding='utf-8') as times_file:
    times_list = [line.split(',')[0].strip() for line in times_file]

# matching_times.txtから時刻を読み込む
with open('matching_times.txt', 'r', encoding='utf-8') as matching_file:
    matching_times_list = [line.split(',')[0].strip() for line in matching_file]

# # 一致する時刻をカウント
# for time in times_list:
#     if time in matching_times_list:
#         matching_count += 1

# # 一致する時刻の数を出力
# print(f"Matching times count: {matching_count}")

# 一致する時刻のカウントを格納する辞書を初期化
matching_counts = {time: 0 for time in matching_times_list}

# 一致する時刻のカウント
for matching_time in matching_times_list:
    if matching_time in times_list: # times_data:
        matching_counts[matching_time] += 1

# カウント結果を新しいファイルに出力
with open('matching_counts.txt', 'w') as output_file:
    for time, count in matching_counts.items():
        output_file.write(f"{time}, count:{count}\n")