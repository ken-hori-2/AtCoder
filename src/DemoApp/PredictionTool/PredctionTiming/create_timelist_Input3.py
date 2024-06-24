from datetime import datetime, timedelta

# 全時刻のリストを生成（0:00から23:55まで5分刻み）
all_times = [(datetime.min + timedelta(minutes=5) * i).strftime('%H:%M') for i in range(24 * 60 // 5)]

# 全時刻のカウントを初期化（初期値は0）
time_counts = {time: 0 for time in all_times}

# state_list = {time: -1 for time in all_times}
state_list = {time: "OTHER" for time in all_times}

#####
# match_state = [state[time] if time in time_data else "None" for time in all_times]

print("*****")
print(state_list)
print("*****")

# times.txtから時刻を読み込む
with open('times.txt', 'r', encoding='utf-8') as times_file:
    times_list = [line.split(',')[0].strip() for line in times_file]

# matching_times.txtから時刻を読み込む
with open('matching_times.txt', 'r', encoding='utf-8') as matching_file:
    matching_times = [line.split(', ')[0].strip() for line in matching_file]
    # matching_states = [line.split(', ')[1].strip() for line in matching_file]
    # matching_times, matching_states, _ = [line.strip().split(', ') for line in matching_file]

file_path = './matching_times.txt' # 自分のデータセット

data = {} # []
time_data = []
state_data = []
# ファイルを開いて各行を読み込む
# with open(file_path, 'r', encoding='utf-8') as file:
#     for line in file:
#         print(line)
#         # 各行をカンマで分割して時刻とカウントを取得
#         time_str, state_str = line.strip().split(', ')
#         # タプルとしてリストに追加
#         data.append((time_str, state_str))
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 改行文字を削除し、カンマで分割
        parts = line.strip().split(', ')
        # 分割したデータをタプルに変換してリストに追加
        if len(parts) == 3:  # データが3つの部分に分割されていることを確認
            # data.append((parts[0], parts[1])) # , parts[2]))
            # data += {parts[0]:parts[1]}
            # data[parts[0]] = parts[1]
            state_list[parts[0]] = parts[1]
            time_data.append((parts[0]))
            state_data.append((parts[1]))
# print(data) # [0])
# data = [[time, state] for time, state, count in data]
print("#####")
print(state_list)
print("#####")
# print(time_data)
# print(state_data)
# # print("data[time]:", data[time])
# # print(matching_file)
print("test:", matching_times)
# print("test:", matching_states)
# 一致する時刻のカウント（重複を含む）
# i = 0
for time in times_list:
    # print("time:", time)
    # print("data[time]:", state_data[time])
    time_counts[time] += matching_times.count(time)
    # # state_list[time] += matching_states.count(time)
    # if time in time_data:
    #     print("match:", time)
    #     state_list[time] = data[time] # time] # matching_states[time]
    # # i += 1

for time, state in zip(times_list, state_list):
    print("time:", time)
    if time in time_data:
        print("match:", time, state_list[time])
        state_list[time] = state_list[time] # data[time]

print(times_list)
print("-----")
print(matching_times)
print("-----")
# print(matching_states)
print(state_list)

# カウント結果を新しいファイルに出力
with open('time_counts2_input3.txt', 'w') as output_file:
    for time in all_times:
        # output_file.write(f"{time}, count:{time_counts.get(time, 0)}\n")
        output_file.write(f"{time}, {state_list.get(time, 1)}, {time_counts.get(time, 0)}\n")