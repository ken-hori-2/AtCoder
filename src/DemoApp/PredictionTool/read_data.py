# ファイルパスを指定
# file_path = './userdata.txt'
file_path = './dataset.txt'

# 空のリストを作成
data = []

# ファイルを開いて各行を読み込む
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 改行文字を削除し、カンマで分割
        parts = line.strip().split(', ')
        # 分割したデータをタプルに変換してリストに追加
        if len(parts) == 3:  # データが3つの部分に分割されていることを確認
            data.append((parts[0], parts[1], parts[2]))

# 結果を表示
for entry in data:
    print(entry)