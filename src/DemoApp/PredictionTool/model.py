import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

file_path = './userdata.txt' # 自分のデータセット
# file_path = './dataset.txt' # ランダムに生成したデータセット
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
# 時刻を分に変換する関数
def time_to_minutes(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return hours * 60 + minutes
# 行動状態とツール名を数値に変換する辞書
state_to_index = {"WALKING": 0, "STABLE": 1, "RUNNING": 2}
# tool_to_index = {"楽曲再生": 0, "会議情報": 1, "天気情報": 2, "レストラン検索": 3, "動画視聴": 4, "ニュース閲覧": 5}
tool_to_index = {"楽曲再生": 0, "会議情報": 1, "天気情報": 2, "レストラン検索": 3, "動画視聴": 4, "ニュース閲覧": 5, "経路検索":6}
print("#####")
# print([[time_to_minutes(time), state_to_index[state]] for time, state, _ in data])
print([[time, state] for time, state, _ in data])
print("#####")
# データを数値に変換
inputs = torch.tensor([[time_to_minutes(time), state_to_index[state]] for time, state, _ in data], dtype=torch.float)
# データの前処理
# 時刻データの正規化
inputs[:, 0] = inputs[:, 0] / inputs[:, 0].max()
# 行動状態のone-hotエンコーディング
states = torch.zeros(len(data), len(state_to_index))
for i, (_, state, _) in enumerate(data):
    states[i, state_to_index[state]] = 1
inputs = torch.cat((inputs[:, :1], states), dim=1)
targets = torch.tensor([tool_to_index[tool] for _, _, tool in data], dtype=torch.long)
print("targets: ", targets)
import copy
targets_copy = [tool for _, _, tool in data] # copy.deepcopy(targets)



# データセットの作成
dataset = TensorDataset(inputs, targets)

# データセットのサイズ
dataset_size = len(dataset)

# 訓練用と検証用のサイズを計算
train_size = int(dataset_size * 0.8)
val_size = dataset_size - train_size

# データセットをランダムに分割
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoaderの作成
batch_size = 4  # 例としてバッチサイズを4に設定
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# モデルの定義、最適化アルゴリズム、損失関数は前のコードと同様なので省略
# モデルの定義
class PredictionModel(nn.Module):
    def __init__(self):
        super(PredictionModel, self).__init__()
        self.fc1 = nn.Linear(4, 128) # 2, 128)  # 入力層の次元を調整
        self.fc2 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.5)  # ドロップアウト層を追加
        self.fc3 = nn.Linear(256, len(tool_to_index))
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# モデルのインスタンス化
model = PredictionModel()

# 損失関数と最適化手法
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 学習プロセス
# epochs = 1000
# epochs = 15000 # 10000
epochs = 1000 # 1 # 00
# match_count_list = []
prediction_list = []
for epoch in range(epochs):
    # 訓練フェーズ
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # 検証フェーズ
    model.eval()
    val_loss = 0.0
    # 正答率
    # match_count = 0
    total_correct = 0
    total_data_len = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            print("outputs:", outputs)
            print("- max:", outputs.max(1)[1]) # .max())
            # if outputs.max(1)[1] == targets:
            #     match_count += 1
            batch_size = len(targets)  # バッチサイズの確認
            for i in range(batch_size):  # データ一つずつループ,ミニバッチの中身出しきるまで
                total_data_len += 1  # 全データ数を集計
                # if outputs[i] == targets[i]:
                if outputs.max(1)[1][i] == targets[i]:
                    total_correct += 1 # 正解のデータ数を集計
                
                # add
                print("predictions: ", outputs.max(1)[1]) # predictions)
                predicted_tools = [list(tool_to_index.keys())[list(tool_to_index.values()).index(idx)] for idx in outputs.max(1)[1]]
                print("Predict: ", predicted_tools)
                prediction_list.append(predicted_tools)

    # エポックごとの損失の表示
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')

    # 早期停止のロジックをここに追加する場合があります


    # this(addに移動)
    # print("predictions: ", outputs.max(1)[1]) # predictions)
    # predicted_tools = [list(tool_to_index.keys())[list(tool_to_index.values()).index(idx)] for idx in outputs.max(1)[1]]
    # print("Predict: ", predicted_tools)
    # prediction_list.append(predicted_tools)

    accuracy = total_correct/total_data_len*100  # 予測精度の算出
    print("正解率: {}".format(accuracy))

import pprint
print("PredictionTool:") # , prediction_list)
pprint.pprint(prediction_list)
print("targets: ") # , targets_copy)
pprint.pprint(targets_copy)





# # 正答率
# match_count = 0
# for i in range(len(targets)):
#     if predictions.max(1)[1][i] == targets[i]:
#         # print(i, predictions.max(1)[1][i], targets[i])
#         # print("match!")
#         match_count += 1

# print("正答率：", match_count/len(targets))

# print("テスト：", match_count_list)