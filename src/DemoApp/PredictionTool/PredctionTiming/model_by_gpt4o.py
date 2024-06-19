import numpy as np
# import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F
import pprint

# 小数表記にする(指数表記を使わない)
torch.set_printoptions(sci_mode=False) # pytorch
np.set_printoptions(suppress=True) # numpy

#####
# 入力データ（時刻, アプリ使用回数）
# data = [
#     ('9:10', 1),
#     ('10:30', 0),
#     ('12:30', 5),
#     ('15:00', 2),
#     ('18:00', 1),
#     ('20:00', 5),

#     ('7:30', 2),
#     ('11:55', 3),
#     ('12:45', 5),
#     ('19:00', 2),
#     ('20:00', 5),

#     ('7:15', 1),
#     ('12:10', 0),
#     ('12:30', 5),
#     ('19:20', 2),
#     ('20:00', 5),

#     ('7:30', 1),
#     ('12:50', 5),
#     ('19:50', 2),
#     ('20:00', 5),

#     ('8:00', 3),
#     ('8:45', 0),
#     ('12:05', 5),
#     ('17:30', 2),
#     ('18:00', 1),
# ]
# or
# file_path = './time_counts.txt' # 自分のデータセット
file_path = './time_counts2.txt' # 自分のデータセット

data = []
# ファイルを開いて各行を読み込む
with open(file_path, 'r', encoding='utf-8') as file:
#     for line in file:
#         # 改行文字を削除し、カンマで分割
#         parts = line.strip().split(', ')
#         # 分割したデータをタプルに変換してリストに追加
#         if len(parts) == 2:  # データが2つの部分に分割されていることを確認
#             data.append((parts[0], parts[1]))
    for line in file:
        # 各行をカンマで分割して時刻とカウントを取得
        time_str, count_str = line.strip().split(', ')
        # タプルとしてリストに追加
        data.append((time_str, count_str))
#####

# # 時刻を24時間形式の数値に変換する関数
# def time_to_float(time_str):
#     hours, minutes = map(int, time_str.split(':'))
#     return hours + minutes / 60.0
# 時刻を分に変換する関数
def time_to_minutes(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return hours * 60 + minutes

# # データの前処理
# # times = np.array([time_to_float(t) for t, _ in data], dtype=np.float32)
# times = np.array([time_to_minutes(t) for t, _ in data], dtype=np.float32)
# usage_counts = np.array([c for _, c in data], dtype=np.float32)

"""
# リコメンドするタイミングを二値で表現する（ここでは仮に2回以上の使用でリコメンド）
"""
# # recommend_labels = np.array([1 if c >= 2 else 0 for c in usage_counts], dtype=np.float32)
# # 1回以上の使用でリコメンド
# recommend_labels = np.array([1 if c >= 1 else 0 for c in usage_counts], dtype=np.float32)




# # データのテンソル化
# times_tensor = torch.tensor(times).unsqueeze(1)
# usage_tensor = torch.tensor(usage_counts).unsqueeze(1)
# # features_tensor = torch.cat((times_tensor, usage_tensor), dim=1)
# features_tensor = times_tensor


"""
# データの前処理
# データのテンソル化
をまとめる
"""
# count_to_index = {"0": 0, "1": 1, "2": 2}
inputs_times = torch.tensor([time_to_minutes(t) for t, _ in data], dtype=torch.float).unsqueeze(1)
usage_counts = torch.tensor([int(c) for _, c in data], dtype=torch.float).unsqueeze(1)
recommend_labels = torch.tensor([1 if c >= 1 else 0 for c in usage_counts], dtype=torch.float).unsqueeze(1) # リコメンドするタイミングを二値で表現する（ここでは仮に2回以上の使用でリコメンド）

# 時刻データの正規化
input_max = inputs_times[:, 0].max()

inputs_times[:, 0] = inputs_times[:, 0] / input_max # inputs_times[:, 0].max()
print("in [:, 0]:", inputs_times[:, 0])
print("*****")
features_tensor = inputs_times





# labels_tensor = torch.tensor(recommend_labels).unsqueeze(1)
labels_tensor = recommend_labels

# print("features tensor")
# print(features_tensor)


#### ステップ2: データセットとデータローダーの定義
# # 次に、データをPyTorchのデータセット形式に変換し、データローダーを定義します。
# class AppUsageDataset(Dataset):
#     def __init__(self, features, labels):
#         self.features = features
#         self.labels = labels
#     def __len__(self):
#         return len(self.features)
#     def __getitem__(self, idx):
#         return self.features[idx], self.labels[idx]
# # データセットとデータローダーの作成
# dataset = AppUsageDataset(features_tensor, labels_tensor)
# # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# or
# データセットの作成
dataset = TensorDataset(features_tensor, labels_tensor) # inputs, targets)
print("*****")
print("inputs:", features_tensor) # inputs)
print("targets:", labels_tensor) # targets)
print("*****")
# データセットのサイズ
dataset_size = len(dataset)
# 訓練用と検証用のサイズを計算
train_size = int(dataset_size * 0.8)
val_size = dataset_size - train_size
# データセットをランダムに分割
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print("dataset:", dataset[0])
print("**********")
print("train datset:", train_dataset[0])
print("val dataset:", val_dataset[0])
print("**********")
# DataLoaderの作成
batch_size = 4  # 例としてバッチサイズを4に設定
# batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

#### ステップ2: データセットとデータローダーの定義




#### ステップ3: モデルの定義

# シンプルなニューラルネットワークモデルを定義します。


class RecommendationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RecommendationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)  # ドロップアウト層を追加
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)
        return x

# モデルのインスタンス化
# input_dim = 2  # 時刻とアプリ使用回数の2つ
input_dim = 1  # 時刻のみ
hidden_dim = 16  # 隠れ層のサイズ
output_dim = 1  # リコメンドするかしないか
model = RecommendationModel(input_dim, hidden_dim, output_dim)


#### ステップ4: モデルの学習

# モデルを訓練します。


# 損失関数とオプティマイザの定義
criterion = nn.BCEWithLogitsLoss()  # 二値分類の場合
optimizer = optim.Adam(model.parameters(), lr=0.01) # 0.001)
# # 損失関数と最適化アルゴリズム
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001) # 01)

# トレーニングループ
num_epochs = 10000 # 100 # 0 # 100  # エポック数

# # 訓練フェーズ
# model.train()
losses = [] # 損失
accs = [] # 精度
losses_val = [] # 損失
accs_val = [] # 精度
for epoch in range(num_epochs):
    # 訓練フェーズ
    model.train()
    train_loss = 0.0
    "----- E資格 -----"
    total_correct = 0
    total_data_len = 0
    "-----------------"

    for features, labels in train_loader: # dataloader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)

        # probabilities = torch.softmax(outputs, dim=1)  # 各クラスに属する確率を計算
        # predicted_classes = torch.argmax(probabilities, dim=1)  # 最も確率の高いクラスを予測
        # 二値分類なので、一つの確率のみ出力される
        predictions_prob = torch.sigmoid(outputs).detach().numpy()  # シグモイド関数で0-1の範囲にスケール
        predictions = (predictions_prob > 0.5).astype(int)  # 閾値0.5で分類
        # predictions = torch.tensor((predictions_prob > 0.5).astype(int))  # 閾値0.5で分類
        
        # print("prediction :", predictions) # predicted_classes)

        # print("-----")
        # # print("features:", features)
        # print("features:")
        # pprint.pprint(features)
        # # print("outputs:", outputs)
        # # print("prob:", predictions_prob) # probabilities)
        # # print("prediction:", predictions) # predicted_classes)
        # # print("lebels:", labels)
        # print("outputs:")
        # pprint.pprint(outputs)
        # print("prob:")
        # pprint.pprint(predictions_prob) # probabilities)
        # print("prediction:")
        # pprint.pprint(predictions) # predicted_classes)
        # print("lebels:")
        # pprint.pprint(labels)

        # print("-----")
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        batch_size = len(labels) # targets)  # バッチサイズの確認

        
        for i in range(batch_size):  # データ一つずつループ,ミニバッチの中身出しきるまで
            # # print("test:", torch.tensor(predictions[i]).item(), int(labels[i].item()))
            # print("test:", predictions[i], int(labels[i].item()))
            
            total_data_len += 1  # 全データ数を集計
            # if torch.tensor(predictions[i]).item() == int(labels[i].item()): # torch.tensor(labels[i]):
            if (predictions[i]) == int(labels[i].item()):
                total_correct += 1 # 正解のデータ数を集計
    # 可視化用に追加
    train_accuracy = total_correct/total_data_len*100  # 予測精度の算出
    # train_loss/=total_data_len  # 損失の平均の算出
    ave_train_loss = train_loss / total_data_len # 損失の平均の算出
    # losses.append(train_loss)
    losses.append(ave_train_loss)
    accs.append(train_accuracy)

    if (epoch+1) % 100 == 0:
        print("correct:", total_correct)
        print("total:", total_data_len)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss(average): {ave_train_loss}')
        # train_accuracy = total_correct/total_data_len*100  # 予測精度の算出
        print("正解率: {}".format(train_accuracy))
        print("**********")


#### ステップ5: モデルの評価

# 学習したモデルを使って予測を行い、評価します。


# 評価のためにモデルを推論モードに切り替え
model.eval()

# print("features tensor")
# print(features_tensor)
val_loss = 0.0
"----- E資格 -----"
total_correct_val = 0
total_data_len_val = 0
"-----------------"

# テストデータでの予測（例: トレーニングデータと同じデータを使う）
with torch.no_grad():
    # # predictions = model(features_tensor)
    # outputs = model(features_tensor)
    # # predictions_prob = torch.sigmoid(predictions).numpy()  # シグモイド関数で0-1の範囲にスケール
    # predictions_prob = torch.sigmoid(outputs).numpy()  # シグモイド関数で0-1の範囲にスケール
    # predictions = (predictions_prob > 0.5).astype(int)  # 閾値0.5で分類

    # print("predictions prob :", predictions_prob) # probabilities)
    # print("predictions      :", predictions) # predicted_classes)
    # print("lebels           :", labels)
    
    # print("予測結果:", predictions.flatten())
    # print("実際のラベル:", recommend_labels)

    for inputs, targets in val_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        val_loss += loss.item()

        predictions_prob = torch.sigmoid(outputs).numpy()  # シグモイド関数で0-1の範囲にスケール
        # predictions_prob = torch.softmax(outputs, dim=1) # .detach().numpy()
        
        predictions = (predictions_prob > 0.5).astype(int)  # 閾値0.5で分類
        batch_size = len(targets)  # バッチサイズの確認
        for i in range(batch_size):  # データ一つずつループ,ミニバッチの中身出しきるまで
            total_data_len_val += 1  # 全データ数を集計
            # if predictions[i] == targets[i]:
            if (predictions[i]) == int(targets[i].item()):
                total_correct_val += 1 # 正解のデータ数を集計

        # print("predictions prob :", predictions_prob) # probabilities)
        # print("predictions      :", predictions) # predicted_classes)
        # print("lebels           :", labels)
        
        # print("予測結果:", predictions.flatten())
        # print("実際のラベル:", recommend_labels)

val_accuracy = total_correct_val/total_data_len_val*100  # 予測精度の算出
# val_loss/=total_data_len_val  # 損失の平均の算出
ave_val_loss = val_loss / total_data_len_val # 損失の平均の算出

# losses_val.append(val_loss)
losses_val.append(ave_val_loss)
accs_val.append(val_accuracy)

# print(f'Epoch {epoch}/{epochs}, Val Loss(average): {ave_val_loss}')
# accuracy = total_correct_val/total_data_len_val*100  # 予測精度の算出
print("正解率: {}".format(val_accuracy))
print("**********")


print("*************************************************************************************************************")
print("予測結果  : {}".format(predictions))
print("正解ラベル: {}".format(targets))
print("*************************************************************************************************************")

epochs = range(len(accs))

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15, 8))
# add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
fig.suptitle("Train Result - loss & acc -")
# plt.style.use("ggplot")
ax1.plot(epochs, losses, label="train loss", color = "red") # "orange")
# ax1.plot(epochs, losses_val, label="val loss", color = "blue")
ax1.legend()
ax2.plot(epochs, accs, label="train accurucy", color = "green")
# ax2.plot(epochs, accs_val, label="val accurucy", color = "orange")
ax2.legend()
plt.savefig('result_20240618.png')








print("\n\n\n#####\nTEST\n#####")
# テスト
test_data = [

    # ('8:00',  ""),
    # # ('10:45', ""),
    # ('10:55', ""),
    # ('12:05', ""),
    # ('17:30', ""),
    # ('18:10', ""),
    # ('20:00', ""),
    # ('20:10', ""),
    ('09:10', ""),
    ('10:30', ""),
    ('12:30', ""),
    ('15:00', ""),
    ('18:00', ""),
    ('20:00', ""),

    # ('7:30', ""),
    # ('11:55', ""),
    # ('12:45', ""),
    # ('19:00', ""),
    # ('20:00', ""),

    # ('7:15', ""),
    # ('12:10', ""),
    # ('12:30', ""),
    # ('19:20', ""),
    # ('20:00', ""),

    # ('7:30', ""),
    # ('12:50', ""),
    # ('19:50', ""),
    # ('20:00', ""),

    # ('8:00', ""),
    # ('8:45', ""),
    # ('12:05', ""),
    # ('17:30', ""),
    # ('18:00', ""),
    ('0:00', ""),

]

# times = np.array([time_to_minutes(t) for t, _ in test_data], dtype=np.float32)
# # データを数値に変換
# # inputs = torch.tensor([[time_to_minutes(time), state_to_index[state]] for time, state, _ in test_data], dtype=torch.float)
# inputs = torch.tensor(times).unsqueeze(1) # times_tensor

# テストデータ
# inputs = torch.tensor([time_to_minutes(t) for t, _ in test_data], dtype=torch.float32).unsqueeze(1) # times_tensor
# 学習データ
inputs = torch.tensor([time_to_minutes(t) for t, _ in data], dtype=torch.float32).unsqueeze(1) # times_tensor

# print("*****")
# print("in:", inputs)
# print("*****")
# print("in [:, 0]:", inputs[:, 0])
# # print("in max:", inputs[:, 0].max())


# # データの前処理
# # 時刻データの正規化
# """
# 学習したサーのデータセットの最大値を使って正規化する必要がある
# 基準がデータセット(0～1)なのでこの基準にそろえる必要がある
# 現在時刻が学習データ全体のうちのどこら辺に位置するか？
# """
inputs[:, 0] = inputs[:, 0] / input_max # inputs[:, 0].max()

# print("in [:, 0]:", inputs[:, 0])
# print("*****")


outputs = model(inputs)

def minutes_to_time(time_str):
    # hours, minutes = map(int, time_str.split(':'))
    # return hours * 60 + minutes
    return time_str/60 # +time_str%60

# print("test inputs:", minutes_to_time(inputs))
print("times:", inputs)

predictions_prob = torch.sigmoid(outputs).detach().numpy()  # シグモイド関数で0-1の範囲にスケール
# predictions_prob = torch.softmax(outputs, dim=1).detach().numpy() # dim=0).detach().numpy()
print("probabilities:", predictions_prob*100)
predictions = (predictions_prob*100 > 0.5) # .astype(int)  # 閾値0.5で分類
# print("probabilities:", predictions_prob)
print("predict classes:", predictions)

# probabilities = torch.softmax(outputs, dim=1)  # 各クラスに属する確率を計算
# predicted_classes = (predictions_prob > 0.5) # torch.argmax(probabilities, dim=1)  # 最も確率の高いクラスを予測
# print("probabilities:", probabilities)
# print("predict classes:", predicted_classes)
print(len(predictions_prob), predictions_prob.ravel())
epochs = range(len(predictions_prob))
fig = plt.figure(figsize=(15, 8))
fig.suptitle("prediction prob")
plt.plot(epochs, predictions_prob.ravel(), label="predictions prob") # , color = "green")
plt.show()