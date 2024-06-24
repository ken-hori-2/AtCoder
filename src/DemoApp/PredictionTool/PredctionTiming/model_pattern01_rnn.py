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
file_path = './time_counts2_input3.txt' # 自分のデータセット

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
        time_str, state_str, count_str = line.strip().split(', ')
        # タプルとしてリストに追加
        data.append((time_str, state_str, count_str))
#####

# 時刻を分に変換する関数
def time_to_minutes(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return hours * 60 + minutes


state_to_index = {"STABLE": 0, "WALKING": 1, "RUNNING": 2, "OTHER": -1}
print("#####")
print("inputs [time, state]")
print([[time, state, count] for time, state, count in data])
print("#####")

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
# # count_to_index = {"0": 0, "1": 1, "2": 2}
# inputs_times = torch.tensor([time_to_minutes(t) for t, _ in data], dtype=torch.float).unsqueeze(1)
# usage_counts = torch.tensor([int(c) for _, c in data], dtype=torch.float).unsqueeze(1)

# データを数値に変換
inputs = torch.tensor([[time_to_minutes(time), state_to_index[state], int(count)] for time, state, count in data], dtype=torch.float)
print("*****")
print("in:", inputs)
print("*****")
print("in [:, 0]:", inputs[:, 0])
print("in max:", inputs[:, 0].max())
test_input_max = inputs[:, 0].max()

# データの前処理
# 時刻データの正規化
inputs[:, 0] = inputs[:, 0] / inputs[:, 0].max()
print("in [:, 0]:", inputs[:, 0])
print("*****")

print("行動状態をone-hotベクトル化 ... STABLE:[1,0,0], WALKING:[0,1,0], RUNNING:[0,0,1]")


print("recommend data (inputs[:, 2]:)", inputs[:, 2])

# recommend_labels = torch.tensor([1 if c >= 1 else 0 for c in inputs[:, 2]], dtype=torch.float) # .unsqueeze(1) # リコメンドするタイミングを二値で表現する（ここでは仮に2回以上の使用でリコメンド）

# 時刻データの正規化
# input_max = inputs_times[:, 0].max()

# inputs_times[:, 0] = inputs_times[:, 0] / input_max # inputs_times[:, 0].max()
# print("in [:, 0]:", inputs_times[:, 0])
# print("*****")
# features_tensor = inputs_times





# labels_tensor = torch.tensor(recommend_labels).unsqueeze(1)
# labels_tensor = recommend_labels


# 行動状態のone-hotエンコーディング
states = torch.zeros(len(data), len(state_to_index))
for i, (_, state, _) in enumerate(data):
    states[i, state_to_index[state]] = 1
print("state: ", states)
print("inputs[:, :1] : ", inputs[:, :1]) # 全ての行, 0列目まで
print("inputs[:, :2] : ", inputs[:, :2]) # 全ての行, 1列目まで
print("inputs[:, :3] : ", inputs[:, :3]) # 全ての行, 2列目まで
print("inputs[:, :5] : ", inputs[:, 2:3]) # 全ての行, 2列目のみ
inputs = torch.cat((inputs[:, :1], states, inputs[:, 2:3]), dim=1) # timeとstatesを連結する

# targets = torch.tensor([tool_to_index[tool] for _, _, tool in data], dtype=torch.long)

# recommend_labels = torch.tensor([1 if c >= 1 else 0 for c in inputs[:, 2]], dtype=torch.float) # .unsqueeze(1) # リコメンドするタイミングを二値で表現する（ここでは仮に2回以上の使用でリコメンド）
# ラベル用（学習時に使用）
usage_counts = torch.tensor([int(c) for _, _, c in data], dtype=torch.float) # .unsqueeze(1)
recommend_labels = torch.tensor([1 if c >= 1 else 0 for c in usage_counts], dtype=torch.float) # .unsqueeze(1)
targets = recommend_labels







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
# # データセットの作成
# dataset = TensorDataset(features_tensor, labels_tensor) # inputs, targets)
# print("*****")
# print("inputs:", features_tensor) # inputs)
# print("targets:", labels_tensor) # targets)
# print("*****")
# データセットの作成
dataset = TensorDataset(inputs, targets)
print("*****")
print("inputs:", inputs)
print("targets:", targets)
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
        # Input : time + state(one-hot) + count = 6
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)  # ドロップアウト層を追加
    
    def forward(self, x):
        # x = x.reshape(x.shape[0], -1)
        # x = x.unsqueeze(1)
        # x = x.view(x.size(0), -1)  # フラット化

        x = self.fc1(x)
        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)
        # return x
        return torch.sigmoid(x)

class CNNNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CNNNet, self).__init__()
        # self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=2, stride=1, padding=1) # In:1*4 *64=256?
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=2, stride=1, padding=1)
        """
        Lout = [ (Lin + 2×padding − dilation×(kernel_size−1) − 1) / stride) + 1 ]
        今回の場合:
            Lout = (1  + 2*1       - 0*(2-1)-1)/1 + 1
                 = 3 - 1 + 1
                 = 3
        つまり、Affineに入力されるのは、「3 * 64」 
        """
        # # self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1) # パディングも指定すると、in_feature=3になる
        self.pool = nn.MaxPool1d(kernel_size=2) # 1*2 *64=128(カーネルのみの指定だと、padding=0...4/2=2(in_feature=2)で、2*64=128になる)
        # """
        # # padding=1の場合
        # Lout = (3+2-0-1)/2+1 = 3 ... in_feature=3

        # # padding=0の場合(指定しない場合、デフォルトで0)
        # Lout = (3+0-0-1)/2+1 = 2 ... in_feature=2
        # """
        # self.fc1 = nn.Linear(in_features=2*64, out_features=128) # 2[input] * 64[conv1のout_channels]
        # # self.fc1 = nn.Linear(in_features=4, out_features=3*64) # 2[input] * 64[conv1のout_channels]
        # """
        # # 今回 ... おそらく、3[input] * 64[conv1のout_channels] = 192
        # #   > それが4列ずつ(バッチサイズ4)モデルに入力される。
        # #       4[batch] * 3[input]*64[conv1のoutput_channels] = 4*192
        
        # # 前回 ... 8*8[input] * 128[conv1のout_channels]
        # """
        # self.dropout = nn.Dropout(0.5)  # ドロップアウト層を追加
        # self.fc2 = nn.Linear(128, 256)
        # # self.fc2 = nn.Linear(3*64, 256)
        # self.fc3 = nn.Linear(256, output_dim)
        # # self.bn1 = nn.BatchNorm1d(128) # 64)
        # # self.bn2 = nn.BatchNorm1d(256) # 64)

        """ 2024/6/24 """
        # self.fc1 = nn.Linear(3*64, 256) # 128) # hidden_dim)
        # self.relu = nn.ReLU(inplace=True)
        # self.fc2 = nn.Linear(256, output_dim) # hidden_dim, output_dim)
        # self.dropout = nn.Dropout(0.5)  # ドロップアウト層を追加
        
        # self.fc1 = nn.Linear(3*64, 128) # hidden_dim)
        # self.relu = nn.ReLU(inplace=True)
        # self.fc2 = nn.Linear(128, 256) # hidden_dim, output_dim)
        # self.dropout = nn.Dropout(0.5)  # ドロップアウト層を追加
        # self.fc3 = nn.Linear(256, output_dim)
        # self.bn1 = nn.BatchNorm1d(128) # 64)
        # self.bn2 = nn.BatchNorm1d(256) # 64)

        self.fc1 = nn.Linear(3*16, 64) # hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        # self.fc2 = nn.Linear(64, 128) # hidden_dim, output_dim)
        self.fc2 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(0.5)  # ドロップアウト層を追加
        # self.fc3 = nn.Linear(128, output_dim)
        self.bn1 = nn.BatchNorm1d(64)
        # self.bn2 = nn.BatchNorm1d(128)
        """ 2024/6/24 """
        self.sigmoid = nn.Sigmoid() # inplace=True)

    def forward(self, x):
        # 畳み込み層は入力が(batch_size, channels, length)の形であることを期待するため、入力xを適切な形に変形する
        x = x.unsqueeze(1)  # (batch_size, 1, length)
        # x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # フラット化
        # print(x)
        
        """ 2024/6/24 """
        # x = torch.relu(self.fc1(x))
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        # x = self.relu(self.bn2(self.fc2(x)))
        # x = self.fc3(x)
        return x
        return self.sigmoid(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x)
        """ 2024/6/24 """


class My_rnn_net(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(My_rnn_net, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

        self.sigmoid = nn.Sigmoid() # inplace=True)

    def forward(self, x):
        # xのサイズを (batch_size, sequence_length, input_size) に整形
        x = x.view(x.size(0), -1, self.input_size)
        
        y_rnn, h = self.rnn(x, None)
        y = self.fc(y_rnn[:, -1, :])

        return y
        # return self.sigmoid(y)






# モデルのインスタンス化
input_dim = 6 # 4 # 5 # 3 # 2  # 時刻とアプリ使用回数の2つ
# input_dim = 1  # 時刻のみ
hidden_dim = 16  # 隠れ層のサイズ
output_dim = 1  # リコメンドするかしないか
# model = RecommendationModel(input_dim, hidden_dim, output_dim)
model = CNNNet(input_dim, hidden_dim, output_dim)


# # テスト用の入力データ
# batch_size = 4 # 10
# sequence_length = 5
# input_size = 6  # ここをモデルのinput_sizeと一致させる
# output_size = 1
# hidden_dim = 20
# n_layers = 2
# model = My_rnn_net(input_size, output_size, hidden_dim, n_layers) # RNNをインスタンス化


#### ステップ4: モデルの学習

# モデルを訓練します。


# 損失関数とオプティマイザの定義
# criterion = nn.BCEWithLogitsLoss()  # 二値分類の場合 # binary_cross_entropy_with_logits
# optimizer = optim.Adam(model.parameters(), lr=0.001) # 0.01)
# # optimizer = optim.RMSprop(model.parameters(), lr=0.01)

# CNN
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# RNN
# 損失関数と最適化アルゴリズム
criterion = nn.MSELoss()
# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) # 01)

# トレーニングループ
num_epochs = 100 # 300 # 100 # 00 # 100 # 0 # 100  # エポック数
threshold = 0.5 # 0.8 # 0.5

print("\n\n\n#####\nTRAIN\n#####")
# # 訓練フェーズ
model.train()
losses = [] # 損失
accs = [] # 精度
losses_val = [] # 損失
accs_val = [] # 精度
for epoch in range(num_epochs):
    # 訓練フェーズ
    # model.train()
    train_loss = 0.0
    "----- E資格 -----"
    total_correct = 0
    total_data_len = 0
    "-----------------"

    # for features, labels in train_loader: # dataloader:
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs) # features)
        targets = targets.unsqueeze(1)
        loss = criterion(outputs, targets) # labels)
        
        
        
        # CNN
        # predictions = (outputs > threshold).float().numpy()  # 閾値0.5で分類
        # RNN
        predictions_prob = torch.sigmoid(outputs).detach().numpy()  # シグモイド関数で0-1の範囲にスケール
        predictions = (predictions_prob > threshold)  # 閾値0.5で分類


        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        batch_size = len(targets) # labels) # targets)  # バッチサイズの確認

        
        for i in range(batch_size):  # データ一つずつループ,ミニバッチの中身出しきるまで
            total_data_len += 1  # 全データ数を集計
            if (predictions[i]) == int(targets[i].item()): # targets[i]: # int(labels[i].item()):
                total_correct += 1 # 正解のデータ数を集計
    # 可視化用に追加
    train_accuracy = total_correct/total_data_len*100  # 予測精度の算出
    # train_loss/=total_data_len  # 損失の平均の算出
    ave_train_loss = train_loss / total_data_len # 損失の平均の算出
    # losses.append(train_loss)
    losses.append(ave_train_loss)
    accs.append(train_accuracy)

    if (epoch+1) % 10 == 0:
        print("correct:", total_correct)
        print("total:", total_data_len)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss(average): {ave_train_loss}')
        # train_accuracy = total_correct/total_data_len*100  # 予測精度の算出
        print("正解率: {}".format(train_accuracy))
        print("**********")


#### ステップ5: モデルの評価

# 学習したモデルを使って予測を行い、評価します。


print("\n\n\n#####\nVAL\n#####")
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

    for inputs, targets in val_loader:
        outputs = model(inputs)

        targets = targets.unsqueeze(1)

        loss = criterion(outputs, targets)
        val_loss += loss.item()

        # CNN
        # predictions = (outputs > threshold).float().numpy()  # 閾値0.5で分類
        # RNN
        predictions_prob = torch.sigmoid(outputs).detach().numpy()  # シグモイド関数で0-1の範囲にスケール
        predictions = (predictions_prob > threshold)  # 閾値0.5で分類

        batch_size = len(targets)  # バッチサイズの確認
        for i in range(batch_size):  # データ一つずつループ,ミニバッチの中身出しきるまで
            total_data_len_val += 1  # 全データ数を集計
            # if predictions[i] == targets[i]:
            if (predictions[i]) == int(targets[i].item()):
                total_correct_val += 1 # 正解のデータ数を集計

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
plt.savefig('result_20240620.png')








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
    ('09:10', "WALKING", 0),
    ('10:30', "WALKING", 0),
    ('12:30', "STABLE", 0),
    ('15:00', "OTHER", 0),
    ('15:00', "WALKING", 3),
    ('18:00', "WALKING", 0),
    ('20:00', "RUNNING", 0),

    ('09:10', "WALKING", 1),
    ('10:30', "WALKING", 1),
    ('12:30', "STABLE", 1),
    ('15:00', "WALKING", 1),
    ('18:00', "WALKING", 1),
    ('20:00', "RUNNING", 1),

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
    ('0:00', "OTHER", 0),
    ('6:00', "OTHER", 0),
    ('22:00', "OTHER", 0),
    ('14:00', "OTHER", 1),
    ('7:00', "OTHER", 0),

]


inputs = torch.tensor([[time_to_minutes(time), state_to_index[state], int(count)] for time, state, count in test_data], dtype=torch.float)

test_usage_counts = torch.tensor([int(c) for _, _, c in test_data], dtype=torch.float) # .unsqueeze(1)
test_recommend_labels = torch.tensor([1 if c >= 1 else 0 for c in test_usage_counts], dtype=torch.float) # .unsqueeze(1)
targets = test_recommend_labels


inputs[:, 0] = inputs[:, 0] / test_input_max # input_max # inputs[:, 0].max()

# print("in [:, 0]:", inputs[:, 0])
# print("*****")
print("行動状態をone-hotベクトル化 ... STABLE:[1,0,0,0], WALKING:[0,1,0,0], RUNNING:[0,0,1,0], OTHER:[0,0,0,1]")
# 行動状態のone-hotエンコーディング
states = torch.zeros(len(test_data), len(state_to_index))
# states = torch.zeros(len(data), len(state_to_index))

for i, (_, state, _) in enumerate(test_data):
    states[i, state_to_index[state]] = 1
print("state: ", states)
inputs = torch.cat((inputs[:, :1], states, inputs[:, 2:3]), dim=1) # timeとstatesを連結する


# データセットの作成
test_dataset = TensorDataset(inputs, targets)
print("*****")
print("inputs:", inputs)
print("targets:", targets)
print("*****")
# DataLoaderの作成
batch_size = len(test_dataset) # 1 # 4  # 例としてバッチサイズを4に設定
# batch_size = 32
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True) # False) # True) # train,valと一緒に作成する方がいいかも
# val_loader = DataLoader(val_dataset, batch_size=batch_size)


# outputs = model(inputs)

# 評価のためにモデルを推論モードに切り替え
model.eval()
test_loss = 0.0
"----- E資格 -----"
total_correct_test = 0
total_data_len_test = 0
"-----------------"
losses_test = [] # 損失
accs_test = [] # 精度
# テストデータでの予測（例: トレーニングデータと同じデータを使う）
with torch.no_grad():

    for inputs, targets in test_loader:
        outputs = model(inputs)
        predictions_prob = outputs
        print("probabilities:", predictions_prob*100)

        targets = targets.unsqueeze(1)

        loss = criterion(outputs, targets)
        test_loss += loss.item()

        # CNN
        # predictions = (outputs > threshold).float().numpy()  # 閾値0.5で分類
        # RNN
        predictions_prob = torch.sigmoid(outputs).detach().numpy()  # シグモイド関数で0-1の範囲にスケール
        predictions = (predictions_prob > threshold)  # 閾値0.5で分類

        # print("probabilities:", predictions_prob)
        print("targets classes:", targets.float().numpy())
        print("predicts classes:", predictions)

        batch_size = len(targets)  # バッチサイズの確認
        for i in range(batch_size):  # データ一つずつループ,ミニバッチの中身出しきるまで
            total_data_len_test += 1  # 全データ数を集計
            # if predictions[i] == targets[i]:
            if (predictions[i]) == int(targets[i].item()):
                total_correct_test += 1 # 正解のデータ数を集計
        # # ラベルが1の物のみを対象とする場合
        # for i in range(batch_size):  # データ一つずつループ,ミニバッチの中身出しきるまで
        #     if int(targets[i].item()) == 1:
        #         total_data_len_test += 1  # 全データ数を集計
        #         if (predictions[i]) == int(targets[i].item()):
        #             total_correct_test += 1 # 正解のデータ数を集計

test_accuracy = total_correct_test/total_data_len_test*100  # 予測精度の算出
ave_test_loss = test_loss / total_data_len_test # 損失の平均の算出
losses_test.append(ave_test_loss)
accs_test.append(test_accuracy)
print("正解率: {}".format(test_accuracy))
print("**********")

# CNN
# print(len(predictions_prob), predictions_prob.detach().numpy())

# epochs = range(len(predictions_prob))
# thr = np.arange(0, len(predictions_prob))
# fig = plt.figure(figsize=(15, 8))
# fig.suptitle("prediction prob")
# plt.plot(epochs, predictions_prob.detach().numpy(), label="predictions prob [0.0-1.0]") # , color = "green")
# plt.legend()
# plt.plot(epochs, targets.detach().numpy(), label="targets [0, 1]") # , color = "green")
# plt.legend()
# # plt.plot(epochs, 0.8, label="thr [0.8]")
# plt.hlines(y=threshold, xmin=0, xmax=len(predictions_prob)-1, ls="-.", label=f"thr [{threshold}]", color="magenta")
# plt.legend()
# plt.show()

# RNN
print(len(predictions_prob), predictions_prob)
epochs = range(len(predictions_prob))
thr = np.arange(0, len(predictions_prob))
fig = plt.figure(figsize=(15, 8))
fig.suptitle("prediction prob")
plt.plot(epochs, predictions_prob, label="predictions prob [0.0-1.0]") # , color = "green")
plt.legend()
plt.plot(epochs, targets.detach().numpy(), label="targets [0, 1]") # , color = "green")
plt.legend()
# plt.plot(epochs, 0.8, label="thr [0.8]")
plt.hlines(y=threshold, xmin=0, xmax=len(predictions_prob)-1, ls="-.", label=f"thr [{threshold}]", color="magenta")
plt.legend()
plt.show()