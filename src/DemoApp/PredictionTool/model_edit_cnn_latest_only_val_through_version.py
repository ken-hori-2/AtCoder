import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F
import numpy as np

# 小数表記にする(指数表記を使わない)
torch.set_printoptions(sci_mode=False) # pytorch
np.set_printoptions(suppress=True) # numpy

# file_path = './userdata.txt' # 自分のデータセット
file_path = './userdata_timeaction.txt' # 自分のデータセット
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
# state_to_index = {"WALKING": 0, "STABLE": 1, "RUNNING": 2}
state_to_index = {"STABLE": 0, "WALKING": 1, "RUNNING": 2}
# tool_to_index = {"楽曲再生": 0, "会議情報": 1, "天気情報": 2, "レストラン検索": 3, "動画視聴": 4, "ニュース閲覧": 5}
# tool_to_index = {"楽曲再生": 0, "会議情報": 1, "天気情報": 2, "レストラン検索": 3, "動画視聴": 4, "ニュース閲覧": 5, "経路検索":6}
tool_to_index = {"楽曲再生": 0, "会議情報": 1, "天気情報": 2, "レストラン検索": 3, "経路検索":4, "何もしない":5}

# print("#####")
# # print([[time_to_minutes(time), state_to_index[state]] for time, state, _ in data])
# print("inputs [time, state]")
# print([[time, state] for time, state, _ in data])
# print("#####")

# データを数値に変換
inputs = torch.tensor([[time_to_minutes(time), state_to_index[state]] for time, state, _ in data], dtype=torch.float)
# print("*****")
# print("in:", inputs)
# print("*****")
# print("in [:, 0]:", inputs[:, 0])
# print("in max:", inputs[:, 0].max())
test_input_max = inputs[:, 0].max()


# データの前処理
# 時刻データの正規化
inputs[:, 0] = inputs[:, 0] / inputs[:, 0].max()
print("in [:, 0]:", inputs[:, 0])
print("*****")

print("行動状態をone-hotベクトル化 ... STABLE:[1,0,0], WALKING:[0,1,0], RUNNING:[0,0,1]")

# 行動状態のone-hotエンコーディング
states = torch.zeros(len(data), len(state_to_index))
for i, (_, state, _) in enumerate(data):
    states[i, state_to_index[state]] = 1
print("state: ", states)
inputs = torch.cat((inputs[:, :1], states), dim=1) # timeとstatesを連結する
targets = torch.tensor([tool_to_index[tool] for _, _, tool in data], dtype=torch.long)
# print("targets: ", targets)

# import copy
targets_copy = [tool for _, _, tool in data] # copy.deepcopy(targets)
# print("targets: ", targets_copy)
# print("*****")



# データセットの作成
dataset = TensorDataset(inputs, targets)
# print("*****")
# print("inputs:", inputs)
# print("targets:", targets)
# print("*****")

# データセットのサイズ
dataset_size = len(dataset)

# 訓練用と検証用のサイズを計算
# train_size = int(dataset_size * 0.8)
""" テスト (9割val用に使う) """
train_size = int(dataset_size * 0.1)
""" テスト (9割val用に使う) """
val_size = dataset_size - train_size

# データセットをランダムに分割
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# """ テスト """
# # val_dataset, train_dataset = random_split(dataset, [train_size, val_size])
# """ テスト """

# print("dataset:", dataset[0])
# print("**********")
# print("train datset:", train_dataset[0])
# print("val dataset:", val_dataset[0])
# print("**********")

# DataLoaderの作成
batch_size = 4  # 例としてバッチサイズを4に設定
# batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)



# CNNモデルのアーキテクチャ
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=2, stride=1, padding=1) # In:1*4 *64=256?
        """
        Lout = [ (Lin + 2×padding − dilation×(kernel_size−1) − 1) / stride) + 1 ]
        今回の場合:
            Lout = (1  + 2*1       - 0*(2-1)-1)/1 + 1
                 = 3 - 1 + 1
                 = 3
        つまり、Affineに入力されるのは、「3 * 64」 
        """

        # self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1) # パディングも指定すると、in_feature=3になる
        self.pool = nn.MaxPool1d(kernel_size=2) # 1*2 *64=128(カーネルのみの指定だと、padding=0...4/2=2(in_feature=2)で、2*64=128になる)
        """
        # padding=1の場合
        Lout = (3+2-0-1)/2+1 = 3 ... in_feature=3

        # padding=0の場合(指定しない場合、デフォルトで0)
        Lout = (3+0-0-1)/2+1 = 2 ... in_feature=2
        """

        # MaxPool1dでパディングも指定すると、in_feature=3になる
        # # self.fc1 = nn.Linear(in_features=3*64, out_features=128) # 3[input] * 64[conv1のout_channels]
        # MaxPool1dでカーネルのみを指定すると、in_feature=2になる
        self.fc1 = nn.Linear(in_features=2*64, out_features=128) # 2[input] * 64[conv1のout_channels]
        """
        # 今回 ... おそらく、3[input] * 64[conv1のout_channels] = 192
        #   > それが4列ずつ(バッチサイズ4)モデルに入力される。
        #       4[batch] * 3[input]*64[conv1のoutput_channels] = 4*192
        
        # 前回 ... 8*8[input] * 128[conv1のout_channels]
        """

        # *** pre ***
        # self.fc2 = nn.Linear(in_features=128, out_features=len(tool_to_index))
        # *** pre ***
        self.dropout = nn.Dropout(0.5)  # ドロップアウト層を追加

        # *** add ***
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, len(tool_to_index))
        # self.softmax = nn.Softmax(dim=1) # 行単位で0~1に収める
        # *** add ***

        self.bn1 = nn.BatchNorm1d(128) # 64)
        self.bn2 = nn.BatchNorm1d(256) # 64)

    def forward(self, x):
        # 畳み込み層は入力が(batch_size, channels, length)の形であることを期待するため、
        # 入力xを適切な形に変形する
        x = x.unsqueeze(1)  # (batch_size, 1, length)
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # フラット化
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        # *** pre ***
        # x = self.fc2(x)
        # *** pre ***
        
        # *** add ***
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        # x = self.softmax(x)
        # *** add ***
        return x

# モデルのインスタンス化
model = CNNNet()
""" CNN version """

# 損失関数と最適化手法
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)
learning_rate = 0.001
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4) # 重み付けが大きくなりすぎないようにする









def Load_Trained_Model(model):
    print("***** validation *****")
    # パラメータの読み込み
    # param_load = torch.load("model_5000epoch.param")
    # param_load = torch.load("model_latest_2500epoch.param") # 学習時の精度が高いモデル
    # param_load = torch.load("model_latest_2800epoch.param") # 学習時の精度が高いモデル
    param_load = torch.load("model_20240701_through_2.param") # 学習時の精度が高いモデル


    model.load_state_dict(param_load)
    # validation(validation_loader, model)
    
    return model

# 学習済みモデルをロード
model = Load_Trained_Model(model)

# 検証フェーズ
model.eval()
# # val_loss = 0.0
# # val_loss_test = 0.0
# total_correct_val = 0
# total_data_len_val = 0
# # prediction_list = []
# # targets_list = []
losses_val = [] # 損失
accs_val = [] # 精度
print("***** validation *****")
epochs = 100
for epoch in range(epochs):

    """"""
    # GPTによる回答例(train時はCrossEntropyにsoftmaxが含まれるのでいらない。val時は別途追記する必要あるかも)
    # モデルの評価時や予測時
    # model.eval()  # モデルを評価モードに設定
    val_loss = 0.0
    # val_loss_test = 0.0
    "----- E資格 -----"
    # total_correct = 0
    # total_data_len = 0
    total_correct_val = 0
    total_data_len_val = 0
    # train_loss = 0.0
    "-----------------"
    with torch.no_grad():
        for inputs, targets in val_loader: # 11回ループする(データ数=41...4[batch]*10[回] +余り1[回] : 11回)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            probabilities = torch.softmax(outputs, dim=1)  # 各クラスに属する確率を計算
            # probabilities = outputs
            predicted_classes = torch.argmax(probabilities, dim=1)  # 最も確率の高いクラスを予測
            # print("probabilities:", probabilities)
            # print("predict classes:", predicted_classes)
            
            batch_size = len(targets)  # バッチサイズの確認
            for i in range(batch_size):  # データ一つずつループ,ミニバッチの中身出しきるまで
                total_data_len_val += 1  # 全データ数を集計
                if predicted_classes[i] == targets[i]:
                    total_correct_val += 1 # 正解のデータ数を集計
                    # print("predicted_classes:", predicted_classes[i])
                else:
                    predicted_tools = [list(tool_to_index.keys())[list(tool_to_index.values()).index(idx)] for idx in predicted_classes]
                    print("Predict Tool : ", predicted_tools)
                    # prediction_list.append(predicted_tools)
                    target_tools = [list(tool_to_index.keys())[list(tool_to_index.values()).index(idx)] for idx in targets]
                    print("Target Tool : ", target_tools)
                    # targets_list.append(target_tools)
                print(f"i={i}, val_loss:{val_loss}")

            # print(f"val_loss:{val_loss}") # 11回ループする
        
    accuracy = total_correct_val/total_data_len_val*100  # 予測精度の算出
    # val_loss/=total_data_len_val  # 損失の平均の算出
    ave_loss = val_loss / total_data_len_val # 損失の平均の算出

    losses_val.append(ave_loss)
    # losses_val.append(val_loss)
    accs_val.append(accuracy)

    # エポックごとの損失の表示
    if epoch % 100 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Val Loss(average): {ave_loss}')
        # print(f'Epoch {epoch+1}/{epochs}, Val Loss: {val_loss}')
        # print("val loss:", val_loss)
        # print("total_data_len_val=", total_data_len_val)
        print("正解率: {}".format(accuracy))
        print("**********")

    # # エポックごとの損失の表示
    # # print("val loss:", val_loss)
    """ val_loader=11で割る場合は全体の平均ではなく、ループした回数で割っているので平均にはならない...41このデータがあるので値がおかしくなる"""
    # val_loss_test = val_loss/len(val_loader) # これ以前にval_loss/=などでval_lossの値を書き換えてしまうと値がおかしくなってしまう
    # print(f'Epoch test {epoch+1}/{epochs}, Val Loss test: {val_loss_test}')
    # print("len(val_loader)=", len(val_loader))
    """"""

# # エポックごとの損失の表示
# # train_loss /= len(train_loader)
# val_loss /= len(val_loader)
# # print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')
# # print(f'Epoch {epoch+1}/{epochs}, Val Loss: {val_loss}')
# print(f"Val Loss: {val_loss}")

print("*************************************************************************************************************")
print("予測結果  : {}".format(predicted_classes))
predicted_tools = [list(tool_to_index.keys())[list(tool_to_index.values()).index(idx)] for idx in predicted_classes]
print("Predict Tool : ", predicted_tools)
print("正解ラベル: {}".format(targets))
target_tools = [list(tool_to_index.keys())[list(tool_to_index.values()).index(idx)] for idx in targets]
print("Target Tool : ", target_tools)
print("*************************************************************************************************************")






# 早期停止のロジックをここに追加する場合があります


# this(addに移動)
# print("predictions: ", outputs.max(1)[1]) # predictions)
# predicted_tools = [list(tool_to_index.keys())[list(tool_to_index.values()).index(idx)] for idx in outputs.max(1)[1]]
# print("Predict: ", predicted_tools)
# prediction_list.append(predicted_tools)

# accuracy = total_correct_val/total_data_len_val*100  # 予測精度の算出
total_accuracy = sum(accs_val)/(epochs) # *total_data_len_val) # accuracyは1epochごtに足しているので、100で割る
print("(総数で割るバージョン) 正解率: {}".format(total_accuracy))
print("**********")
# print(f"内訳:{total_correct_val}/{total_data_len_val}")
print(f"内訳:{sum(accs_val)}/{epochs*total_data_len_val}")

import numpy as np
accs_val_np = np.array(accs_val)
total_accuracy = np.average(accs_val_np)
print("(numpyのaverage関数を使うバージョン) 正解率: {}".format(total_accuracy))
print("**********")
print(f"内訳: sum={sum(accs_val)}, len(np.array)={len(accs_val_np)} -> average()")

# # 可視化用に追加
# # epochs = 1
# epochs = range(len(accs_val))
# # accuracy = total_correct_val/total_data_len_val*100  # 予測精度の算出
# # val_loss/=total_data_len_val  # 損失の平均の算出
# # losses_val.append(val_loss)
# # accs_val.append(accuracy)

# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(15, 8))
# # add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所
# ax1 = fig.add_subplot(1, 2, 1)
# ax2 = fig.add_subplot(1, 2, 2)
# fig.suptitle("Val Result - loss & acc -")
# # plt.style.use("ggplot")
# ax1.plot(epochs, losses_val, label="loss", color = "red") # "orange")
# ax1.legend()
# ax2.plot(epochs, accs_val, label="accurucy", color = "green")
# ax2.legend()
# plt.savefig('result_only_val.png')





# import pprint
# print("PredictionTool:") # , prediction_list)
# pprint.pprint(prediction_list)
# print("targets: ") # , targets_copy)
# # pprint.pprint(targets_copy)
# pprint.pprint(targets_list)



print("\n\n\n**********\nTEST Dataset\n**********")
# テスト
test_data = [
    ("7:30", "STABLE", ""),
    ("7:30", "WALKING", ""),
    ("10:00", "STABLE", ""), # ("10:30", "STABLE", ""),
    ("12:15", "STABLE", ""), # ("11:55", "STABLE", ""),
    ("12:15", "WALKING", ""),
    ("16:55", "RUNNING", ""),
]
# データを数値に変換
inputs = torch.tensor([[time_to_minutes(time), state_to_index[state]] for time, state, _ in test_data], dtype=torch.float)
print("*****")
print("in:", inputs)
print("*****")
print("in [:, 0]:", inputs[:, 0])
# print("in max:", inputs[:, 0].max())


# データの前処理
# 時刻データの正規化
"""
学習したサーのデータセットの最大値を使ってで正規化する必要がある
基準がデータセット(0～1)なのでこの基準にそろえる必要がある
現在時刻が学習データ全体のうちのどこら辺に位置するか？
"""
inputs[:, 0] = inputs[:, 0] / test_input_max # inputs[:, 0].max()
print("in [:, 0]:", inputs[:, 0])
print("*****")
print("行動状態をone-hotベクトル化 ... STABLE:[1,0,0], WALKING:[0,1,0], RUNNING:[0,0,1]")
# 行動状態のone-hotエンコーディング
states = torch.zeros(len(test_data), len(state_to_index))
for i, (_, state, _) in enumerate(test_data):
    states[i, state_to_index[state]] = 1
print("state: ", states)
inputs = torch.cat((inputs[:, :1], states), dim=1) # timeとstatesを連結する

# targets = torch.tensor([tool_to_index[tool] for _, _, tool in test_data], dtype=torch.long)
# print("targets: ", targets)

outputs = model(inputs)
print("test inputs:", inputs)
# print("test targets:", targets)
# print("test pred  max:", outputs.max(1)[1])
probabilities = torch.softmax(outputs, dim=1)  # 各クラスに属する確率を計算
predicted_classes = torch.argmax(probabilities, dim=1)  # 最も確率の高いクラスを予測
print("probabilities:", probabilities)
print("predict classes:", predicted_classes)

# predicted_tools = [list(tool_to_index.keys())[list(tool_to_index.values()).index(idx)] for idx in outputs.max(1)[1]]
predicted_tools = [list(tool_to_index.keys())[list(tool_to_index.values()).index(idx)] for idx in predicted_classes]
print("Predict: ", predicted_tools)
# target_tools = [list(tool_to_index.keys())[list(tool_to_index.values()).index(idx)] for idx in targets]
# print("Target: ", target_tools)