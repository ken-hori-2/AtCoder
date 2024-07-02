import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F
import numpy as np

# 小数表記にする(指数表記を使わない)
torch.set_printoptions(sci_mode=False) # pytorch
np.set_printoptions(suppress=True) # numpy

# file_path = './userdata_add_through.txt' # 自分のデータセット
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
# tool_to_index = {"楽曲再生": 0, "会議情報": 1, "天気情報": 2, "レストラン検索": 3, "ニュース閲覧": 4, "経路検索":5, "何もしない":6}
tool_to_index = {"楽曲再生": 0, "会議情報": 1, "天気情報": 2, "レストラン検索": 3, "経路検索":4, "何もしない":5}


print("#####")
# print([[time_to_minutes(time), state_to_index[state]] for time, state, _ in data])
print("inputs [time, state]")
print([[time, state] for time, state, _ in data])
print("#####")

# データを数値に変換
inputs = torch.tensor([[time_to_minutes(time), state_to_index[state]] for time, state, _ in data], dtype=torch.float)
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

# 行動状態のone-hotエンコーディング
states = torch.zeros(len(data), len(state_to_index))
for i, (_, state, _) in enumerate(data):
    states[i, state_to_index[state]] = 1
print("state: ", states)
inputs = torch.cat((inputs[:, :1], states), dim=1) # timeとstatesを連結する
targets = torch.tensor([tool_to_index[tool] for _, _, tool in data], dtype=torch.long)
# print("targets: ", targets)

import copy
targets_copy = [tool for _, _, tool in data] # copy.deepcopy(targets)
print("targets: ", targets_copy)
print("*****")



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

# モデルの定義、最適化アルゴリズム、損失関数は前のコードと同様なので省略
# モデルの定義
class PredictionModel(nn.Module):
    def __init__(self):
        super(PredictionModel, self).__init__()
        # Input : time + state(one-hot) = 4

        self.fc1 = nn.Linear(4, 128) # 2, 128)  # 入力層の次元を調整
        self.fc2 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.5)  # ドロップアウト層を追加
        self.fc3 = nn.Linear(256, len(tool_to_index))
        self.bn1 = nn.BatchNorm1d(128) # 64)
        self.bn2 = nn.BatchNorm1d(256) # 64)

        # overlearning version
        # self.fc1 = nn.Linear(4, 32) # 2, 128)  # 入力層の次元を調整
        # self.fc2 = nn.Linear(32, 64)
        # self.dropout = nn.Dropout(0.5)  # ドロップアウト層を追加
        # self.fc3 = nn.Linear(64, len(tool_to_index))
        # self.bn1 = nn.BatchNorm1d(32)
        # self.bn2 = nn.BatchNorm1d(64)

        # **1layer**
        # self.conv = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=5, padding=2), # in:3, out:64
        self.conv = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=5, padding=0)
        # self.conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=2) # sizeは1/2になる # size:16*16 # size:14*14
        self.softmax = nn.Softmax() # inplace=True)
    def forward(self, x):
        """
        Batch Normalizationが無い方が精度良いかも(self.bn)
        """
        # main
        # x = torch.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = torch.relu(self.fc2(x))
        # # x = self.dropout(x) # test
        # x = self.fc3(x)
        # return x

        # # # overlearning version
        # # x = F.relu(self.bn1(self.fc1(x)))
        # # # x = self.dropout(x)
        # # # x = F.relu(self.bn2(self.fc2(x)))
        # # x = F.relu(self.fc2(x))
        # # x = self.dropout(x)
        # # x = self.fc3(x)
        # # return x

        # test
        # x = x.reshape(x.size(0), -1) # データを1次元に変換
        print("x:", x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # x = self.maxpool(self.relu(self.conv(x)))
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.softmax(self.fc2(x))
        return x

""" CNN version """

# 別バージョン
"inputs = [4[batch], 1*len(dataset)[size] *1[channel]]" # まだ勉強中
print("test inputs : ", inputs)

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

    # def forward(self, x):
    #     # 畳み込み層は入力が(batch_size, channels, length)の形であることを期待するため、
    #     # 入力xを適切な形に変形する
    #     x = x.unsqueeze(1)  # (batch_size, 1, length)
    #     x = self.pool(torch.relu(self.conv1(x)))
    #     x = x.view(x.size(0), -1)  # フラット化
    #     x = torch.relu(self.fc1(x))
    #     x = self.dropout(x)
    #     # *** pre ***
    #     # x = self.fc2(x)
    #     # *** pre ***
        
    #     # *** add ***
    #     x = torch.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     # x = self.softmax(x)
    #     # *** add ***
    #     return x
    def forward(self, x):
        # 畳み込み層は入力が(batch_size, channels, length)の形であることを期待するため、入力xを適切な形に変形する
        x = x.unsqueeze(1)  # (batch_size, 1, length)
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # フラット化

        """ bn無い方が精度良いかも """ # どっちか一個のみのバージョンも試してみる
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.bn1(self.fc1(x)))

        x = self.dropout(x)

        """ bn無い方が精度良いかも """
        x = torch.relu(self.fc2(x))
        # x = torch.relu(self.bn2(self.fc2(x)))

        x = self.fc3(x)
        return x
    """
    def forward(self, x):
        
        x = self.features(x) # 4*4サイズ, 128チャネルの画像が出力
        "(1)"
        # x = x.view(x.size(0), -1) # 1次元のベクトルに変換 # size(0) = batch数 = 32
        # ** x = [batch, c * h * w] になる **
        "-----"
        "(2)"
        # x = x.view(num_batches, -1) # 画像の順番になっているので、バッチを先頭に持ってくる. 後ろは任意.
        # x = x.reshape(-1, 4*4*128)  # 画像データを1次元に変換
        # x = x.reshape(-1, 16*16*128)  # 軽量化ver.
        x = x.reshape(x.size(0), -1) # データを1次元に変換
        "-----"
        x = self.classifier(x)
        return x
    """

# モデルのインスタンス化
model = CNNNet()
""" CNN version """

# 損失関数と最適化手法
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)
learning_rate = 0.001
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4) # 重み付けが大きくなりすぎないようにする




class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# 早期終了の使用例
early_stopping = EarlyStopping(patience=5, min_delta=0.01)
# early_stopping = EarlyStopping(patience=20, min_delta=0.001)
# early_stopping = EarlyStopping(patience=20, min_delta=0.01)



# 学習プロセス
# epochs = 1000
# epochs = 15000 # 10000
# epochs = 1000 # 300 # 1 # 000 # 5000
epochs = 5000

# epochs = 1000
# match_count_list = []
# prediction_list = []
# # 正答率
# total_correct = 0
# total_data_len = 0
# # val用
# total_correct_val = 0
# total_data_len_val = 0
# # prediction_list = []
# # targets_list = []

losses = [] # 損失
accs = [] # 精度
losses_val = [] # 損失
accs_val = [] # 精度
for epoch in range(epochs):
    # 訓練フェーズ
    model.train()
    train_loss = 0.0
    "----- E資格 -----"
    total_correct = 0
    total_data_len = 0
    "-----------------"
    for inputs, targets in train_loader: # 1loopで4(バッチサイズ)個のデータを処理する

        # # inputs = inputs.view(batch_size, -1) # 画像の順番になっているので、バッチを先頭に持ってくる. 後ろは任意.
        # # inputs = inputs.reshape(-1, 1*4*4) # [x,x,x,x]で1入力
        # print("train() ... inputs:", inputs)
        # # inputs: tensor([
        # #           [0.4125, 0.0000, 1.0000, 0.0000],
        # #           [0.6250, 1.0000, 0.0000, 0.0000],
        # #           [0.9708, 0.0000, 0.0000, 1.0000],
        # #           [0.3750, 1.0000, 0.0000, 0.0000]
        # #           ])
        
        optimizer.zero_grad()
        outputs = model(inputs) # 順伝播
        # print("train() ... outputs:", outputs)
        # print(" > max : ", outputs.max(1)[1])
        # outputs = それぞれのToolの確率
        # # outputs: tensor([[ 0.0265, -0.0305,  0.0743,  0.0954,  0.2295, -0.0297, -0.0686],
        # #                   [ 0.0238,  0.0094, -0.0340,  0.0740,  0.1984,  0.0076, -0.0693],
        # #                   [ 0.1911,  0.1034, -0.1005,  0.1237,  0.0234, -0.1200, -0.0729],
        # #                   [ 0.1248,  0.0504,  0.0249,  0.2033,  0.0876,  0.1277, -0.1410]], grad_fn=<AddmmBackward0>)
        # # > max :  tensor([4, 4, 0, 3])

        loss = criterion(outputs, targets)
        loss.backward() # 逆伝播で勾配を計算
        optimizer.step() # 勾配をもとにパラメータを最適化
        train_loss += loss.item()

        batch_size = len(targets)  # バッチサイズの確認
        """
        criterionの中にsoftmaxが含まれているので、誤差逆伝播の際には二重に使わないようにモデルの中には記載せずに、一致しているかどうかの確認だけに使うために別途softmaxを記載している
        """
        probabilities = torch.softmax(outputs, dim=1)  # 各クラスに属する確率を計算
        # predicted_classes = torch.argmax(outputs, dim=1)  # 最も確率の高いクラスを予測
        predicted_classes = torch.argmax(probabilities, dim=1)  # 最も確率の高いクラスを予測
        for i in range(batch_size):  # データ一つずつループ,ミニバッチの中身出しきるまで
            total_data_len += 1  # 全データ数を集計
            # if outputs[i] == targets[i]:
            # if outputs.max(1)[1][i] == targets[i]:
            if predicted_classes[i] == targets[i]:
                total_correct += 1 # 正解のデータ数を集計
    

    # 可視化用に追加
    train_accuracy = total_correct/total_data_len*100  # 予測精度の算出
    # train_loss/=total_data_len  # 損失の平均の算出
    ave_train_loss = train_loss / total_data_len # 損失の平均の算出
    # losses.append(train_loss)
    losses.append(ave_train_loss)
    accs.append(train_accuracy)

    # エポックごとの損失の表示
    if epoch % 100 == 0:
        # """
        # 2024/6/10 次回TODO
        # """
        # # train_loss /= len(train_loader) # こっちよりも
        # train_loss = train_loss / total_data_len # こっちの方がいいかも？ # これだとtotal_data_lenはどんどん大きくなるのにtrain_lossはepochごとにリセットされてしまうのでダメかも


        # # print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}') # , Val Loss: {val_loss}')
        # print(f'Epoch {epoch}/{epochs}, Train Loss: {train_loss}') # , Val Loss: {val_loss}')
        print(f'Epoch {epoch}/{epochs}, Train Loss(average): {ave_train_loss}')
        # train_accuracy = total_correct/total_data_len*100  # 予測精度の算出
        print("正解率: {}".format(train_accuracy))
        print("**********")
        # # 早期終了のチェック
        # early_stopping(train_loss) # val_loss)
        # """
        # # ここだとtrainとvalのデータ数に差が出てしまうので下に移動
        # """
        # # if early_stopping.early_stop:
        # #     print("Early stopping")
        # #     break
    
    # # 早期終了のチェック
    # early_stopping(train_loss) # val_loss)
    # if early_stopping.early_stop:
    #     print("Early stopping")
    #     break




    # 検証フェーズ
    model.eval()
    val_loss = 0.0
    "----- E資格 -----"
    total_correct_val = 0
    total_data_len_val = 0
    "-----------------"
    """"""
    # GPTによる回答例(train時はCrossEntropyにsoftmaxが含まれるのでいらない。val時は別途追記する必要あるかも)
    # モデルの評価時や予測時
    # model.eval()  # モデルを評価モードに設定
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            probabilities = torch.softmax(outputs, dim=1)  # 各クラスに属する確率を計算
            predicted_classes = torch.argmax(probabilities, dim=1)  # 最も確率の高いクラスを予測
            # print("probabilities:", probabilities)
            # print("predict classes:", predicted_classes)
            
            batch_size = len(targets)  # バッチサイズの確認
            for i in range(batch_size):  # データ一つずつループ,ミニバッチの中身出しきるまで
                total_data_len_val += 1  # 全データ数を集計
                if predicted_classes[i] == targets[i]:
                    total_correct_val += 1 # 正解のデータ数を集計
                # predicted_tools = [list(tool_to_index.keys())[list(tool_to_index.values()).index(idx)] for idx in predicted_classes]
                # print("Predict: ", predicted_tools)
                # prediction_list.append(predicted_tools)
                # target_tools = [list(tool_to_index.keys())[list(tool_to_index.values()).index(idx)] for idx in targets]
                # print("Target: ", target_tools)
                # targets_list.append(target_tools)
    """"""
    
    """
    2024/6/10 次回TODO
    val_lossという変数は/=len(val_loader)と/total_data_len_valで分けた方がいい
    len(val_loader)=11で固定なので
    """
    val_accuracy = total_correct_val/total_data_len_val*100  # 予測精度の算出
    # val_loss/=total_data_len_val  # 損失の平均の算出
    ave_val_loss = val_loss / total_data_len_val # 損失の平均の算出
    
    # losses_val.append(val_loss)
    losses_val.append(ave_val_loss)
    accs_val.append(val_accuracy)

    # エポックごとの損失の表示
    if epoch % 100 == 0:
        # # エポックごとの損失の表示
        # train_loss /= len(train_loader)
        # val_loss /= len(val_loader) # こっちよりも
        
        # val_loss = val_loss / total_data_len_val # こっちの方がいいかも？


        # print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')
        # print(f'Epoch {epoch}/{epochs}, Val Loss: {val_loss}')
        print(f'Epoch {epoch}/{epochs}, Val Loss(average): {ave_val_loss}')
        # accuracy = total_correct_val/total_data_len_val*100  # 予測精度の算出
        print("正解率: {}".format(val_accuracy))
        print("**********")
        # 早期終了のチェック
        early_stopping(ave_val_loss) # val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    # if early_stopping.early_stop:
    #     print("Early stopping")
    #     break

""" パラメータの保存 """
# パラメータの保存
params = model.state_dict()
# torch.save(params, "model_20240611.param")
torch.save(params, "model_20240701_through_2.param")
# torch.save(params, "model_5000epoch.param")

# 最後の予測結果32個と正解ラベル32個を比較
# だけど今は72/32=2...8なので、最後は8枚のみ出力される
print("*************************************************************************************************************")
print("予測結果  : {}".format(predicted_classes))
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
ax1.plot(epochs, losses_val, label="val loss", color = "blue")
ax1.legend()
ax2.plot(epochs, accs, label="train accurucy", color = "green")
ax2.plot(epochs, accs_val, label="val accurucy", color = "orange")
ax2.legend()
plt.savefig('result_20240611.png')
""" パラメータの保存 """



print("\n\n\n#####\nTEST\n#####")
# テスト
test_data = [
    # ("7:30", "STABLE", ""),
    # ("7:30", "WALKING", ""),
    ("10:00", "STABLE", ""), # ("10:30", "STABLE", ""),
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