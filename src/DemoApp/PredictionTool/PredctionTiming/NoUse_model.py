# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader, random_split
# import torch.nn.functional as F
# from datetime import datetime

# # Read the data from times_count.txt
# data = []
# with open('time_counts.txt', 'r', encoding='utf-8') as file:
#     for line in file:
#         time_str, count = line.strip().split(', ')
#         # Convert time to minutes since 00:00
#         time = datetime.strptime(time_str, "%H:%M")
#         minutes = time.hour * 60 + time.minute
#         data.append((minutes, int(count)))

# # Convert to PyTorch tensors
# X = torch.tensor([minutes for minutes, count in data], dtype=torch.float32).view(-1, 1)
# y = torch.tensor([count for minutes, count in data], dtype=torch.float32)


# ### Step 2: Define the Model
# # Create a simple linear regression model using PyTorch's `nn.Module`.


# class TimeCountPredictor(nn.Module):
#     def __init__(self):
#         super(TimeCountPredictor, self).__init__()
#         self.linear = nn.Linear(1, 1)  # One input feature (time), one output (count)

#     def forward(self, x):
#         return self.linear(x)


# ### Step 3: Train the Model
# # Set up the loss function, optimizer, and train the model using the data.


# # Instantiate the model
# model = TimeCountPredictor()

# # Loss function and optimizer
# criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# # Training loop
# num_epochs = 1000
# for epoch in range(num_epochs):
#     # Forward pass
#     outputs = model(X)
#     loss = criterion(outputs, y)
    
#     # Backward pass and optimization
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     if (epoch+1) % 100 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')


# ### Step 4: Evaluate the Model
# # After training, you can evaluate the model's performance on test data or use it to make predictions.


# # Make predictions (example)
# with torch.no_grad():
#     predicted_counts = model(X).flatten()
#     # You can compare predicted_counts with actual counts (y)




import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

# データの読み込みと前処理
data = []
with open('time_counts.txt', 'r', encoding='utf-8') as file:
    next(file)  # ヘッダー行をスキップ
    for line in file:
        time_str, count = line.strip().split(', ')
        # 時刻を分単位の数値に変換
        time_val = datetime.strptime(time_str, "%H:%M")
        minutes = time_val.hour * 60 + time_val.minute
        data.append((minutes, int(count)))

# データをテンソルに変換
inputs = torch.tensor([item[0] for item in data], dtype=torch.float32).view(-1, 1)
targets = torch.tensor([item[1] for item in data], dtype=torch.float32).view(-1, 1)

# # 時刻を分に変換する関数
# def time_to_minutes(time_str):
#     hours, minutes = map(int, time_str.split(':'))
#     return hours * 60 + minutes

# # データを数値に変換
# inputs = torch.tensor([time_to_minutes(time) for time in data], dtype=torch.float)

print(inputs)
print("*****")
# データの前処理
# 時刻データの正規化
inputs[:, 0] = inputs[:, 0] / inputs[:, 0].max()
print("in [:, 0]:", inputs[:, 0])
print("*****")

print("-----")
print(targets)

# モデルの定義
class CountPredictor(nn.Module):
    # def __init__(self):
    #     super(CountPredictor, self).__init__()
    #     self.linear = nn.Linear(1, 1)  # 入力は1次元、出力も1次元

    # def forward(self, x):
    #     return self.linear(x)
    def __init__(self):
        super(CountPredictor, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # 入力は1次元、出力は10次元
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)  # ドロップアウト層（25%のユニットを無効化）
        self.fc2 = nn.Linear(10, 1)  # 隠れ層からの出力は1次元（カウント）

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# モデルのインスタンス化
model = CountPredictor()

# # 損失関数と最適化アルゴリズム
# criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001)
# 損失関数と最適化手法
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)
learning_rate = 0.001
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4) # 重み付けが大きくなりすぎないようにする

# トレーニングループ
epochs = 1000
losses = [] # 損失
accs = [] # 精度
losses_val = [] # 損失
accs_val = [] # 精度
train_loss = 0.0
"----- E資格 -----"
total_correct = 0
total_data_len = 0
"-----------------"

for epoch in range(epochs):
    # # Forward pass: モデルで予測値を計算
    # predictions = model(X)
    # loss = criterion(predictions, y)
    
    # # Backward pass: 勾配を計算
    # optimizer.zero_grad()
    # loss.backward()
    # 訓練フェーズ
    model.train()
    # train_loss = 0.0
    # "----- E資格 -----"
    # total_correct = 0
    # total_data_len = 0
    # "-----------------"
    # # パラメータの更新
    # optimizer.step()
    optimizer.zero_grad()
    outputs = model(inputs) # 順伝播

    loss = criterion(outputs, targets)
    loss.backward() # 逆伝播で勾配を計算
    optimizer.step() # 勾配をもとにパラメータを最適化
    train_loss += loss.item()

    # batch_size = len(targets)  # バッチサイズの確認
    # """
    # criterionの中にsoftmaxが含まれているので、誤差逆伝播の際には二重に使わないようにモデルの中には記載せずに、一致しているかどうかの確認だけに使うために別途softmaxを記載している
    # """
    # probabilities = torch.softmax(outputs, dim=1)  # 各クラスに属する確率を計算
    # # predicted_classes = torch.argmax(outputs, dim=1)  # 最も確率の高いクラスを予測
    # predicted_classes = torch.argmax(probabilities, dim=1)  # 最も確率の高いクラスを予測
    # for i in range(batch_size):  # データ一つずつループ,ミニバッチの中身出しきるまで
    #     total_data_len += 1  # 全データ数を集計
    #     # if outputs[i] == targets[i]:
    #     # if outputs.max(1)[1][i] == targets[i]:
    #     if predicted_classes[i] == targets[i]:
    #         total_correct += 1 # 正解のデータ数を集計
    
    # # 可視化用に追加
    # train_accuracy = total_correct/total_data_len*100  # 予測精度の算出
    # # train_loss/=total_data_len  # 損失の平均の算出
    # ave_train_loss = train_loss / total_data_len # 損失の平均の算出
    # # losses.append(train_loss)
    # losses.append(ave_train_loss)
    # accs.append(train_accuracy)
    if (epoch+1) % 100 == 0:
        #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
        print(f'Epoch {epoch}/{epochs}, Train Loss(average): {train_loss}')
        # print(f'Epoch {epoch}/{epochs}, Train Loss(average): {ave_train_loss}')
        # train_accuracy = total_correct/total_data_len*100  # 予測精度の算出
        # print("正解率: {}".format(train_accuracy))
        print("**********")

# # 検証フェーズ
# model.eval()
# val_loss = 0.0
# "----- E資格 -----"
# total_correct_val = 0
# total_data_len_val = 0
# "-----------------"
# # 予測値の計算（例）
# with torch.no_grad():
#     # predicted_counts = model(inputs).flatten()

#     # # 予測値と実際のカウントを比較するなどして評価
#     outputs = model(inputs)
#     loss = criterion(outputs, targets)
#     val_loss += loss.item()
#     probabilities = torch.softmax(outputs, dim=1)  # 各クラスに属する確率を計算
#     predicted_classes = torch.argmax(probabilities, dim=1)  # 最も確率の高いクラスを予測
#     # print("probabilities:", probabilities)
#     # print("predict classes:", predicted_classes)
    
#     batch_size = len(targets)  # バッチサイズの確認
#     for i in range(batch_size):  # データ一つずつループ,ミニバッチの中身出しきるまで
#         total_data_len_val += 1  # 全データ数を集計
#         if predicted_classes[i] == targets[i]:
#             total_correct_val += 1 # 正解のデータ数を集計
    
#     val_accuracy = total_correct_val/total_data_len_val*100  # 予測精度の算出
#     ave_val_loss = val_loss / total_data_len_val # 損失の平均の算出
    
#     losses_val.append(ave_val_loss)
#     accs_val.append(val_accuracy)

#     # エポックごとの損失の表示
#     if epoch % 100 == 0:
#         print(f'Epoch {epoch}/{epochs}, Val Loss(average): {ave_val_loss}')
#         print("正解率: {}".format(val_accuracy))
#         print("**********")