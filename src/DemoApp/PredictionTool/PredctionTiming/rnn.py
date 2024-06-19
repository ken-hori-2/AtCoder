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
X = torch.tensor([item[0] for item in data], dtype=torch.float32).view(-1, 1, 1)  # RNNには3Dテンソルが必要
y = torch.tensor([item[1] for item in data], dtype=torch.float32).view(-1, 1)

# RNNモデルの定義
class TimeCountRNN(nn.Module):
    def __init__(self):
        super(TimeCountRNN, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=10, num_layers=1, batch_first=True)
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        # RNNの出力と最後の隠れ状態を取得
        out, _ = self.rnn(x)
        # 最後のタイムステップの出力を取得
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# モデルのインスタンス化
model = TimeCountRNN()

# # 損失関数と最適化アルゴリズム
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01) # 01)
# # 損失関数と最適化手法
# criterion = nn.CrossEntropyLoss()
# learning_rate = 0.001
# optimizer = optim.Adam(model.parameters(), lr=learning_rate) # , weight_decay=5e-4) # 重み付けが大きくなりすぎないようにする

# トレーニングループ
num_epochs = 10000
# training
running_loss = 0.0
training_accuracy = 0.0
import numpy as np
for epoch in range(num_epochs):
    # # Forward pass: モデルで予測値を計算
    # predictions = model(X)
    # loss = criterion(predictions, y)
    
    # # Backward pass: 勾配を計算
    # optimizer.zero_grad()
    # loss.backward()
    
    # # パラメータの更新
    # optimizer.step()

    optimizer.zero_grad()
    # data, label = mkRandomBatch(train_x, train_t, batch_size)
    output = model(X)
    loss = criterion(output, y) # label)
    loss.backward()
    optimizer.step()
    running_loss += loss.item() # data[0]
    # training_accuracy += np.sum(np.abs((output - y).numpy()) < 0.1)
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# 予測値の計算（例）
with torch.no_grad():
    predicted_counts = model(X).flatten()
    # 予測値と実際のカウントを比較するなどして評価