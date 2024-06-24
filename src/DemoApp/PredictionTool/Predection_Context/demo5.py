"""
### 一般的な行動パターンに基づくラベル設定
- **通勤・通学**: 8:00 - 9:00, 17:00 - 18:00, location: train, station, walking, running
- **昼食・休憩**: 12:00 - 13:00, location: office, shopping, sitdown
- **運動・トレーニング**: 6:00 - 7:00, 18:00 - 20:00, location: gym, park, running, walking
- **買い物**: 10:00 - 20:00, location: shopping, walking
- **会議・打ち合わせ**: 9:00 - 17:00, location: office, sitdown, gesture
- **緊急避難**: noise, running, walking
- **友人との集まり**: 18:00 - 22:00, location: home, other, conversation, sitdown
- **観光・散策**: 10:00 - 18:00, location: park, other, walking
- **子供の送り迎え**: 8:00 - 9:00, 15:00 - 16:00, location: home, school, walking
- **イベント参加**: event time, location: other, walking, standing
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ダミーデータの生成
def generate_dummy_data(num_samples):
    np.random.seed(0)
    behavior_data = np.random.choice(['stable', 'walking', 'running', 'standup', 'sitdown', 'gesture'], num_samples)
    environment_data = np.random.choice(['quiet', 'conversation', 'noise'], num_samples)
    location_data = np.random.choice(['home', 'office', 'train', 'station', 'shopping', 'gym', 'park', 'other'], num_samples)
    time_data = np.random.choice(generate_time_data(), num_samples)
    labels = generate_labels(behavior_data, environment_data, location_data, time_data)
    return behavior_data, environment_data, location_data, time_data, labels

# 時刻データの生成（5分刻み）
def generate_time_data():
    times = []
    for hour in range(24):
        for minute in range(0, 60, 5):
            times.append(f"{hour:02}:{minute:02}")
    return times

# ラベルの生成
def generate_labels(behavior_data, environment_data, location_data, time_data):
    labels = []
    for behavior, environment, location, time in zip(behavior_data, environment_data, location_data, time_data):
        hour = int(time.split(':')[0])
        if location in ['train', 'station'] and (8 <= hour < 9 or 17 <= hour < 18):
            labels.append(0)  # 通勤・通学
        elif location in ['office', 'shopping'] and (12 <= hour < 13):
            labels.append(1)  # 昼食・休憩
        elif location in ['gym', 'park'] and (6 <= hour < 7 or 18 <= hour < 20):
            labels.append(2)  # 運動・トレーニング
        elif location == 'shopping' and (10 <= hour < 20):
            labels.append(3)  # 買い物
        elif location == 'office' and (9 <= hour < 17):
            labels.append(4)  # 会議・打ち合わせ
        elif environment == 'noise' and behavior in ['running', 'walking']:
            labels.append(5)  # 緊急避難
        elif location in ['home', 'other'] and (18 <= hour < 22):
            labels.append(6)  # 友人との集まり
        elif location in ['park', 'other'] and (10 <= hour < 18):
            labels.append(7)  # 観光・散策
        elif location == 'home' and (8 <= hour < 9 or 15 <= hour < 16):
            labels.append(8)  # 子供の送り迎え
        else:
        # elif location == 'other' and (9 <= hour < 17):
            labels.append(9)  # イベント参加
    return labels

# データの前処理（例）
def preprocess_data(behavior_data, environment_data, location_data, time_data):
    # カテゴリカルデータを数値に変換
    behavior_mapping = {'stable': 0, 'walking': 1, 'running': 2, 'standup': 3, 'sitdown': 4, 'gesture': 5}
    environment_mapping = {'quiet': 0, 'conversation': 1, 'noise': 2}
    location_mapping = {'home': 0, 'office': 1, 'train': 2, 'station': 3, 'shopping': 4, 'gym': 5, 'park': 6, 'other': 7}
    time_mapping = {time: idx for idx, time in enumerate(generate_time_data())}
    
    behavior_data = np.array([behavior_mapping[b] for b in behavior_data])
    environment_data = np.array([environment_mapping[e] for e in environment_data])
    location_data = np.array([location_mapping[l] for l in location_data])
    time_data = np.array([time_mapping[t] for t in time_data])
    
    # データの統合
    data = np.hstack((behavior_data.reshape(-1, 1), environment_data.reshape(-1, 1), location_data.reshape(-1, 1), time_data.reshape(-1, 1)))
    return torch.tensor(data, dtype=torch.float32)

# モデルの定義
# class MultiModalModel(nn.Module):
#     def __init__(self):
#         super(MultiModalModel, self).__init__()
#         self.fc1 = nn.Linear(4, 50)  # 入力サイズは4（行動状態、周辺環境、位置情報、時刻）
#         self.fc2 = nn.Linear(50, 10)  # 出力サイズは10（予測するパターンの数）
    
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
class MultiModalModel(nn.Module):
    def __init__(self):
        super(MultiModalModel, self).__init__()
        # self.fc1 = nn.Linear(4, 50)  # 入力サイズは4（行動状態、周辺環境、位置情報、時刻）
        # self.fc2 = nn.Linear(50, 10)  # 出力サイズは10（予測するパターンの数）
        self.fc1 = nn.Linear(4, 128)  # 入力サイズは4（行動状態、周辺環境、位置情報、時刻）
        self.fc2 = nn.Linear(128, 256)  # 出力サイズは10（予測するパターンの数）
        self.fc3 = nn.Linear(256, 10)  # 出力サイズは10（予測するパターンの数）
    
    def forward(self, x):
        # x = torch.relu(self.fc1(x))
        # x = self.fc2(x)
        # return x
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ハイパーパラメータの設定
input_size = 4
hidden_size = 50
output_size = 10

# モデルのインスタンス化
model = MultiModalModel()

# 損失関数と最適化手法の定義
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# LangChainのモデルとして登録
class PyTorchModel:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
    
    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data)
        return outputs.argmax(dim=1)

# エージェントの定義
class BehaviorPredictionAgent:
    def __init__(self, model):
        self.model = model
    
    def predict_behavior(self, data):
        # モデルによる予測
        prediction = self.model.predict(data)
        return prediction

# ダミーデータの生成
num_samples = 10 * 10
behavior_data, environment_data, location_data, time_data, labels = generate_dummy_data(num_samples)

# データの前処理
data = preprocess_data(behavior_data, environment_data, location_data, time_data)

# ラベルの変換
labels = torch.tensor(labels, dtype=torch.long)

# モデルのトレーニング
def train_model(model, data, labels, epochs=1000):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# モデルのトレーニング
train_model(model, data, labels)

# LangChainのモデルとして登録
pytorch_model = PyTorchModel(model=model, criterion=criterion, optimizer=optimizer)

# エージェントのインスタンス化
agent = BehaviorPredictionAgent(model=pytorch_model)

# エージェントによる予測の実行
prediction = agent.predict_behavior(data)

# 結果の出力
behavior_mapping = {0: 'stable', 1: 'walking', 2: 'running', 3: 'standup', 4: 'sitdown', 5: 'gesture'}
environment_mapping = {0: 'quiet', 1: 'conversation', 2: 'noise'}
location_mapping = {0: 'home', 1: 'office', 2: 'train', 3: 'station', 4: 'shopping', 5: 'gym', 6: 'park', 7: 'other'}
time_mapping = {idx: time for idx, time in enumerate(generate_time_data())}
intent_mapping = {0: '通勤・通学', 1: '昼食・休憩', 2: '運動・トレーニング', 3: '買い物', 4: '会議・打ち合わせ', 5: '緊急避難', 6: '友人との集まり', 7: '観光・散策', 8: '子供の送り迎え', 9: 'イベント参加'}

# for i in range(100): # num_samples):
#     behavior = behavior_mapping[data[i, 0].item()]
#     environment = environment_mapping[data[i, 1].item()]
#     location = location_mapping[data[i, 2].item()]
#     time = time_mapping[data[i, 3].item()]
#     predicted_intent = intent_mapping[prediction[i].item()]
#     actual_intent = intent_mapping[labels[i].item()]
#     # print(f"データ: 行動状態={behavior}, 周辺環境={environment}, 位置情報={location}, 時刻={time} -> 予測された行動の意図: {predicted_intent}, 実際の行動の意図: {actual_intent}")
#     print(f"データ: \n - 行動状態={behavior}, \n - 周辺環境={environment},\n - 位置情報={location}, \n - 時刻={time} \n   -> 予測された行動の意図: {predicted_intent}, 実際の行動の意図: {actual_intent}\n")

num = 10 * 10
for i in range(num): # num_samples_per_class):
    behavior = behavior_mapping[data[i, 0].item()]
    environment = environment_mapping[data[i, 1].item()]
    location = location_mapping[data[i, 2].item()]
    time = time_mapping[data[i, 3].item()]
    predicted_intent = intent_mapping[prediction[i].item()]
    actual_intent = intent_mapping[labels[i].item()]
    # print(f"データ: 行動状態={behavior}, 周辺環境={environment}, 位置情報={location}, 時刻={time} -> 予測された行動の意図: {predicted_intent}, 実際の行動の意図: {actual_intent}")
    print(f"データ: \n - 行動状態={behavior}, \n - 周辺環境={environment},\n - 位置情報={location}, \n - 時刻={time} \n   -> 予測された行動の意図: {predicted_intent}, 実際の行動の意図: {actual_intent}\n")


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import japanize_matplotlib
# 混同行列の作成
conf_matrix = confusion_matrix(labels, prediction)

# 混同行列のヒートマップ表示
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=intent_mapping.values(), yticklabels=intent_mapping.values())
plt.xlabel('予測された行動の意図')
plt.ylabel('実際の行動の意図')
plt.title('予測と実際の行動の意図の一致')
# plt.legend(prop = {'family' : 'MS Gothic'})
plt.show()