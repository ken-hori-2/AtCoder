"""
### 対策
1. **正解ラベルの均等化**: 各ラベルが均等に分布するようにデータを生成します。
2. **データの前処理**: データの正規化やエンコーディングを行います。
3. **モデルの改善**: モデルのアーキテクチャを改善し、トレーニングプロセスを最適化します。
"""


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ダミーデータの生成
def generate_dummy_data(num_samples_per_class):
    np.random.seed(0)
    behavior_data = []
    environment_data = []
    location_data = []
    time_data = []
    labels = []

    # 各クラスのデータを均等に生成
    for label in range(10):
        for _ in range(num_samples_per_class):
            if label == 0:  # 通勤・通学
                behavior_data.append(np.random.choice(['walking', 'running']))
                environment_data.append(np.random.choice(['quiet', 'noise']))
                location_data.append(np.random.choice(['train', 'station']))
                time_data.append(np.random.choice(['08:00', '08:05', '08:10', '08:15', '08:20', '08:25', '08:30', '08:35', '08:40', '08:45', '08:50', '08:55', '17:00', '17:05', '17:10', '17:15', '17:20', '17:25', '17:30', '17:35', '17:40', '17:45', '17:50', '17:55']))
            elif label == 1:  # 昼食・休憩
                behavior_data.append('sitdown')
                environment_data.append('conversation')
                location_data.append('office')
                time_data.append(np.random.choice(['12:00', '12:05', '12:10', '12:15', '12:20', '12:25', '12:30', '12:35', '12:40', '12:45', '12:50', '12:55']))
            elif label == 2:  # 運動・トレーニング
                behavior_data.append('running')
                environment_data.append('quiet')
                location_data.append(np.random.choice(['gym', 'park']))
                time_data.append(np.random.choice(['06:00', '06:05', '06:10', '06:15', '06:20', '06:25', '06:30', '06:35', '06:40', '06:45', '06:50', '06:55', '18:00', '18:05', '18:10', '18:15', '18:20', '18:25', '18:30', '18:35', '18:40', '18:45', '18:50', '18:55']))
            elif label == 3:  # 買い物
                behavior_data.append('walking')
                environment_data.append('conversation')
                location_data.append('shopping')
                time_data.append(np.random.choice(['10:00', '10:05', '10:10', '10:15', '10:20', '10:25', '10:30', '10:35', '10:40', '10:45', '10:50', '10:55', '11:00', '11:05', '11:10', '11:15', '11:20', '11:25', '11:30', '11:35', '11:40', '11:45', '11:50', '11:55', '12:00', '12:05', '12:10', '12:15', '12:20', '12:25', '12:30', '12:35', '12:40', '12:45', '12:50', '12:55', '13:00', '13:05', '13:10', '13:15', '13:20', '13:25', '13:30', '13:35', '13:40', '13:45', '13:50', '13:55', '14:00', '14:05', '14:10', '14:15', '14:20', '14:25', '14:30', '14:35', '14:40', '14:45', '14:50', '14:55', '15:00', '15:05', '15:10', '15:15', '15:20', '15:25', '15:30', '15:35', '15:40', '15:45', '15:50', '15:55', '16:00', '16:05', '16:10', '16:15', '16:20', '16:25', '16:30', '16:35', '16:40', '16:45', '16:50', '16:55', '17:00', '17:05', '17:10', '17:15', '17:20', '17:25', '17:30', '17:35', '17:40', '17:45', '17:50', '17:55', '18:00', '18:05', '18:10', '18:15', '18:20', '18:25', '18:30', '18:35', '18:40', '18:45', '18:50', '18:55', '19:00', '19:05', '19:10', '19:15', '19:20', '19:25', '19:30', '19:35', '19:40', '19:45', '19:50', '19:55']))
            elif label == 4:  # 会議・打ち合わせ
                behavior_data.append('gesture')
                environment_data.append('conversation')
                location_data.append('office')
                time_data.append(np.random.choice(['09:00', '09:05', '09:10', '09:15', '09:20', '09:25', '09:30', '09:35', '09:40', '09:45', '09:50', '09:55', '10:00', '10:05', '10:10', '10:15', '10:20', '10:25', '10:30', '10:35', '10:40', '10:45', '10:50', '10:55', '11:00', '11:05', '11:10', '11:15', '11:20', '11:25', '11:30', '11:35', '11:40', '11:45', '11:50', '11:55', '12:00', '12:05', '12:10', '12:15', '12:20', '12:25', '12:30', '12:35', '12:40', '12:45', '12:50', '12:55', '13:00', '13:05', '13:10', '13:15', '13:20', '13:25', '13:30', '13:35', '13:40', '13:45', '13:50', '13:55', '14:00', '14:05', '14:10', '14:15', '14:20', '14:25', '14:30', '14:35', '14:40', '14:45', '14:50', '14:55', '15:00', '15:05', '15:10', '15:15', '15:20', '15:25', '15:30', '15:35', '15:40', '15:45', '15:50', '15:55', '16:00', '16:05', '16:10', '16:15', '16:20', '16:25', '16:30', '16:35', '16:40', '16:45', '16:50', '16:55']))
            elif label == 5:  # 緊急避難
                behavior_data.append('running')
                environment_data.append('noise')
                location_data.append(np.random.choice(['home', 'office', 'train', 'station', 'shopping', 'gym', 'park', 'other']))
                time_data.append(np.random.choice(generate_time_data()))
            elif label == 6:  # 友人との集まり
                behavior_data.append('sitdown')
                environment_data.append('conversation')
                location_data.append(np.random.choice(['home', 'other']))
                time_data.append(np.random.choice(['18:00', '18:05', '18:10', '18:15', '18:20', '18:25', '18:30', '18:35', '18:40', '18:45', '18:50', '18:55', '19:00', '19:05', '19:10', '19:15', '19:20', '19:25', '19:30', '19:35', '19:40', '19:45', '19:50', '19:55', '20:00', '20:05', '20:10', '20:15', '20:20', '20:25', '20:30', '20:35', '20:40', '20:45', '20:50', '20:55', '21:00', '21:05', '21:10', '21:15', '21:20', '21:25', '21:30', '21:35', '21:40', '21:45', '21:50', '21:55']))
            elif label == 7:  # 観光・散策
                behavior_data.append('walking')
                environment_data.append('quiet')
                location_data.append(np.random.choice(['park', 'other']))
                time_data.append(np.random.choice(['10:00', '10:05', '10:10', '10:15', '10:20', '10:25', '10:30', '10:35', '10:40', '10:45', '10:50', '10:55', '11:00', '11:05', '11:10', '11:15', '11:20', '11:25', '11:30', '11:35', '11:40', '11:45', '11:50', '11:55', '12:00', '12:05', '12:10', '12:15', '12:20', '12:25', '12:30', '12:35', '12:40', '12:45', '12:50', '12:55', '13:00', '13:05', '13:10', '13:15', '13:20', '13:25', '13:30', '13:35', '13:40', '13:45', '13:50', '13:55', '14:00', '14:05', '14:10', '14:15', '14:20', '14:25', '14:30', '14:35', '14:40', '14:45', '14:50', '14:55', '15:00', '15:05', '15:10', '15:15', '15:20', '15:25', '15:30', '15:35', '15:40', '15:45', '15:50', '15:55', '16:00', '16:05', '16:10', '16:15', '16:20', '16:25', '16:30', '16:35', '16:40', '16:45', '16:50', '16:55', '17:00', '17:05', '17:10', '17:15', '17:20', '17:25', '17:30', '17:35', '17:40', '17:45', '17:50', '17:55']))
            elif label == 8:  # 子供の送り迎え
                behavior_data.append('walking')
                environment_data.append('conversation')
                location_data.append('home')
                time_data.append(np.random.choice(['08:00', '08:05', '08:10', '08:15', '08:20', '08:25', '08:30', '08:35', '08:40', '08:45', '08:50', '08:55', '15:00', '15:05', '15:10', '15:15', '15:20', '15:25', '15:30', '15:35', '15:40', '15:45', '15:50', '15:55']))
            elif label == 9:  # イベント参加
                behavior_data.append('walking')
                environment_data.append('conversation')
                location_data.append('other')
                time_data.append(np.random.choice(generate_time_data()))
            labels.append(label)

    return behavior_data, environment_data, location_data, time_data, labels

# 時刻データの生成（5分刻み）
def generate_time_data():
    times = []
    for hour in range(24):
        for minute in range(0, 60, 5):
            times.append(f"{hour:02}:{minute:02}")
    return times

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
num_samples_per_class = 10
behavior_data, environment_data, location_data, time_data, labels = generate_dummy_data(num_samples_per_class)

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

## 結果の出力
behavior_mapping = {0: 'stable', 1: 'walking', 2: 'running', 3: 'standup', 4: 'sitdown', 5: 'gesture'}
environment_mapping = {0: 'quiet', 1: 'conversation', 2: 'noise'}
location_mapping = {0: 'home', 1: 'office', 2: 'train', 3: 'station', 4: 'shopping', 5: 'gym', 6: 'park', 7: 'other'}
time_mapping = {idx: time for idx, time in enumerate(generate_time_data())}
intent_mapping = {0: '通勤・通学', 1: '昼食・休憩', 2: '運動・トレーニング', 3: '買い物', 4: '会議・打ち合わせ', 5: '緊急避難', 6: '友人との集まり', 7: '観光・散策', 8: '子供の送り迎え', 9: 'イベント参加'}

# print(intent_mapping)

num = 100
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
plt.figure(figsize=(15, 8)) # 10, 8))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=intent_mapping.values(), yticklabels=intent_mapping.values())
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', xticklabels=intent_mapping.values(), yticklabels=intent_mapping.values())
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='BrBG_r', xticklabels=intent_mapping.values(), yticklabels=intent_mapping.values())
plt.xlabel('予測された行動の意図')
plt.ylabel('実際の行動の意図')
plt.title('予測と実際の行動の意図の一致')
# plt.legend(prop = {'family' : 'MS Gothic'})
plt.show()
