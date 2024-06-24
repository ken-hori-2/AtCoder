
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
    labels = np.random.randint(0, 10, num_samples)  # 行動の意図ラベルをランダムに生成
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
        self.fc1 = nn.Linear(4, 50)  # 入力サイズは4（行動状態、周辺環境、位置情報、時刻）
        self.fc2 = nn.Linear(50, 10)  # 出力サイズは10（予測するパターンの数）
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
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
num_samples = 100
behavior_data, environment_data, location_data, time_data, labels = generate_dummy_data(num_samples)

# データの前処理
data = preprocess_data(behavior_data, environment_data, location_data, time_data)

# ラベルの変換
labels = torch.tensor(labels, dtype=torch.long)

# モデルのトレーニング
def train_model(model, data, labels, epochs=100):
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

for i in range(num_samples):
    behavior = behavior_mapping[data[i, 0].item()]
    environment = environment_mapping[data[i, 1].item()]
    location = location_mapping[data[i, 2].item()]
    time = time_mapping[data[i, 3].item()]
    predicted_intent = intent_mapping[prediction[i].item()]
    print(f"データ: \n - 行動状態={behavior}, \n - 周辺環境={environment},\n - 位置情報={location}, \n - 時刻={time} \n   -> 予測された行動の意図: {predicted_intent}\n")