import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F
import numpy as np

# CNNモデルのアーキテクチャ
class CNNNet(nn.Module):
    def __init__(self, tool_to_index):
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

class PredctionModel():
    def __init__(self):
        

        # 小数表記にする(指数表記を使わない)
        torch.set_printoptions(sci_mode=False) # pytorch
        np.set_printoptions(suppress=True) # numpy

        # file_path = './userdata.txt' # 自分のデータセット
        self.file_path = './userdata4timeaction.txt' # 自分のデータセット
        # 空のリストを作成
        self.data = []
        # ファイルを開いて各行を読み込む
        with open(self.file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # 改行文字を削除し、カンマで分割
                parts = line.strip().split(', ')
                # 分割したデータをタプルに変換してリストに追加
                if len(parts) == 3:  # データが3つの部分に分割されていることを確認
                    self.data.append((parts[0], parts[1], parts[2]))
        

        # 行動状態とツール名を数値に変換する辞書
        # state_to_index = {"WALKING": 0, "STABLE": 1, "RUNNING": 2}
        self.state_to_index = {"STABLE": 0, "WALKING": 1, "RUNNING": 2}
        self.tool_to_index = {"楽曲再生": 0, "会議情報": 1, "天気情報": 2, "レストラン検索": 3, "経路検索":4, "何もしない":5}

        # データを数値に変換
        inputs = torch.tensor([[self.time_to_minutes(time), self.state_to_index[state]] for time, state, _ in self.data], dtype=torch.float)
        
        # ここは必要（時刻データの正規化をする際に学習データの最大値を使用する
        self.test_input_max = inputs[:, 0].max()
        

        # テスト
        # self.test_data = [
        #     ("7:30", "STABLE", ""),
        #     ("7:30", "WALKING", ""),
        #     ("10:00", "STABLE", ""), # ("10:30", "STABLE", ""),
        #     ("12:15", "STABLE", ""), # ("11:55", "STABLE", ""),
        #     ("12:15", "WALKING", ""),
        #     ("16:55", "RUNNING", ""),
        # ]
        
    # 時刻を分に変換する関数
    def time_to_minutes(self, time_str):
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes
    
    def time_to_minutes_4test(self, time_str):
        print("現在時刻：", time_str)
        hours, minutes, _ = map(int, time_str.split(':')) # datetime.timedeltaからstrにした場合
        return hours * 60 + minutes
    
    def Load_Trained_Model(self, model):
        print("***** validation *****")
        # パラメータの読み込み
        param_load = torch.load("predictiontool_model.param") # 学習時の精度が高いモデル


        model.load_state_dict(param_load)
        # validation(validation_loader, model)
        
        return model
    
    def run(self, time, UserAction):
        self.test_data = [
            (time, UserAction, ""),
        ]

        # time = str(time)
        print("time:", time)


        # # データの前処理
        # # 時刻データの正規化
        # inputs[:, 0] = inputs[:, 0] / inputs[:, 0].max()
        # print("in [:, 0]:", inputs[:, 0])
        # print("*****")

        # print("行動状態をone-hotベクトル化 ... STABLE:[1,0,0], WALKING:[0,1,0], RUNNING:[0,0,1]")

        # # 行動状態のone-hotエンコーディング
        # states = torch.zeros(len(self.data), len(state_to_index))
        # for i, (_, state, _) in enumerate(self.data):
        #     states[i, state_to_index[state]] = 1
        # print("state: ", states)
        # inputs = torch.cat((inputs[:, :1], states), dim=1) # timeとstatesを連結する
        # targets = torch.tensor([tool_to_index[tool] for _, _, tool in self.data], dtype=torch.long)
        # targets_copy = [tool for _, _, tool in self.data] # copy.deepcopy(targets)

        # # データセットの作成
        # dataset = TensorDataset(inputs, targets)

        # # データセットのサイズ
        # dataset_size = len(dataset)

        # # 訓練用と検証用のサイズを計算
        # # train_size = int(dataset_size * 0.8)
        # """ テスト (9割val用に使う) """
        # train_size = int(dataset_size * 0.1)
        # """ テスト (9割val用に使う) """
        # val_size = dataset_size - train_size

        # # データセットをランダムに分割
        # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # # DataLoaderの作成
        # batch_size = 4  # 例としてバッチサイズを4に設定
        # # batch_size = 32
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # モデルのインスタンス化
        model = CNNNet(self.tool_to_index)
        """ CNN version """

        # # 損失関数と最適化手法
        # criterion = nn.CrossEntropyLoss()
        # # optimizer = optim.SGD(model.parameters(), lr=0.01)
        # learning_rate = 0.001
        # # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4) # 重み付けが大きくなりすぎないようにする

        # 学習済みモデルをロード
        model = self.Load_Trained_Model(model)

        # 検証フェーズ
        model.eval()



        print("\n\n\n**********\nTEST Dataset\n**********")
        # データを数値に変換
        inputs = torch.tensor([[self.time_to_minutes_4test(time), self.state_to_index[state]] for time, state, _ in self.test_data], dtype=torch.float)
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
        inputs[:, 0] = inputs[:, 0] / self.test_input_max # inputs[:, 0].max()
        print("in [:, 0]:", inputs[:, 0])
        print("*****")
        print("行動状態をone-hotベクトル化 ... STABLE:[1,0,0], WALKING:[0,1,0], RUNNING:[0,0,1]")
        # 行動状態のone-hotエンコーディング
        states = torch.zeros(len(self.test_data), len(self.state_to_index))
        for i, (_, state, _) in enumerate(self.test_data):
            states[i, self.state_to_index[state]] = 1
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
        predicted_tools = [list(self.tool_to_index.keys())[list(self.tool_to_index.values()).index(idx)] for idx in predicted_classes]
        print("Predict: ", predicted_tools)
        # target_tools = [list(tool_to_index.keys())[list(tool_to_index.values()).index(idx)] for idx in targets]
        # print("Target: ", target_tools)

        return predicted_tools


if __name__ == "__main__":
    predictionmodel = PredctionModel()

    import datetime
    H = 8
    M = 36
    dt_now_for_time_action = datetime.timedelta(hours=H, minutes=M) # 経路案内
    # dt_now_for_time_action = dt_now_for_time_action.strftime('%H:%M')
    # dt_now_for_time_action = datetime.strptime(dt_now_for_time_action, '%H:%M')
    # この例では、'%a', '%d', '%b', '%Y', '%H', '%M', '%S' という書式コードを使って、曜日、日、月、年、時、分、秒をそれぞれ表示しています。
    # str_td = dt_now_for_time_action.strftime("%a %d %b %Y %H:%M:%S")
    # str_td = dt_now_for_time_action.strftime("%H:%M")
    print(dt_now_for_time_action.days, dt_now_for_time_action.seconds)
    # str_td = f"{dt_now_for_time_action.hours:02d}:{dt_now_for_time_action.minutes:02d}"

    # print("現在時刻：", str_td) # dt_now_for_time_action)
    print("現在時刻：", dt_now_for_time_action)
    UserAction = "WALKING"
    # predictionmodel.run(str_td, UserAction)
    ret = predictionmodel.run(str(dt_now_for_time_action), UserAction)

    print(ret[0])
    print(type(ret[0]))