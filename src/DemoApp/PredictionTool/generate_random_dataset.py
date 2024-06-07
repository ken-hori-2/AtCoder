import random
from datetime import timedelta, datetime

# データセットを格納するリスト
data = []

# 行動状態とツール名の選択肢
states = ['STABLE', 'WALKING', 'RUNNING']
tools = ['楽曲再生', '会議情報', '天気情報', 'レストラン検索', '動画視聴', 'ニュース閲覧', '経路検索']

# ランダムな時刻を生成する関数
def random_time():
    return (datetime.combine(datetime.today(), datetime.min.time()) + 
            timedelta(minutes=random.randint(0, 1439))).strftime('%H:%M')

# データセットを生成
for _ in range(1000):
    time = random_time()
    state = random.choice(states)
    tool = random.choice(tools)
    data.append((time, state, tool))

# 結果を表示（最初の10件のみ）
for i in range(10):
    print(data[i])