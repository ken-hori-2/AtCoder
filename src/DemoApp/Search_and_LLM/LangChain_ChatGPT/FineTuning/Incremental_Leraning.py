# 修正と詳細な説明を踏まえた新しいスクリプトを提供します。

# ### 重要なポイント
# 1. **インクリメンタルラーニング**:
#     - 再学習したモデルを引数で渡し続けて、新しいデータを追加していく。
# 2. **全モデルの再学習**:
#     - 既存のファインチューニングされていないモデル（ベースモデル）を使って、全データを一から学習する。

# ### 必要なライブラリのインストール
# まず、必要なライブラリをインストールします。

# ```bash
# pip install openai pandas scikit-learn datasets
# ```

# ### 完全なスクリプト
# 以下が修正後の完全なスクリプトです。

# ```python
import os
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets import Dataset
import openai
from dotenv import load_dotenv
load_dotenv()

# OpenAI APIキーの設定
# openai.api_key = 'your-api-key'

# データ収集と前処理
def collect_data(data_dir):
    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    data_frames = [pd.read_csv(f) for f in data_files]
    data = pd.concat(data_frames, ignore_index=True)
    return data

def preprocess_data(data):
    data = data.dropna()
    data['text'] = data['text'].apply(lambda x: x.lower())
    return data

# データセットの準備
def prepare_dataset(data):
    return Dataset.from_pandas(data)

# インクリメンタルラーニング
def incremental_learning(model, new_data):
    # 新しいデータでファインチューニング（OpenAI APIの仕組み上、インクリメンタルラーニングはファインチューニングとして扱います）
    print("Incremental learning with new data")
    # OpenAIのファインチューニングエンドポイントを使用してモデルを更新
    response = openai.FineTune.create(
        training_file=new_data['text'].to_list(),
        model=model
    )
    fine_tune_id = response['id']
   
    # ファインチューニングが完了するまで待機
    while True:
        status_response = openai.FineTune.retrieve(id=fine_tune_id)
        if status_response['status'] == 'succeeded':
            model = status_response['fine_tuned_model']
            break
        elif status_response['status'] == 'failed':
            raise Exception("Fine-tuning failed")
        time.sleep(60)  # 1分ごとにステータスを確認
   
    return model

# 全モデルの再学習
def full_retraining(base_model, data):
    # 全データで再学習（OpenAI APIの仕組み上、全モデルの再学習もファインチューニングとして扱います）
    print("Full retraining with all data")
    # OpenAIのファインチューニングエンドポイントを使用してモデルを再学習
    response = openai.FineTune.create(
        training_file=data['text'].to_list(),
        model=base_model
    )
    fine_tune_id = response['id']
   
    # ファインチューニングが完了するまで待機
    while True:
        status_response = openai.FineTune.retrieve(id=fine_tune_id)
        if status_response['status'] == 'succeeded':
            model = status_response['fine_tuned_model']
            break
        elif status_response['status'] == 'failed':
            raise Exception("Fine-tuning failed")
        time.sleep(60)  # 1分ごとにステータスを確認
   
    return model

# モデルの評価
def evaluate_model(model, test_data):
    # テストデータを用いてモデルを評価
    print("Evaluating model")
    predictions = []
    for text in test_data['text']:
        response = openai.Completion.create(
            model=model,
            prompt=text,
            max_tokens=1,
            n=1,
            stop=None,
            temperature=0.0
        )
        predictions.append(response['choices'][0]['text'])
   
    accuracy = accuracy_score(test_data['label'], predictions)
    print(f'Model accuracy: {accuracy}')

# メイン処理
def main():
    data_dir = 'path/to/new/data'
    data = collect_data(data_dir)
    data = preprocess_data(data)
    train_data, test_data = train_test_split(data, test_size=0.2)
    train_dataset = prepare_dataset(train_data)
    test_dataset = prepare_dataset(test_data)

    # ベースモデルの設定
    base_model = 'gpt-4'
    model = base_model

    # 定期的なインクリメンタルラーニングと全モデルの再学習
    try:
        while True:
            model = incremental_learning(model, train_dataset)
            evaluate_model(model, test_dataset)
           
            # 一定期間ごとに全モデルの再学習を実行
            if time.time() % (30 * 24 * 60 * 60) < 60:
                model = full_retraining(base_model, data)
                evaluate_model(model, test_dataset)
           
            time.sleep(7 * 24 * 60 * 60)  # 7日ごとにインクリメンタルラーニングを実行
    except KeyboardInterrupt:
        print("Process interrupted")

if __name__ == "__main__":
    main()
# ```

# ### スクリプトの説明
# 1. **データ収集と前処理**:
#     - `collect_data`関数でデータを収集し、`preprocess_data`関数で前処理を行います。

# 2. **インクリメンタルラーニング**:
#     - `incremental_learning`関数で、新しいデータを用いてモデルを部分的に更新します。ファインチューニングが完了するまで待機し、更新されたモデルを返します。

# 3. **全モデルの再学習**:
#     - `full_retraining`関数で、ベースモデルを使って全データを再学習します。ファインチューニングが完了するまで待機し、更新されたモデルを返します。

# 4. **モデルの評価**:
#     - `evaluate_model`関数で、新しいデータを用いてモデルの評価を行います。

# 5. **メイン処理**:
#     - `main`関数で、上記のプロセスを順次実行します。インクリメンタルラーニングは7日ごとに実行され、全モデルの再学習は30日ごとに実行されます。

# このスクリプトを実行することで、データ収集から前処理、インクリメンタルラーニング、全モデルの再学習、評価までのプロセスを自動化できます。再学習の頻度は、実際の要件に応じて調整してください。