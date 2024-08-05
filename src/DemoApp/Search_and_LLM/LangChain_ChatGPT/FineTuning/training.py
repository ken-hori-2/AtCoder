# # import openai

# # from dotenv import load_dotenv
# # # load_dotenv()
# # load_dotenv('..\\WebAPI\\Secret\\.env') # たぶんload_dotenv()のみでいい

# # # ファイル読み込み
# # test_data_file_object = openai.File.create(
# #   file=open("user_data.jsonl", "rb"),
# #   purpose='fine-tune'
# # )

# # file_id = test_data_file_object.id

# # # モデル作成Jobの実行
# # job_response = openai.FineTuningJob.create(training_file=file_id, model="gpt-3.5-turbo")

# # job_id = job_response.id
# # response_retrieve = openai.FineTuningJob.retrieve(job_id)
# # print(response_retrieve)

# # fine_tuned_model = response_retrieve.fine_tuned_model
# # print(fine_tuned_model)

# # response = openai.ChatCompletion.create(
# #     model=fine_tuned_model,
# #     messages=[
# #         {"role": "user", "content": "8:30にWALKINGをしています。何を提案できますか？"}
# #     ]
# # )

# # print(response["choices"][0]["message"]["content"])

# import openai
# from dotenv import load_dotenv

# # 環境変数をロード
# load_dotenv('..\\WebAPI\\Secret\\.env')  # 必要に応じてパスを修正

# # OpenAI APIキーを設定
# openai.api_key = 'your-api-key'  # 環境変数から読み込む場合は、os.getenv('OPENAI_API_KEY')を使用

# # ファイルをアップロード
# with open("user_data.jsonl", "rb") as file:
#     test_data_file_object = openai.File.create(
#         file=file,
#         purpose='fine-tune'
#     )

# file_id = test_data_file_object['id']

# # Fine-tuningジョブを作成
# job_response = openai.FineTune.create(training_file=file_id, model="gpt-3.5-turbo")

# job_id = job_response['id']

# # Fine-tuningジョブのステータスを取得
# response_retrieve = openai.FineTune.retrieve(job_id)
# print(response_retrieve)

# fine_tuned_model = response_retrieve['fine_tuned_model']
# print(fine_tuned_model)

# # Fine-tunedモデルを使用してチャットを作成
# response = openai.ChatCompletion.create(
#     model=fine_tuned_model,
#     messages=[
#         {"role": "user", "content": "8:30にWALKINGをしています。何を提案できますか？"}
#     ]
# )

# print(response["choices"][0]["message"]["content"])

# import os
# from dotenv import load_dotenv

# load_dotenv()

# print(os.environ['OPENAI_API_KEY'])

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])


import openai
import time
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

# res = client.files.create(
#   file=open("user_data.jsonl", "rb"),
#   purpose="fine-tune"
# )

# print( res.id )

# res = client.fine_tuning.jobs.create(
#   training_file=res.id, # "file-xxxxxxxxxxxxxxxxxxxx", # ここに file ID をコピー
#   model="gpt-3.5-turbo"
# )
# print( res.id )

base_model = 'gpt-3.5-turbo'
model = base_model
model = full_retraining(base_model, data)
# evaluate_model(model, test_dataset)