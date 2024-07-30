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

res = client.files.create(
  file=open("user_data.jsonl", "rb"),
  purpose="fine-tune"
)

print( res.id )

res = client.fine_tuning.jobs.create(
  training_file=res.id, # "file-xxxxxxxxxxxxxxxxxxxx", # ここに file ID をコピー
  model="gpt-3.5-turbo"
)
print( res.id )