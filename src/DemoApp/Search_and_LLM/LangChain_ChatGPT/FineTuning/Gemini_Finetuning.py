import google.generativeai as genai

import os
from dotenv import load_dotenv
# .envファイルの読み込み
load_dotenv()
# API-KEYの設定
GOOGLE_GEMINI_API_KEY=os.getenv('GOOGLE_GEMINI_API_KEY')
genai.configure(api_key=GOOGLE_GEMINI_API_KEY)


# # import os
# # if 'COLAB_RELEASE_TAG' in os.environ:
# import pathlib
# pathlib.Path('client_secret.json') # .write_text(userdata.get('CLIENT_SECRET'))

# # # Use `--no-browser` in colab
# # !gcloud auth application-default login --no-browser --client-id-file client_secret.json --scopes='https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/generative-language.tuning'
# # else:
# #   !gcloud auth application-default login --client-id-file client_secret.json --scopes='https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/generative-language.tuning'

# # ファインチューニングモデルの確認
# for model in genai.list_tuned_models(): # google_api_key=GOOGLE_GEMINI_API_KEY):
#     print(model.name)

# from google.oauth2 import service_account
# from google.auth.transport.requests import Request
# from google.generativeai import models

# # サービスアカウントキーのパス
# SERVICE_ACCOUNT_FILE = 'client_secret.json' # 'path/to/service-account-file.json'

# # 認証情報を取得
# credentials = service_account.Credentials.from_service_account_file(
#     SERVICE_ACCOUNT_FILE,
#     scopes=['https://www.googleapis.com/auth/cloud-platform']
# )

# # 認証情報をリフレッシュ
# credentials.refresh(Request())

# # 認証情報を使用してAPIクライアントを作成
# client = models.ModelServiceClient(credentials=credentials)

# # チューニングされたモデルのリストを取得
# for model in client.list_tuned_models():
#     print(model)

# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# from google.generativeai import models

# # OAuth 2.0 フローを設定
# flow = InstalledAppFlow.from_client_secrets_file(
#     'client_secret.json',
#     scopes=['https://www.googleapis.com/auth/cloud-platform']
# )

# # 認証情報を取得
# credentials = flow.run_local_server(port=0)

# # 認証情報をリフレッシュ
# credentials.refresh(Request())

# # # 認証情報を使用してAPIクライアントを作成
# # client = models.ModelServiceClient(credentials=credentials)

# # # チューニングされたモデルのリストを取得
# # for model in client.list_tuned_models():
# # # for model in genai.list_tuned_models():
# #     print(model)

# from google.cloud import aiplatform

# # 認証情報を使用してAPIクライアントを作成
# client = aiplatform.gapic.ModelServiceClient(credentials=credentials)

# # チューニングされたモデルのリストを取得
# for model in client.list_models(): # parent="projects/your-project/locations/your-location"):
#     print(model)

# from google.cloud import aiplatform
# from google.oauth2 import service_account

# # サービスアカウントキーのパス
# SERVICE_ACCOUNT_FILE = 'path/to/service-account-file.json'

# # 認証情報を取得
# credentials = service_account.Credentials.from_service_account_file(
#     # SERVICE_ACCOUNT_FILE,
#     'client_secret.json', 
#     scopes=['https://www.googleapis.com/auth/cloud-platform']
# )

# # 認証情報を使用してAPIクライアントを作成
# client = aiplatform.gapic.ModelServiceClient(credentials=credentials)

# # プロジェクトとロケーションを指定
# parent = "projects/your-project/locations/your-location"

# # チューニングされたモデルのリストを取得
# for model in client.list_models(parent=parent):
#     print(model)

# """
# Install the Google AI Python SDK

# $ pip install google-generativeai

# See the getting started guide for more information:
# https://ai.google.dev/gemini-api/docs/get-started/python
# """

# import os

# import google.generativeai as genai
# from langchain_google_genai import ChatGoogleGenerativeAI

# # genai.configure(api_key=os.environ["GEMINI_API_KEY"])
# genai.configure(api_key=GOOGLE_GEMINI_API_KEY)

# import pathlib
# pathlib.Path("client_secret_2.json")

# # Create the model
# generation_config = {
#   "temperature": 0.9,
#   "top_p": 1,
#   "max_output_tokens": 8192,
#   "response_mime_type": "text/plain",
# }

# model = genai.GenerativeModel(
#   model_name="tunedModels/geminidemoapp-iobrohy5rkoc",
#   generation_config=generation_config,
#   # safety_settings = Adjust safety settings
#   # See https://ai.google.dev/gemini-api/docs/safety-settings
# )
# llm_gemini = ChatGoogleGenerativeAI(
#     # model=base_model,
#     model = "tunedModels/geminidemoapp-iobrohy5rkoc",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     # other params...
#     google_api_key=GOOGLE_GEMINI_API_KEY # GOOGLE_GEMINI_API_KEYという名前で使う場合は指定が必要（GOOGLE_API_KEYという名前なら指定しなくても認識してくれるが、かぶると使えない？ので別途指定するのがいいかも）
# )

# response = model.generate_content([
#   "input: ",
#   "output: ",
# ])

# print(response.text)

from google.cloud import aiplatform

# # サービスアカウントキーのパスを設定
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'client_secret_gemini.json'
# print(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))

from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file(
    # 'client_secret_gemini.json'
    'test_secret.json'

)

# プロジェクトIDを設定
# project = "gen-lang-client-0031315644"

# モデルの地域を設定
# location = "us-central1"

# モデルの名前を設定
model_name = "tunedModels/geminidemoapp-iobrohy5rkoc"

# 予測クライアントを作成
prediction_client = aiplatform.gapic.PredictionServiceAsyncClient()

# 予測リクエストを作成
instances = [
    {
        "text": "あなたの質問"
    }
]
parameters = {
    # 必要に応じてパラメータを設定
}
request = aiplatform.gapic.PredictRequest(
    endpoint=model_name,
    instances=instances,
    parameters=parameters
)

# 予測を実行
response = prediction_client.predict(request=request)
print(response.predictions)
