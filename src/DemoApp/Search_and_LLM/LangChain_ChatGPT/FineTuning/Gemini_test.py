from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import sys
import os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from pathlib import Path
# sys.path.append(str(Path('__file__').resolve().parent.parent)) # LangChain_ChatGPTまでのパス
# print(sys.path)
from dotenv import load_dotenv
# .envファイルの読み込み
# load_dotenv()
load_dotenv('..\\WebAPI\\Secret\\.env')

# # API-KEYの設定
# GOOGLE_GEMINI_API_KEY=os.getenv('GOOGLE_GEMINI_API_KEY') # GOOGLE_GEMINI_API_KEYという名前で使う場合は指定が必要（GOOGLE_API_KEYという名前なら指定しなくても認識してくれるが、かぶると使えない？ので別途指定するのがいいかも）
# genai.configure(api_key=GOOGLE_GEMINI_API_KEY)

""" エラーメッセージ """
# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# langchain-google-community 1.0.3 requires langchain-core<0.2,>=0.1.33, but you have langchain-core 0.2.28 which is incompatible.
# langchain-openai 0.1.4 requires langchain-core<0.2.0,>=0.1.46, but you have langchain-core 0.2.28 which is incompatible.


base_model = "gemini-1.5-flash" # (Input:0.35$/MtoK, Output:0.53$/MtoK) # FlashはAPIのみ使用可能なモデル
# "gemini-1.5-pro" (Input:3.5$/MtoK, Output:10.5$/MtoK)
# "gpt-4o" (Input:5$/MtoK, Output:15$/MtoK)

print("***************************************************************************************************************************************")
# # 最初に試したもの
# # llm = ChatGoogleGenerativeAI(model=base_model)
# llm = ChatGoogleGenerativeAI(model=base_model, google_api_key=GOOGLE_GEMINI_API_KEY) # "gemini-pro") # , google_api_key=GOOGLE_GEMINI_API_KEY)
# # result = llm.invoke("どのプログラミング言語を学べば良いですか？最もオススメの言語を一つだけ教えてください。")


# result = llm.invoke("日本の総理大臣は誰ですか？")
# print(result.content)


# # from langchain_google_genai import ChatGoogleGenerativeAI

# # llm = ChatGoogleGenerativeAI(
# #     model=base_model,
# #     temperature=0,
# #     max_tokens=None,
# #     timeout=None,
# #     max_retries=2,
# #     # other params...
# #     google_api_key=GOOGLE_GEMINI_API_KEY # envを使う場合はいらない
# # )

# messages = [
#     (
#         "system",
#         # "You are a helpful assistant that translates English to French. Translate the user sentence.",
#         "You are a helpful assistant that translates English to Japanese. Translate the user sentence.",
#     ),
#     ("human", "I love programming."),
# ]
# # ai_msg = llm.invoke(messages)
# # print(ai_msg)

# # print(ai_msg.content)

print("***************************************************************************************************************************************")

# # from langchain_core.prompts import ChatPromptTemplate

# # prompt = ChatPromptTemplate.from_messages(
# #     [
# #         (
# #             "system",
# #             "You are a helpful assistant that translates {input_language} to {output_language}.",
# #         ),
# #         ("human", "{input}"),
# #     ]
# # )

# # chain = prompt | llm
# # res = chain.invoke(
# #     {
# #         "input_language": "English",
# #         "output_language": "Japanese", # German",
# #         "input": "I love programming.",
# #     }
# # )
# # print(res)

# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# output_parser = StrOutputParser()
# llm_gemini = ChatGoogleGenerativeAI(
#     model=base_model,
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     # other params...
#     google_api_key=GOOGLE_GEMINI_API_KEY # GOOGLE_GEMINI_API_KEYという名前で使う場合は指定が必要（GOOGLE_API_KEYという名前なら指定しなくても認識してくれるが、かぶると使えない？ので別途指定するのがいいかも）
# )
# prompt_template = """
#                         Based on the following sensor data, prompts representing the user's current intentions and needs are generated in a straightforward manner:

#                         - Current Time: {time}
#                         - User Action Status: {action}
#                         - Environmental Status: {environment}

#                         Provide a detailed description of the user's likely intention or need.
#                         Final output must be in Japanese.
#                         """
#         # ,"Users are doing Route Search and have a need to find a route from current location to destination when going out."
        
# prompt = PromptTemplate(template=prompt_template, input_variables=["time", "action", "environment"])
# sensor_data = {
#                 "time": "12:05:00",
#                 "action": "WALKING",
#                 "environment": "OFFICE"
#             }
# chain = prompt | llm_gemini | output_parser

# # result = chain.invoke(sensor_data)
# # print("Generated Prompt: ", result)



#     # messages = [
#     #     (
#     #         "system",
#     #         "You are a helpful assistant that translates English to Japanese. Translate the user sentence.",
#     #     ),
#     #     ("human", "I love programming."),
#     # ]



# messages_2 = [
#     {
#         "role": 
#             "system", 
#         "content": 
#             "A secretary who is close to the user and anticipates and answers potential needs.  (e.g., Users are doing XXX and have need to XX.)"
#     }, 
#     {
#         "role": 
#             "user", 
#         "content": 
#             "Based on the following sensor data, prompts representing the user's current intentions and needs are generated in a straightforward manner: \
#                 [- Current Time: '14:55:00', - User Action Status: 'STABLE', - Environmental Status: 'OFFICE']. Provide a detailed description of the user's likely intention or need. \
#                     Final output must be in Japanese."
#     }, 
#         # {
#         #     "role": "assistant", 
#         #     "content": "Users are doing 'Meeting Info Check' and needs to know the details of the next meeting."
#         # }
# ]
#     # messages_2 = [
#     #     (
#     #         "system", 
#     #         "A secretary who is close to the user and anticipates and answers potential needs.  (e.g., Users are doing XXX and have need to XX.)",
#     #     ), 
#     #     (
#     #         "user", 
#     #         "Based on the following sensor data, prompts representing the user's current intentions and needs are generated in a straightforward manner: [- Current Time: '14:55:00', - User Action Status: 'STABLE', - Environmental Status: 'OFFICE']. Provide a detailed description of the user's likely intention or need.",
#     #     )
#     # ]
# # chain2 = prompt | llm_gemini | output_parser
# chain2 = llm_gemini | output_parser
# result = chain2.invoke(messages_2)
# # result = llm_gemini.invoke(messages_2)
# print("Generated Prompt: ", result) # ['content'])
print("***************************************************************************************************************************************")







# fine tuning test
# import os
# if 'COLAB_RELEASE_TAG' in os.environ:
#   from google.colab import userdata
#   import pathlib
#   pathlib.Path('client_secret.json').write_text(userdata.get('CLIENT_SECRET'))

  # Use `--no-browser` in colab
#   !gcloud auth application-default login --no-browser --client-id-file client_secret.json --scopes='https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/generative-language.tuning'
# else:
#   !gcloud auth application-default login --client-id-file client_secret.json --scopes='https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/generative-language.tuning'

# name = 'geminidemoapp-iobrohy5rkoc'
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# from google.generativeai import models
# # OAuth 2.0 フローを設定
# flow = InstalledAppFlow.from_client_secrets_file(
#     # 'client_secret_gemini.json',
#     # 'client_secret_2.json',
#     "gemini_secret.json",
#     scopes=['https://www.googleapis.com/auth/cloud-platform'] # ['https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/generative-language.tuning']
# )

# # 認証情報を取得
# credentials = flow.run_local_server(port=0)
from google.cloud import aiplatform
import google.auth
# import datetime, re
# import googleapiclient.discovery
# 認証情報を使用してAPIクライアントを作成

# ①Google APIの準備をする
SCOPES = ['https://www.googleapis.com/auth/cloud-platform'] # ['https://www.googleapis.com/auth/calendar']
# calendar_id = 'stmuymte@gmail.com' # '自身のGoogleカレンダーIDを記述'
# Googleの認証情報をファイルから読み込む
credentials = google.auth.load_credentials_from_file('..\\WebAPI\\Secret\\credentials.json', SCOPES)[0]
client = aiplatform.gapic.ModelServiceClient(credentials=credentials)

# calendar_id = 'stmuymte@gmail.com' # '自身のGoogleカレンダーIDを記述'
# # Googleの認証情報をファイルから読み込む
# scopes=['https://www.googleapis.com/auth/cloud-platform']
# client = google.auth.load_credentials_from_file('client_secret_gemini.json', scopes)[0]
# # APIと対話するためのResourceオブジェクトを構築する
# # service = googleapiclient.discovery.build('calendar', 'v3', credentials=gapi_creds)

# # base_model = client.get_tuned_model(f'tunedModels/geminidemoapp-iobrohy5rkoc') # {name}')
# from google.cloud import aiplatform
# # クライアントの初期化
# # client = aiplatform.gapic.ModelServiceClient()

# # モデルの取得
model_name = "tunedModels/geminidemoapp-iobrohy5rkoc" # projects/YOUR_PROJECT_ID/locations/YOUR_LOCATION/models/YOUR_MODEL_ID"
# model = client.get_model(name=model_name)
# print("Model details:", model)
# base_model = model_name
# print("Model details:", base_model)
# # # model = genai.GenerativeModel(model_name=f'tunedModels/{name}')
# # result = base_model.generate_content('55')
# # print("tuning model : ", result.text)
# 予測の実行

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '..\\WebAPI\\Secret\\credentials.json'
prediction_client = aiplatform.gapic.PredictionServiceClient() # credentials=credentials)
try:
    response = prediction_client.predict(
        # name=model_name,
        endpoint=model_name,
        instances=[{"input": "your input data"}]
    )

    # # 予測リクエストを作成
    # instances = [
    #     {
    #         "text": "あなたの質問"
    #     }
    # ]
    # parameters = {
    #     # 必要に応じてパラメータを設定
    # }
    # request = aiplatform.gapic.PredictRequest(
    #     endpoint=model_name,
    #     instances=instances,
    #     parameters=parameters
    # )

    # # 予測を実行
    # response = prediction_client.predict(request=request)
    # # print(response.predictions)

    print("Prediction results:", response)
except google.api_core.exceptions.NotFound:
    print("Prediction service not found. Please check the model ID and location.")
except google.api_core.exceptions.MethodNotImplemented:
    print("The method is not implemented. Please check the API documentation.")
# fine tuning test




from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()
# prompt_test = PromptTemplate(template=prompt_template, input_variables=["time", "action", "environment"])
llm_gemini = ChatGoogleGenerativeAI(
    model=base_model,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
    # google_api_key=GOOGLE_GEMINI_API_KEY # GOOGLE_GEMINI_API_KEYという名前で使う場合は指定が必要（GOOGLE_API_KEYという名前なら指定しなくても認識してくれるが、かぶると使えない？ので別途指定するのがいいかも）
)
sensor_data = {
                "time": "12:05:00",
                "action": "WALKING",
                "environment": "OUTDOORS" # "TRAIN" # "OFFICE"
              }
prompt_template = """
                  Based on the following sensor data, prompts representing the user's current intentions and needs are generated in a straightforward manner:

                  - Current Time: {time}
                  - User Action Status: {action}
                  - Environmental Status: {environment}

                  Provide a detailed description of the user's likely intention or need.
                  Final output must be in Japanese.
                  """
prompt = ChatPromptTemplate.from_messages([
            (
                "system", "You are the user's secretary. You are the expert who takes the user's potential needs and proposes solutions."
            ),
            (
                "user", prompt_template
            )
        ])
# music playback check
    # prompt2 = PromptTemplate(
    #             input_variables=["text"],
    #             template="Return only True if this text is a music playback, otherwise return only False. {text}"
    #         )
chain_gemini = prompt | llm_gemini | output_parser
res = chain_gemini.invoke(sensor_data)
    # res = chain_gemini.invoke( # 上と同じ、sensor_dataという変数に辞書を格納するか、直接辞書を渡すかの違い
    #     {
    #         "time": "12:01", 
    #         "action":"RUNNING", 
    #         "environment":"GYM"
    #     }
    # )
    # print(prompt.format({"time": "12:01", "action":"RUNNING", "environment":"GYM"}))

print("answer:", res)