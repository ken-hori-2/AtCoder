# ### 必要なライブラリのインポート

# # ```python
# from langchain import OpenAI, LLMChain
# from langchain.prompts import PromptTemplate
# from langchain.chains import SimpleChain
# import random
# # ```

# ### センサーデータの擬似入力

# # ここでは、センサーデータをシミュレートします。実際のアプリケーションでは、センサーデータはリアルタイムに取得される必要がありますが、この例では簡略化のために手動でデータを設定します。

# # ```python
# # センサーデータの擬似入力
# sensor_data = {
#     "location": "office",
#     "motion": "sitting",
#     "sound": "quiet",
#     "light": "bright",
#     "heart_rate": "stable"
# }

# # データを変動させるための擬似ランダム化
# def randomize_sensor_data(data):
#     motions = ["sitting", "walking", "running"]
#     sounds = ["quiet", "noisy", "talking"]
#     lights = ["bright", "dim", "dark"]
#     heart_rates = ["stable", "elevated", "low"]
    
#     data["motion"] = random.choice(motions)
#     data["sound"] = random.choice(sounds)
#     data["light"] = random.choice(lights)
#     data["heart_rate"] = random.choice(heart_rates)
    
#     return data

# sensor_data = randomize_sensor_data(sensor_data)
# # ```

# ### プロンプトテンプレートの作成

# # センサーデータに基づいてユーザーの意図を予測するためのプロンプトテンプレートを作成します。

# # ```python
# prompt_template = """
# Based on the following sensor data, generate a prompt that describes the user's current intention or need:

# - Location: {location}
# - Motion: {motion}
# - Sound: {sound}
# - Light: {light}
# - Heart Rate: {heart_rate}

# Provide a detailed description of the user's likely intention or need.
# """
# # ```

# ### LangChainを使用してプロンプトを生成

# # 次に、LangChainのチェーンを設定し、LLMを使用してプロンプトを生成します。

# # ```python
# # OpenAIのAPIキーを設定
# openai_api_key = "your-openai-api-key"

# # LLMのインスタンスを作成
# llm = OpenAI(api_key=openai_api_key, model="gpt-4")

# # プロンプトテンプレートを設定
# prompt = PromptTemplate(template=prompt_template, input_variables=["location", "motion", "sound", "light", "heart_rate"])

# # チェーンを設定
# chain = LLMChain(llm=llm, prompt=prompt)

# # プロンプトを生成
# result = chain.run(sensor_data)

# print("Generated Prompt: ", result)
# # ```

### 完全なコードの例

# 以下に、全体のコードをまとめます。これにより、センサーデータを元にユーザーの意図を生成するプロンプトをLangChainで生成できます。

# ```python
# from langchain import OpenAI, LLMChain
from langchain_openai import ChatOpenAI # 新しいやり方
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# from langchain.chains import SimpleChain
import random

from dotenv import load_dotenv
load_dotenv()

llm_4o=ChatOpenAI(
    model="gpt-4o",
    # model="gpt-3.5-turbo",
    temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
)
llm_3p5t=ChatOpenAI(
    # model="gpt-4o",
    model="gpt-3.5-turbo",
    temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
)

# OpenAIのAPIキーを設定
# openai_api_key = "your-openai-api-key"

# センサーデータの擬似入力
sensor_data = {
    "location": "office",
    "motion": "sitting",
    "sound": "quiet",
    # "light": "bright",
    "heart_rate": "stable"
}

# データを変動させるための擬似ランダム化
def randomize_sensor_data(data):
    locations = ["office", "train", "home"]
    data["location"] = random.choice(locations)

    # motions = ["sitting", "walking", "running"]
    motions = ["sitting", "standup", "gesture", "stable", "walking", "running"]
    sounds = ["quiet", "noisy", "talking"]
    # lights = ["bright", "dim", "dark"]
    heart_rates = ["stable", "elevated", "low"]
    
    data["motion"] = random.choice(motions)
    data["sound"] = random.choice(sounds)
    # data["light"] = random.choice(lights)
    data["heart_rate"] = random.choice(heart_rates)
    
    return data

sensor_data = randomize_sensor_data(sensor_data)

# プロンプトテンプレートを作成
# prompt_template = """
# Based on the following sensor data, generate a prompt that describes the user's current intention or need:

# - Location: {location}
# - Motion: {motion}
# - Sound: {sound}
# - Light: {light}
# - Heart Rate: {heart_rate}

# Provide a detailed description of the user's likely intention or need.
# """
prompt_template = """
Based on the following sensor data, prompts representing the user's current intentions and needs are generated in a straightforward manner:

- Location: {location}
- Motion: {motion}
- Sound: {sound}
- Heart Rate: {heart_rate}

Provide a detailed description of the user's likely intention or need.
Final output must be in Japanese.
"""

# LLMのインスタンスを作成
# llm = OpenAI(api_key=openai_api_key, model="gpt-4")

# プロンプトテンプレートを設定
# prompt = PromptTemplate(template=prompt_template, input_variables=["location", "motion", "sound", "light", "heart_rate"])
prompt = PromptTemplate(template=prompt_template, input_variables=["location", "motion", "sound", "heart_rate"])

# チェーンを設定
chain = LLMChain(llm=llm_4o, prompt=prompt)
# chain = prompt | llm_4o

# プロンプトを生成
print("Sensor Data:", sensor_data)
result = chain.invoke(sensor_data)

print("Generated Prompt: ", result['text']) # ["content"])