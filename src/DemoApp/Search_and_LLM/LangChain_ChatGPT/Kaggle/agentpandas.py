# from langchain.agents import create_csv_agent, create_pandas_dataframe_agent
# from langchain.llms import OpenAI
# import pandas as pd
# df = pd.read_csv("penguin.csv")


# agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df=df, verbose=True)

# agent.invoke("全部で何件のデータがありますか？")


from dotenv import load_dotenv
load_dotenv()

# from langchain.agents import create_pandas_dataframe_agent
from langchain_experimental.agents import create_pandas_dataframe_agent
# from langchain_experimental.agents.agent_toolkits.spark.base import create_spark_dataframe_agent
from langchain.llms import OpenAI

from sklearn.datasets import fetch_openml
dataset = fetch_openml(data_id=40945, parser='auto')
# df = dataset['frame']

import pandas as pd
df= pd.read_csv("DemoTrain.csv")



print(df) #確認


agent = create_pandas_dataframe_agent(
    OpenAI(temperature=0),
    df,
    verbose=True,
)

prompt="""
タイタニック号には何人の乗客が乗っていたのでしょうか？
"""

prompt="""
男女別の生存率を計算してください。
変数sexの値ですが、0は女性で1が男性です。
変数survivedの値ですが、0は死亡で1が生存です。
"""

prompt="""
男女別の生存率を計算し、棒グラフで描いてください。
変数sexの値ですが、0は女性で1が男性です。
変数survivedの値ですが、0は死亡で1が生存です。
"""

prompt="""
生存を予測する分類モデルを作ってください。
データセットを、70%を学習データに、30%をテストデータに分割してください。
学習データでモデルを学習し、テストデータで評価して下さい。
テストデータの評価結果を教えてください。
"""
prompt="""
timeとactionからtoolを予測する分類モデルを作って。
データセットを、70%を学習データに、30%をテストデータに分割してください。
学習データでモデルを学習し、テストデータで評価して下さい。
テストデータの評価結果を教えてください。
"""
# 精度が90%以上になるようにモデルを作成、学習させてください。
# """


agent.invoke(prompt)