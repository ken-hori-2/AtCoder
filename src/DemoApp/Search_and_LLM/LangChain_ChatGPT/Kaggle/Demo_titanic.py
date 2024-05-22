
from langchain.llms import OpenAI
# from langchain.agents import create_pandas_dataframe_agent
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
llm=ChatOpenAI(
    # model="gpt-4o",
    model="gpt-3.5-turbo",
    temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
) # チャット特化型モデル

df = pd.read_csv('titanic.csv')
agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
# agent = create_pandas_dataframe_agent(llm, df, verbose=True)
agent.run("whats the square root of the average age?")