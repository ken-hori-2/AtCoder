
from langchain.llms import OpenAI
# from langchain.agents import create_pandas_dataframe_agent
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

df = pd.read_csv('titanic.csv')
agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
agent.run("whats the square root of the average age?")