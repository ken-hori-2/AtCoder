import os
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, load_tools, AgentType, Tool
# from tool_directory import ToolLoader
from weather_api import OpenWeatherMapQueryRun
from langchain.agents import load_tools
from dotenv import load_dotenv
# .envファイルの内容を読み込見込む
load_dotenv()
from tool_directory import ToolLoader

# OpenAIのAPI KEY
open_weather_api_key = os.environ['OPENWEATHERMAP_API_KEY'] #  = '<OpenAIのAPI KEYをここに指定>'

# 使用したいtool
llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
# endpoint = "https://ipinfo.io/" # "https://ipinfo.io"


token = os.environ['IPINFO_API_KEY'] # ipinfo API KEY


tools = []
tools.extend(ToolLoader('openweather').get_tools(parameters={'appid': open_weather_api_key}))
tools.extend(ToolLoader('ipinfo').get_tools(parameters={'token': token}))
tools.extend(ToolLoader('timeapi').get_tools())

# 初期化
# llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
agent = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# 実行
# answer = agent('日本のタイムゾーンで今の時刻を教えて。')
# answer = agent('今の時刻を教えて。')
answer = agent('現在位置と最寄り駅、時刻を教えて。')
