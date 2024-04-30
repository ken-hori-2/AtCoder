from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
# from langchain.llms import OpenAI
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI # 新しいやり方
import os

# APIキーをセット (変数名はLangChain側で決められています)
# open_api_key = os.environ["openai_api_key"]
# serpapi_api_key = os.environ["serpapi_api_key"]
# agentと agentが使用するtool
from langchain.agents import Tool, initialize_agent, AgentType
import os
from dotenv import load_dotenv
# .envファイルの内容を読み込見込む
load_dotenv()
from langchain_google_community import GoogleSearchAPIWrapper # 新しいやり方
# agent が使用するGoogleSearchAPIWrapperのツールを作成
search = GoogleSearchAPIWrapper()

# 言語モデルを指定し、ツールをセットアップ
llm = OpenAI(temperature=0)
# agent の使用する LLM
# llm=ChatOpenAI(temperature=0)

# load_tools : 組み込み済みのAPIをロード ⇒自作の物や覚えるためにはあんまり使わなくてもいいかも
# tools = load_tools(["serpapi", "llm-math"], llm=llm)
tools = load_tools(["google-search", "llm-math", "wikipedia"], llm=llm)

# エージェントに能力を付与し、Zero-Shot & ReActで機能させることを定義 
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# デフォルトの場合、裏側がtext-davince-003なので日本語もいけます
# agent.run("東京都の人口は、日本の総人口の何パーセントを占めていますか？")
agent.invoke("東京都の人口は、日本の総人口の何パーセントを占めていますか？計算して求めてください。")
# agent.invoke("123×123を計算して。")
