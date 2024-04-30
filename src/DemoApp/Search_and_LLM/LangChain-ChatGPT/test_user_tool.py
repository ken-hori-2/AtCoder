# OpenAI のラッパーをインポート
from langchain.llms import OpenAI

# LLM ラッパーを初期化
llm = OpenAI(temperature=0.7)

# SerpAPIWrapperのインポート
from langchain import SerpAPIWrapper, LLMMathChain

# serpapiのラッパーを初期化
search = SerpAPIWrapper()

# llm_math_chainのラッパーを初期化
llm_math_chain = LLMMathChain(llm=llm, verbose=True)

# カスタムツールを定義して、ツール一覧に追加
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    )
]