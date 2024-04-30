# from langchain.chat_models import ChatOpenAI
# from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import dotenv
dotenv.load_dotenv()
# # .envファイルの内容を読み込見込む
# load_dotenv()
# os.environ["OPENAI_API_KEY"]
# os.environ["GOOGLE_API_KEY"]
# os.environ["GOOGLE_CSE_ID"]

llm = ChatOpenAI(temperature=0.9)
prompt = ChatPromptTemplate.from_messages(
    [("human", "What is a good name for a company that makes {product}?")]
)

"""
1. LLMChainでは以下のように、文字列を入力し、文字列を出力として得ることができます。
"""
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain only specifying the input variable.
# print(chain.run("colorful socks"))
print("**********")
res = chain.invoke("colorful socks")
print(res) # ["output"])




"""
2. ChatOpenAIを使うと類似の処理が以下のように書けます。
"""
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
chat = ChatOpenAI()
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

input = [HumanMessage(content="What is a good name for a company that makes colorful socks?")]
output = chat(input)
print("**********")
print(output)

"""
LLMChainは内部でLLM（今回はChatOpenAI）を呼び出しているはずですので、どのように呼ばれているのかを追うのが目的です。
"""

