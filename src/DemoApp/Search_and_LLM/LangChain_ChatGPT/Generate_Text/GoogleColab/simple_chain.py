from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from dotenv import load_dotenv
load_dotenv()

llm = OpenAI(temperature=0)
prompt = PromptTemplate(
    # input_variables=["job"],
    input_variables=["time", "UserAction"],
    # template="{job}に一番オススメのプログラミング言語は何?"
    template="現在が{time}、ユーザーの行動状態が{UserAction}の場合どの機能を提案するか教えてください。"
)
chain = LLMChain(llm=llm, prompt=prompt, output_key="response")
# print(chain("データサイエンティスト"))
# print(chain({"11時30分", "WALKING"}))

from langchain.chains import SequentialChain
overall_chain = SequentialChain(
    chains=[chain],
    input_variables=["time", "UserAction"],
    output_variables=["response"], # あくまで辞書型のなんていう要素に出力が格納されるかの変数
    verbose=True,
)

response = overall_chain({
    # "time" : "11時30分",
    # "time" : "12時05分",
    "time" : "9時10分",
    # "UserAction" : "STABLE",
    "UserAction" : "WALKING",
    # "UserAction" : "RUNNING",
})
print(response['response'])