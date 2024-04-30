from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = OpenAI(model_name="text-davinci-003")
prompt = PromptTemplate(
    input_variables=["job"],
    template="{job}に一番オススメのプログラミング言語は何?"
)
chain = LLMChain(llm=llm, prompt=prompt)
print(chain("データサイエンティスト"))




"""
ここではLLMのモデルで実行しましたが、ChatGPTのようなChat ModelでChainを作りたいときには、LLMの引数にChatOpenAIなどChat Modelのインスタンスを入れれば、OKです。
"""
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import LLMChain

human_message_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["job"],
				template="{job}に一番オススメのプログラミング言語は何?"
    )
)

chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
chain = LLMChain(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
    prompt=chat_prompt_template
)

print(chain("データサイエンティスト"))