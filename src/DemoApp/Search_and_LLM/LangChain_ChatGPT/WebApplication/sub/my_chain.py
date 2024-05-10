from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain

template = """
関西弁に変換してください.
変換前:{input}
変換後:
"""


def create_chain() -> Chain:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)
    prompt = PromptTemplate(template=template, input_variables=["input"])
    return LLMChain(prompt=prompt, llm=llm, verbose=True)
