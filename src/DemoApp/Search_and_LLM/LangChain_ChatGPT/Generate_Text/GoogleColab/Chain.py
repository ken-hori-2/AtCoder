import os
from dotenv import load_dotenv
load_dotenv()

# SequentialChainの使い方
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.chains import SequentialChain

# llm = OpenAI(model_name="gpt-3.5-turbo")
# prompt_1 = PromptTemplate(
#     input_variables=["adjective", "job"],
#     template="{adjective}{job}に一番オススメのプログラミング言語は?\nプログラミング言語：",
# )
# chain_1 = LLMChain(llm=llm, prompt=prompt_1, output_key="programming_language")

# prompt_2 = PromptTemplate(
#     input_variables=["programming_language"],
#     template="{programming_language}を学ぶためにやるべきことを3ステップで100文字で教えて。",
# )
# chain_2 = LLMChain(llm=llm, prompt=prompt_2, output_key="learning_step")

# overall_chain = SequentialChain(
#     chains=[chain_1, chain_2], 
#     input_variables=["adjective", "job"],
#     output_variables=["programming_language", "learning_step"],
#     verbose=True,
# )

# # openai.chat.completions.create

# output = overall_chain({
#     "adjective": "ベテランの",
#     "job": "データサイエンティスト",
# })
# print(output)

from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain_openai import ChatOpenAI
llm=ChatOpenAI(
    # model="gpt-4o",
    model="gpt-3.5-turbo",
    temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
) # チャット特化型モデル

from typing import Dict, List

class ConcatenateChain(Chain):
    chain_1: LLMChain
    chain_2: LLMChain

    @property
    def input_keys(self) -> List[str]:
        all_input_vars = set(self.chain_1.input_keys).union(set(self.chain_2.input_keys))
        return list(all_input_vars)

    @property
    def output_keys(self) -> List[str]:
        return ['concat_output']

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        output_1 = self.chain_1.run(inputs)
        output_2 = self.chain_2.run(inputs)
        return {'concat_output': output_1 + "\n" + output_2}

# llm = OpenAI(model_name="gpt-3.5-turbo")

prompt_1 = PromptTemplate(
    input_variables=["job"],
    template="{job}に一番オススメのプログラミング言語は?\nプログラミング言語：",
)
chain_1 = LLMChain(llm=llm, prompt=prompt_1)

prompt_2 = PromptTemplate(
    input_variables=["job"],
    template="{job}の平均年収は？\n平均年収：",
)
chain_2 = LLMChain(llm=llm, prompt=prompt_2)

concat_chain = ConcatenateChain(chain_1=chain_1, chain_2=chain_2, verbose=True)
print(concat_chain.run("データサイエンティスト"))