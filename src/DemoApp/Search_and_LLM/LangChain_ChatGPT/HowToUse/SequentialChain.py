from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain

llm = OpenAI() # model_name="text-davinci-003")
prompt_1 = PromptTemplate(
		input_variables=["job"],
		template="{job}に一番オススメのプログラミング言語は何?"
)
chain_1 = LLMChain(llm=llm, prompt=prompt_1)

prompt_2 = PromptTemplate(
    input_variables=["programming_language"],
    template="{programming_language}を学ぶためにやるべきことを3ステップで100文字で教えて。",
)
chain_2 = LLMChain(llm=llm, prompt=prompt_2)

overall_chain = SimpleSequentialChain(chains=[chain_1, chain_2], verbose=True)
print(overall_chain("データサイエンティスト"))




# 実行結果
# > Entering new SimpleSequentialChain chain...

# Pythonがデータサイエンティストに最もオススメのプログラミング言語です。Pythonは、統計分析、機械学習、データ可視化などのデータサイエンティストにとって重要な分野をカバーしており、プログラミング言語としても容易に学習できるために、多くのデータサイエンティストにとって最も理想的なプログラミング言語として認識されています。

# 1.Pythonを学ぶために、まずは基本的な構文を習得します。
# 2.データサイエンティスト向けのPythonライブラリーを使用してデータ分析を行います。
# 3.データ可視化ツールを使用してデータを可視化します。

# > Finished chain.
# {'input': 'データサイエンティスト', 'output': '\n\n1.Pythonを学ぶために、まずは基本的な構文を習得します。\n2.データサイエンティスト向けのPythonライブラリーを使用してデータ分析を行います。\n3.データ可視化ツールを使用してデータを可視化します。'}









"""
先ほど説明したSimpleSequentialChainは、ソースコードを見てもわかる通り、引数と返り値は1つである必要があります。
そこで、複数の引数と返り値を用いたい場合は、SimpleSequentialChainではなく「SequentialChain」を使いましょう。
"""

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

llm = OpenAI() # model_name="text-davinci-003")
prompt_1 = PromptTemplate(
    input_variables=["adjective", "job"],
    template="{adjective}{job}に一番オススメのプログラミング言語は?\nプログラミング言語：",
)
chain_1 = LLMChain(llm=llm, prompt=prompt_1, output_key="programming_language")

prompt_2 = PromptTemplate(
    input_variables=["programming_language"],
    template="{programming_language}を学ぶためにやるべきことを3ステップで100文字で教えて。",
)
chain_2 = LLMChain(llm=llm, prompt=prompt_2, output_key="learning_step")

overall_chain = SequentialChain(
    chains=[chain_1, chain_2],
    input_variables=["adjective", "job"],
    output_variables=["programming_language", "learning_step"],
    verbose=True,
)
output = overall_chain({
    "adjective": "ベテランの",
    "job": "データサイエンティスト",
})
print(output)


# > Entering new SequentialChain chain...

# > Finished chain.
# {'adjective': 'ベテランの', 'job': 'データサイエンティスト', 'programming_language': 'Python', 'learning_step': '\n\n1.基本文法を学ぶ:Pythonの基本文法をマスターするために、書籍やチュートリアルなどを参照しながら学習する。\n\n2.プログラミングを実践:実際にプログラミングを行うことで、文法を身につける。ハンズオン形式のチュートリアルを参考に、実際にコードを書いてみる。\n\n3.実務経験を積む:面接で実務経験を聞かれたときに答えられるように、実務でPythonを使った開発経験を積ん'}