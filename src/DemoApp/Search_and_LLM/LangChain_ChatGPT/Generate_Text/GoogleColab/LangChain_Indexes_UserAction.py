

# conda環境で実行する必要がある

import os

"""
1. Document Loadersの使い方
"""
# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

# loader = PyPDFLoader("https://blog.freelance-jp.org/wp-content/uploads/2023/03/FreelanceSurvey2023.pdf")
# pages = loader.load_and_split()
# print(pages[0])

# chroma_index = Chroma.from_documents(pages, OpenAIEmbeddings())
# docs = chroma_index.similarity_search("「フリーランスのリモートワークの実態」について教えて。", k=2)
# for doc in docs:
#     print(str(doc.metadata["page"]) + ":", doc.page_content)

"""
2. Text Splittersの使い方
"""
from langchain.text_splitter import CharacterTextSplitter

# long_text = """
# GPT-4は、OpenAIが開発したAI技術であるGPTシリーズの第4世代目のモデルです。

# 自然言語処理(NLP)という技術を使い、文章の生成や理解を行うことができます。

# これにより、人間と同じような文章を作成することが可能です。

# GPT-4は、トランスフォーマーアーキテクチャに基づいており、より強力な性能を発揮します。

# GPT-4は、インターネット上の大量のテキストデータを学習し、豊富な知識を持っています。

# しかし、2021年9月までの情報しか持っていません。

# このモデルは、質問応答や文章生成、文章要約など、様々なタスクで使用できます。

# ただし、GPT-4は完璧ではありません。

# 時々、誤った情報や不適切な内容を生成することがあります。

# 使用者は、その限界を理解し、

# 適切な方法で利用することが重要です。
# """
# print(len(long_text))

# text_splitter = CharacterTextSplitter(        
#     separator = "\n\n",
#     chunk_size = 100,
#     chunk_overlap = 0,
#     length_function = len,
# )
# text_list = text_splitter.split_text(long_text)
# print(text_list)
# print(len(text_list))

# document_list = text_splitter.create_documents([long_text])
# print(document_list)
# print(len(document_list))

"""
3. Vectorstoresの使い方
"""
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import TextLoader
from langchain_openai import OpenAI

# long_text = """
# GPT-4は、OpenAIが開発したAI技術であるGPTシリーズの第4世代目のモデルです。

# 自然言語処理(NLP)という技術を使い、文章の生成や理解を行うことができます。

# これにより、人間と同じような文章を作成することが可能です。

# GPT-4は、トランスフォーマーアーキテクチャに基づいており、より強力な性能を発揮します。

# GPT-4は、インターネット上の大量のテキストデータを学習し、豊富な知識を持っています。

# しかし、2021年9月までの情報しか持っていません。

# このモデルは、質問応答や文章生成、文章要約など、様々なタスクで使用できます。

# ただし、GPT-4は完璧ではありません。

# 時々、誤った情報や不適切な内容を生成することがあります。

# 使用者は、その限界を理解し、

# 適切な方法で利用することが重要です。
# """
# print(len(long_text))
# with open("./UserAction.txt", "r") as f:
#     # f.write(long_text)
#     f.close()

loader = TextLoader('./UserAction.txt') # , encoding='utf8')

# 100文字のチャンクで区切る
text_splitter = CharacterTextSplitter(        
    separator = "\n\n",
    chunk_size = 100,
    chunk_overlap = 0,
    length_function = len,
)

# VectorStoreIndexの中身.pyにに中身の処理についての説明
"""
このVectorStoreIndexCreatorが様々な機能が一つにまとまっている
"""
index = VectorstoreIndexCreator(
    vectorstore_cls=Chroma, # Default
    embedding=OpenAIEmbeddings(), # Default
    text_splitter=text_splitter, # text_splitterのインスタンスを使っている
).from_loaders([loader])

# 最初のプロンプト
query = "Q1. 文書はユーザーの行動履歴です。このユーザーの一日の行動はどのような傾向がありますか？どのような時間や行動をするときにどんな機能を使う傾向にあるかをできるだけ、長く細かく教えて。"
print(f"\n\n{query}")
# answer = index.query(query, llm = OpenAI(temperature=0))
# print(answer)






# # 中身の処理から
# texts = text_splitter.split_text(loader)
# docsearch = Chroma.from_texts(texts, OpenAIEmbeddings())
# docs = docsearch.similarity_search(query)
# # print("類似度の高い文章：", docs)
# print("\n##### 文書内から類似度の高い文章を抽出： #####")
# print(docs[0].page_content)
# print("##############################################")






# with sourcesは、質問文と参考にしたデータのファイル名も表示してくれる
answer_with_sources = index.query_with_sources(query, llm = OpenAI(temperature=0))
print(answer_with_sources)

# query = "Q2. GPT4は第何世代のモデル？"
# query = "今日の東京の天気は？"
# print(f"\n\n{query}")
# # answer = index.query(query, llm = OpenAI(temperature=0))
# # print(answer)

# answer_with_sources = index.query_with_sources(query, llm = OpenAI(temperature=0))
# print(answer_with_sources)
