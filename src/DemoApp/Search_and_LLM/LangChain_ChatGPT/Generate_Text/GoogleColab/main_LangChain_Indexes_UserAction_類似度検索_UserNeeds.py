

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

"""
3. Vectorstoresの使い方
"""
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import TextLoader
from langchain_openai import OpenAI

# loader = TextLoader('./UserAction.txt') # , encoding='utf8')
# loader = TextLoader('./UserAction2.txt', encoding='utf8')
# loader = TextLoader('./UserAction2_mini.txt', encoding='utf8')
loader = TextLoader('./UserAction2_mini_loop.txt', encoding='utf8')

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


from langchain_openai import ChatOpenAI
llm=ChatOpenAI(
    model="gpt-4o",
    # model="gpt-3.5-turbo",
    temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
) # チャット特化型モデル
# query = """
#         あなたは人間のニーズを予測する専門家です。以下に答えてください。
#         ###
#         文書はユーザーの行動履歴です。
#         このユーザーの一日の行動はどのような傾向がありますか？
#         どのような時間や行動をするときにどんな機能を使う傾向にあるかを教えなければならない。
#         その際、使っている機能をすべて答え、どの時間でどの行動をしているときかも合わせて一日の時系列順にこたえて。(例:朝は～、午前中は～、昼は～、午後は～、夕方は～、夜は～)
#         """

# query = """
#         あなたはニーズを予測する専門家です。以下に答えて。
#         txtファイルはユーザーの「どの時刻に、どの行動状態で、どの機能を使用したか」の履歴です。
#         このユーザーの傾向を述べてください。(例: 1. 時間帯ごとの行動パターン: , 2. 行動状態: )
#         その後、現在が11:41、ユーザーの行動状態がWALKINGの場合、どの機能を提案するかこのユーザーの傾向を加味して予測して。
#         その際、各機能の提案する確率と最終的な提案(Final Answer:)、その理由も教えて。
#         あなたが提案できる機能は、
#         "会議情報", "楽曲再生", "経路検索", "リアルタイム情報検索", "レストラン検索", "ニュース情報", "天気情報"
#         です。
#         """

"""
重要 (2024/05/24)
"""
# これだとドキュメントは上から参照して似てたらそれを返しているように感じる
# なので、履歴を一旦要約して傾向にまとめて、その後それを入力する方がいいかも？
query = """
        あなたはニーズを予測する専門家です。以下に答えて。
        txtファイルの文書はユーザーの「どの時刻に、どの行動状態で、どの機能を使用したか」の履歴です。
        このユーザーの傾向を予測し箇条書きでまとめて。
        (例：朝の時間帯(xx:xx - xx:xx): ,午前中(xx:xx - xx:xx), 昼の時間帯(xx:xx - xx:xx), 午後(xx:xx - xx:xx), 夕方(xx:xx - xx:xx), 夜の時間帯(xx:xx - xx:xx))
        """

# こっちは別のLLMに入力する
query2 = """
        現在が11:41、ユーザーの行動状態がWALKINGの場合、どの機能を提案するかこのユーザーの傾向を加味して予測して。
        その際、各機能の提案する確率と最終的な提案(Final Answer:)、その理由も教えて。
        あなたが提案できる機能は、
        "会議情報", "楽曲再生", "経路検索", "リアルタイム情報検索", "レストラン検索", "ニュース情報", "天気情報"
        です。
        """





print(f"\n\n{query}")
# answer = index.query(query, llm = OpenAI(temperature=0))
# print(answer)

# with sourcesは、質問文と参考にしたデータのファイル名も表示してくれる
# answer_with_sources = index.query_with_sources(query, llm = OpenAI(temperature=0))
answer_with_sources = index.query_with_sources(query, llm = llm)
# print(answer_with_sources)
print(answer_with_sources['answer'])
print("-----")
print(answer_with_sources['sources']) # 出力されない

print("-----")
next_input = answer_with_sources['answer'] + query2 # chainが使えそう
print(f"\n\n{next_input}")