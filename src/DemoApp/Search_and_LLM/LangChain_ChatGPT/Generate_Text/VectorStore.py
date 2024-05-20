# from langchain_community.document_loaders import CSVLoader

# from langchain.text_splitter import CharacterTextSplitter
# docs = CSVLoader("data/restaurant.csv", encoding="utf-8").load()
# for d in docs:
#     d.page_content = d.page_content.replace("\n", " ") 
# # text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10, separator="。")
# text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10, separator="。")
# chunks = text_splitter.split_documents(docs)
# for i, chunk in enumerate(chunks):
#     print(f"{i:2d}: length={len(chunk.page_content):3d} 行:{chunk.metadata['row']:1d} {chunk.page_content[:30]}")




# FAISS
from dotenv import load_dotenv
load_dotenv()
# from langchain_community.document_loaders import TextLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# # docs = TextLoader('./data/restaurant.txt', encoding="utf-8").load()
# docs = CSVLoader("data/restaurant.csv", encoding="utf-8").load()
# for d in docs:
#     d.page_content = d.page_content.replace("\n", " ") 
# # text_splitter = CharacterTextSplitter(chunk_size=50,
# #     chunk_overlap=0, separator="　")
# text_splitter = CharacterTextSplitter(chunk_size=50,
#       chunk_overlap=0, separator="。")
# chunks = text_splitter.split_documents(docs)
# contents = [d.page_content for d in chunks]

# embeddings = OpenAIEmbeddings()
# faiss = FAISS.from_documents(chunks, embedding=embeddings)
# faiss.save_local("faiss-db")

# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# embeddings = OpenAIEmbeddings()
# db = FAISS.load_local("faiss-db", embeddings)
# r = db.similarity_search("竹藪", k=2)
# print(r)

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import CSVChain

csv_path = "data/restaurant.csv"
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_csv(csv_path, embeddings)

chain = CSVChain(vectorstore=vector_store)