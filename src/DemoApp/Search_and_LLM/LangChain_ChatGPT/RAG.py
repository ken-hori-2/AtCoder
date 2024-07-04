
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import PromptTemplate
from langchain import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI

llm_4o=ChatOpenAI(
    model="gpt-4o",
    # model="gpt-3.5-turbo",
    temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
) # チャット特化型モデル
llm_3p5t=ChatOpenAI(
    # model="gpt-4o",
    model="gpt-3.5-turbo",
    temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
) # チャット特化型モデル

class OutputUserTrends():
    def __init__(self):
        self.loader = TextLoader('./SuggestToolTimeAction/userdata_context_version.txt', encoding='utf8')
        self.index = VectorstoreIndexCreator(embedding= HuggingFaceEmbeddings()).from_loaders([self.loader])
        # self.callbacks = [StreamingStdOutCallbackHandler()]
        self.llm=llm_4o
    
    def getUserTrends(self):
        results = self.index.vectorstore.similarity_search("What are the user trends?", k=4)
        context = "\n".join([document.page_content for document in results])
        # print(f"{context}")
        template = """
        Please use the following context to answer questions.
        Context: {context}
        ---
        Question: {question}
        Answer: Let's think step by step.
        Final output must be in Japanese.
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)
        llm_chain = LLMChain(prompt=prompt, llm=self.llm)
        response = llm_chain.invoke("What are the user trends?")
        print(response['text'])

        return response['text']

if __name__ == "__main__":

    output_usertrends = OutputUserTrends()
    user_trends = output_usertrends.getUserTrends()
