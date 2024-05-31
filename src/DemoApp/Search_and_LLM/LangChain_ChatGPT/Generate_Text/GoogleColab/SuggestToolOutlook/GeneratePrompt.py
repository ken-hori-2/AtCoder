import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain_openai import ChatOpenAI
from langchain.chains import SequentialChain
from UIModelOutlookVer import UserInterfaceModel # ユーザーとのやり取りをするモデル

"""
モデルもどこか一か所にまとめる
"""
llm_4o=ChatOpenAI(
    model="gpt-4o",
    # model="gpt-3.5-turbo",
    temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
)
llm_3p5t=ChatOpenAI(
    # model="gpt-4o",
    model="gpt-3.5-turbo",
    temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
)




# 提案ツールからプロンプト生成
class GeneratePromptbyTool():

    def __init__(self, Suggested_Tool = "経路検索"):

        prompt_3 = PromptTemplate(
            input_variables=["SuggestedTool"],
            template = """
                    LLMに入力するための短いプロンプトを生成して。
                    具体的には「{SuggestedTool}」をもとに、ユーザーのニーズを満たせるようなプロンプトを簡潔に生成して。
                    ###
                    Prompt: (例：~を教えて。)
                    """
        )
        
        chain_3 = LLMChain(llm=llm_4o, prompt=prompt_3, output_key="output")
        # chain_3 = LLMChain(llm=llm_3p5t, prompt=prompt_3, output_key="output")

        self.overall_chain_3 = SequentialChain(
            chains=[chain_3],
            # input_variables=["UserNeeds", "SuggestedTool", "UserAction"],
            input_variables=["SuggestedTool"],
            # output_variables=["response"], # あくまで辞書型のなんていう要素に出力が格納されるかの変数
            verbose=True,
        )
        self.Suggested_Tool = Suggested_Tool
        self.userinterface = UserInterfaceModel()
    
    def execPrompt(self):
        Generated_Prompt = self.overall_chain_3({
            "SuggestedTool" : self.Suggested_Tool,
            })
        print("\n--------------------------------------------------")
        print(Generated_Prompt['output'])
        print("--------------------------------------------------")
        """
        LLMで入力の揺らぎに対処する
        """
    
    def getGeneratedPrompt(self):


        # ここのタイミングでユーザーに確認を求める
        # 例：「{Suggested_Tool}を提案します。実行しますか？」
        print(f"「{self.Suggested_Tool}を提案します。実行しますか？」")
        # UserInputExec = self.userinterface.recognize_speech() # 音声認識をする場 # UserInputExec = recognize_speech() # 音声認識をする場合
        UserInputExec = "YES"

        if ("はい" in UserInputExec) or ("YES" in UserInputExec) or ("イエス" in UserInputExec) or ("実行" in UserInputExec) or ("お願い" in UserInputExec):
            print("実行します。")
            
            self.execPrompt()
            


if __name__ == "__main__":
    generate_prompt = GeneratePromptbyTool()
    generate_prompt.getGeneratedPrompt()