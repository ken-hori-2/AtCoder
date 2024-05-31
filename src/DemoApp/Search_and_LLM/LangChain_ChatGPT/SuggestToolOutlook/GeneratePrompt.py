import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain_openai import ChatOpenAI
from langchain.chains import SequentialChain
import sys
# from UIModelOutlookVer import UserInterfaceModel # ユーザーとのやり取りをするモデル
# JudgeModel.pyから呼び出すときに必要
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from SuggestToolOutlook.UIModelOutlookVer import UserInterfaceModel # ユーザーとのやり取りをするモデル

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





class GeneratePromptbyTool():

    """
    # 提案ツールからプロンプト生成する場合
    """
    def __init__(self, Suggested_Tool):

        prompt_3 = PromptTemplate(
            input_variables=["SuggestedTool"],
            # template = """
            #         LLMに入力するための短いプロンプトを生成して。
            #         具体的には「{SuggestedTool}」をもとに、ユーザーのニーズを満たせるようなプロンプトを簡潔に生成して。
            #         ###
            #         Prompt: (例：~を教えて。)
            #         """
            # template = """
            #         LLMに入力するための短いプロンプトを生成して。
            #         具体的には「{SuggestedTool}」というキーワードをもとに、LLMに要求を命令するプロンプトを生成して。
                    
            #         経路やレストランの検索など、場所の情報が必要な場合は
            #         「情報が不足する場合は、予定から今日のユーザーの目的地を取得し、それを使って経路やレストランの情報を取得して」という文を含めてプロンプトを生成して。
            #         ###
            #         Prompt: 
            #         """
            template = """
                    LLMに入力するための短いプロンプトを生成して。
                    具体的には「{SuggestedTool}して。」というキーワードを含めて、LLMに要求を命令するプロンプトを生成して。
                    経路やレストランの検索など、場所の情報が必要な場合は
                    「ユーザーに関する情報が足りない場合は予定を参照し出社場所を取得して。検索結果が複数ある場合は3件までにして。」という文を含めてプロンプトを生成して。
                    ###
                    Prompt: 
                    """
                    # 楽曲再生を提案するときは気分の上がるような楽曲を再生するようなプロンプトにして。


            # template = """
            #           LLMに入力するための短いプロンプトを生成して。
            #           具体的には「{SuggestedTool}」という機能をもとに、LLMに入力するためのプロンプトを簡潔に生成して。
            #           ###
            #           Prompt: (例：~を教えて。)
            #           """
        )
        
        # chain_3 = LLMChain(llm=llm_4o, prompt=prompt_3, output_key="output")
        chain_3 = LLMChain(llm=llm_3p5t, prompt=prompt_3, output_key="output")

        self.overall_chain_3 = SequentialChain(
            chains=[chain_3],
            # input_variables=["UserNeeds", "SuggestedTool", "UserAction"],
            input_variables=["SuggestedTool"],
            # output_variables=["response"], # あくまで辞書型のなんていう要素に出力が格納されるかの変数
            verbose=True,
        )
        self.Suggested_Tool = Suggested_Tool
        self.userinterface = UserInterfaceModel()
    
    def getGeneratedPrompt(self):
        Generated_Prompt = self.overall_chain_3({
            "SuggestedTool" : self.Suggested_Tool,
            })
        print("\n--------------------------------------------------")
        print(Generated_Prompt['output'])
        print("--------------------------------------------------")
        """
        LLMで入力の揺らぎに対処する
        """
        return Generated_Prompt['output']
    
    """
    # 機能をそのままプロンプトにする場合
    """
    # def __init__(self, Suggested_Tool):
    #     self.Suggested_Tool = Suggested_Tool
    #     self.userinterface = UserInterfaceModel()
    
    # def getGeneratedPrompt(self):
    #     pass


    
    def getJudgeResult(self):


        # ここのタイミングでユーザーに確認を求める
        # 例：「{Suggested_Tool}を提案します。実行しますか？」
        # print(f"「{self.Suggested_Tool}を提案します。実行しますか？」")
        self.userinterface.text_to_speach(f"「{self.Suggested_Tool}を提案します。実行しますか？」\n")
        
        for i in range(10):


            # UserInputExec = self.userinterface.recognize_speech() # 音声認識をする場 # UserInputExec = recognize_speech() # 音声認識をする場合
            UserInputExec = "YES"
            isExecute = False

        
            if ("はい" in UserInputExec) or ("YES" in UserInputExec) or ("イエス" in UserInputExec) or ("実行" in UserInputExec) or ("お願い" in UserInputExec):
                # print("実行します。")
                self.userinterface.text_to_speach("実行します。\n")
                
                # self.execPrompt()
                isExecute = True
                break
            elif ("いいえ" in UserInputExec) or ("しないで" in UserInputExec) or ("いらない" in UserInputExec):
                break
        
        return isExecute
            


if __name__ == "__main__":
    tool = "レストラン検索"
    generate_prompt = GeneratePromptbyTool(tool)
    isExecuteTool = generate_prompt.getJudgeResult()
    if isExecuteTool:
        prompt_answer = generate_prompt.getGeneratedPrompt()