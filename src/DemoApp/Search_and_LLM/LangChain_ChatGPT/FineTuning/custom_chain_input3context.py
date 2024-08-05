from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()


"""
カスタムチェーン
・シチュエーションと要求を予測するテキスト
・楽曲再生が必要かどうかを確認
"""
class CustomChain:
    def __init__(self, time, user_status, environmental_status):
        self.time = time
        self.user_status = user_status
        self.env_status = environmental_status

        self.llm_demo = ChatOpenAI(model_name="ft:gpt-3.5-turbo-1106:personal:generateprompt:9qcfcS6N", temperature=0)
        # self.llm_details = ChatOpenAI(model_name="ft:gpt-3.5-turbo-1106:personal:genpromptdetails:9r0E1Bvv", temperature=0) # より詳細なデータにしたバージョン # 精度良くない

        self.output_parser = StrOutputParser()
        self.prompt_template = """
            Here is what you know about the user.
            [The current date and time is {time}, and The current user action state is {status}.]
            Predict and generate only sentences that describe the user's situation in as much detail as possible. (e.g., The user is doing XXX and has needs like XX)

            Answer each possible user request with its respective probability.(e.g., 1. xx:xx%, 2. xx:xx%, ...   :Total to be 100%.)


            Also, add one most likely user request to the output.(Most likely user request:)
        """
        #   Total to be 100%.
        # Give a number of possible user quests and the probability of each.

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the user's secretary. You are the expert who takes the user's potential needs and proposes solutions."),
            ("user", self.prompt_template)
        ])
        self.llm3p5t = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        self.prompt2 = PromptTemplate(
            input_variables=["text"],
            template="Return only True if this text is a music playback, otherwise return only False. {text}"
        )

    def generate_user_situation(self): # ユーザーのシチュエーションを予測
        chain1 = self.prompt | self.llm_demo | self.output_parser
        res = chain1.invoke({"time": self.time, "status": self.status})
        return res

    def check_music_playback(self, text): # 楽曲再生が必要かどうか確認
        # chain2 = self.prompt2 | self.llm_demo | self.output_parser
        chain2 = self.prompt2 | self.llm3p5t | self.output_parser
        is_playback = chain2.invoke(text)
        return is_playback

    def run(self):
        user_situation = self.generate_user_situation()
        print(f"input: The current date and time is {self.time}, and The current user action state is {self.status}.\n************************************************************")
        print(user_situation)
        is_playback = self.check_music_playback(user_situation)
        print(is_playback)

# 使用例
# time = '12:05'
# status = 'WALKING'

time = '8:30' # 00'
time = '9:10' # 30'
time = '10:50'
# time = '12:05'
# time = '17:30'
# time = '19:30'
# time = '20:00'
# time = '23:00'
# time = '0:00'

status = 'WALKING'
# status = 'STABLE'
# status = 'RUNNNING'

# time:12:05, status:STABLE [Do Nothing]


custom_chain = CustomChain(time, status)
custom_chain.run()