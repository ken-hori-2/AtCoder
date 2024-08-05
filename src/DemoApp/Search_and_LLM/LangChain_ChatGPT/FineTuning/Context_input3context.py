
from langchain_openai import ChatOpenAI # 新しいやり方
# from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# from langchain.chains import SimpleChain
import random
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

llm_4o=ChatOpenAI(
    model="gpt-4o",
    # model="gpt-3.5-turbo",
    temperature=0 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
)
llm_3p5t=ChatOpenAI(
    # model="gpt-4o",
    model="gpt-3.5-turbo",
    # model="ft:gpt-3.5-turbo-1106:personal:demoinput3context:9riJLf82",
    temperature=0 # 2 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
)
# llm_3p5t_tuning=ChatOpenAI(
#     model="ft:gpt-3.5-turbo-1106:personal:demoinput3context:9riJLf82",
#     temperature=0 # 2 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
# )

llm_3p5t_tuning=ChatOpenAI(
    # model="ft:gpt-3.5-turbo-1106:personal:demoin3con-ver2:9sgQ5awN", # user_data_input3context_ver2.jsonl
    # model="ft:gpt-3.5-turbo-1106:personal:newdemocon3inver2:9spEQboq", # [NEW] user_data_input3context_ver2.jsonl
    model="ft:gpt-3.5-turbo-1106:personal:demollm1:9spvnrK3",
    temperature=0 # 2 # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
    # temperature=1 # 2
)


class GenerateContext():

    def __init__(self):
        
        # 初期化
        # センサーデータの擬似入力
        self.sensor_data = {
            
            "time": "7:00",
            "action": "sitting",
            "environment": "office",

        }


        self.sensor_data2 = {
            # "time": "8:45:00",
            # "action": "STABLE",
            # "environment": "HOME"
            # "time": "8:00:00",
            # "action": "WALKING",
            # "environment": "OUTDOORS"
            
            "time": "12:05:00",
            "action": "WALKING",
            "environment": "OFFICE"
            # "time": "8:05:00",
            # "action": "WALKING",
            # "environment": "OUTDOORS"
        }

    # データを変動させるための擬似ランダム化
    def randomize_sensor_data(self, data):
        # actions = ["sitting", "standup", "gesture", "stable", "walking", "running"]
        actions = ["stable", "walking", "running"]
        environments = ["office", "train", "home", "outdoors", "talking", "car", "bus", "rain", "noisy"]
        time = ["7:00", "8:00", "10:00", "12:00", "15:00", "17:30", "19:00"]
        
        data["user_action"] = random.choice(actions) # motions)
        
        data["time"] = random.choice(time)
        data["environment"] =  random.choice(environments)
        
        # 今は指定
        data["time"] = "19:00" # "12:00" # "10:50" # "8:00" # "19:00" # "10:58" # "8:00"
        data["user_action"] = "walking" # "stable" # UserActionState # "walking"
        data["environment"] = "outdoors" # "train" # "office" # "talking" # "train" # "office" # random.choice(environments)
        
        return data
    
    def run(self):

        # sensor_data = self.randomize_sensor_data(self.sensor_data) # main
        sensor_data = self.randomize_sensor_data2(self.sensor_data2) # test

        
        # main
        prompt_template = """
        Based on the following sensor data, prompts representing the user's current intentions and needs are generated in a straightforward manner:

        - Current Time: {time}
        - User Action Status: {action}
        - Environmental Status: {environment}

        Provide a detailed description of the user's likely intention or need.
        """
        # Final output must be in Japanese.
        # """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["environment", "action", "time"])
        # 秘書という役職を与えると文が長くなる
        # prompt = ChatPromptTemplate.from_messages([
        #     # ("system", "You are secretary who is close to the user and answers potential needs.  (e.g., Users are doing XXX and have need to XX.)"),
        #     ("system", "You are the secretary who responds to the user's needs. (e.g., Users are doing XXX and have need to XX.)"),
        #     ("user", prompt_template)
        # ])

        # # ***** test *****
        # prompt_template = """
        #                     The current date and time is {time}, the current user action status is '{action}', and the status of the surrounding environment is '{environment}'.
        #                     In this case, only sentences that describe the user's situation in as much detail as possible are predicted and generated.
        #                     Final output must be in Japanese.
        #                     """
        # prompt = ChatPromptTemplate.from_messages([
        #     # ("system", "You are secretary who is close to the user and answers potential needs.  (e.g., Users are doing XXX and have need to XX.)"),
        #     ("system", "You are the secretary who responds to the user's needs. (e.g., Users are doing XXX and have need to XX.)"),
        #     ("user", prompt_template)
        # ])
        # # ***** test *****

        # チェーンを設定
        # chain = LLMChain(llm=llm_4o, prompt=prompt)
        output_parser = StrOutputParser()
        chain = prompt | llm_4o | output_parser
        # chain = prompt | llm_3p5t_tuning | output_parser
        # chain = prompt | llm_3p5t | output_parser

        # プロンプトを生成
        print("Sensor Data:", sensor_data)
        result = chain.invoke(sensor_data)

        print("Generated Prompt: ", result) # ['text']) # ["content"])

        return result # ['text']
        

    def randomize_sensor_data2(self, data):
        
        is_shuffle = True # False
        if is_shuffle:
            # actions = ["sitting", "standup", "gesture", "stable", "walking", "running"]
            actions = ["STABLE", "WALKING", "RUNNING"]
            environments = ["OFFICE", "TRAIN", "HOME", "OUTDOORS", "GYM"] # BUS, CAR, CONFERENCEROOM, LIVINGROOM, INDOORS 
            time = ["7:35", "8:02", "10:00", "12:02", "15:00", "17:30", "19:08"]
            
            data["action"] = random.choice(actions)
            data["time"] = random.choice(time)
            data["environment"] =  random.choice(environments)
            
            # # 今は指定
            # data["time"] = "19:00" # "12:00" # "10:50" # "8:00" # "19:00" # "10:58" # "8:00"
            # data["action"] = "walking" # "stable" # UserActionState # "walking"
            # data["environment"] = "outdoors" # "train" # "office" # "talking" # "train" # "office" # random.choice(environments)
        
        return data
    def run_for_dataset(self):
        # prompt_template = """
        #                     The current date and time is {time}, the current user action status is '{action}', and the status of the surrounding environment is '{environment}'.
        #                     In this case, only sentences that describe the user's situation in as much detail as possible are predicted and generated.
        #                     Final output must be in Japanese.
        #                     """
        # 学習時は上のデータ形式だったが、以下の形式でもしっかり答えられそう
        prompt_template = """
                        Based on the following sensor data, prompts representing the user's current intentions and needs are generated in a straightforward manner:

                        - Current Time: {time}
                        - User Action Status: {action}
                        - Environmental Status: {environment}

                        Provide a detailed description of the user's likely intention or need.
                        Final output must be in Japanese.
                        """
        # prompt_template = """
        #                 Based on the following sensor data, prompts representing the user's current intentions and needs are generated in a straightforward manner:

        #                 - Current Time: {time}
        #                 - User Action Status: {action}
        #                 - Environmental Status: {environment}

        #                 In this case, only sentences that explain the user's situation, intentions and needs in as much detail as possible are predicted and generated.
        #                 Final output must be in Japanese.
        #                 """
        # ,"Users are doing Route Search and have a need to find a route from current location to destination when going out."
        
        # prompt = PromptTemplate(template=prompt_template, input_variables=["time", "action", "environment"])

        # 秘書という役職を与えると文が長くなる
        # prompt = ChatPromptTemplate.from_messages([
        #     # ("system", "You are secretary who is close to the user and answers potential needs.  (e.g., Users are doing XXX and have need to XX.)"),
        #     ("system", "You are the secretary who responds to the user's needs. (e.g., Users are doing XXX and have need to XX.)"),
        #     ("user", prompt_template)
        # ])
        prompt = ChatPromptTemplate.from_messages([
            # ("system", "You are secretary who is close to the user and answers potential needs.  (e.g., Users are doing XXX and have need to XX.)"),
            
            # e.g.があったほうが想定の回答を得やすい
            (
                "system", "You are an excellent secretary, able to anticipate and make suggestions about the potential requirements of any user." #  (e.g., Users are doing 'XX' and have a need to XX)" # (e.g., Users are doing XXX and have need to XX.)"
            ),
            # e.g.がないと再学習時のような回答が得にくい
            # (
            #     "system", "You are an excellent secretary, able to anticipate and make suggestions about the potential requirements of any user."
            # ),
            (
                "user", prompt_template
            )
        ])
        
        output_parser = StrOutputParser()
        # chain = prompt | llm_4o | output_parser
        chain = prompt | llm_3p5t_tuning | output_parser # fine-tuning

        sensor_data2 = self.randomize_sensor_data2(self.sensor_data2)

        # プロンプトを生成
        print("Sensor Data:", sensor_data2)
        result = chain.invoke(sensor_data2)

        print("Generated Prompt: ", result) # ['text']) # ["content"])

        return result # ['text']
if __name__ == "__main__":

    # このファイルを直接実行する場合のみ必要
    from dotenv import load_dotenv
    load_dotenv()
    from langchain_core.output_parsers import StrOutputParser
    
    

    for i in range(1):

        prompt_answer = ""
        
        generate_context = GenerateContext()

        # context = generate_context.run() # main
        print("\n***********************************************************\n")
        context = generate_context.run_for_dataset() # Input 3 Context