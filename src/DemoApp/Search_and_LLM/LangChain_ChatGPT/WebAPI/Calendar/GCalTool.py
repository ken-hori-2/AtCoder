import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Calendar.GoogleCalendarLib import GoogleCalendar

from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.chains import LLMChain
import datetime
from langchain import PromptTemplate 

from Calendar.Pydantic import CalendarAction

class GoogleCalendarTool():
    def __init__(self, credentials_file, llm=None, time_zone='JST', memory=None, prompt="""Follow the user query and take action on the calendar appointments.
        Current time: {current_time}, timeZone: JST.
        History: {chat_history}
        Format: {format_instructions}

        User Query: {query}
        Processing and reporting must be done in Japanese. If unclear, do not process and ask questions.""" ,scopes=['https://www.googleapis.com/auth/calendar']):

        self.cal = GoogleCalendar(credentials_file)
        self.llm = llm
        if self.llm is None:
            raise ValueError("Error: LLM is undefined.")
        self.time_zone = time_zone
        self.memory = memory
        # Parser に元になるデータの型を提供する
        self.parser =  PydanticOutputParser(pydantic_object=CalendarAction)
        self.prompt = PromptTemplate(
            template=prompt,
            input_variables=["query", "current_time", "chat_history"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def run(self, query):
        _input = self.prompt.format_prompt(query=query, current_time=datetime.datetime.now(), chat_history=self.memory)
        output = self.chain.run({'query': query, 'current_time': datetime.datetime.now(), 'chat_history': self.memory})
        # output = self.chain.invoke({'query': query, 'current_time': datetime.datetime.now(), 'chat_history': self.memory})

        return self.cal.process_json_data(output)