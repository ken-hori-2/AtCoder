"""Tool for the OpenWeatherMap API."""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool

# from langchain_community.utilities.openweathermap import OpenWeatherMapAPIWrapper
# from RouteSearch.route_search_Lib import RouteSearch
from WebAPI.Schedule.OutlookSchedule_Lib import OutlookSchedule

from DateTime.WhatTimeIsItNow import SetTime

class ScheduleQueryRun(BaseTool):
    """Tool that queries the Schedule Information API."""

    # api_wrapper: OpenWeatherMapAPIWrapper = Field(
    #     default_factory=OpenWeatherMapAPIWrapper
    # )
    outlook_schedule = OutlookSchedule()

    # name: str = "schedule_information"
    name: str = "Schedule-Information"

    description: str = (
        "This function is useful for retrieving schedule information from a specified outlook appointment list."
        "Get information about the next meeting."

        # 2024/07/01 追加
        # dt_nowをOutlookSchedule_api.py内で取得する場合はいらない
        "The input should be a time string (e.g. 2024-7-1 7:00)." # "It takes the current time as an argument. This is used as the basis for the search. (e.g. 2024-7-1 7:00)"
        # 2024/07/01 追加

        
        # 2024/05/28
        "It also helps users to know their departure station and destination station, which is necessary when searching for routes to commute or travel." # ユーザーが通勤や移動する際の経路検索時に必要な、出発地や目的地を知るのにも役立ちます。
        "This is especially useful when the starting point or destination is unknown." # 特に出発地や目的地が不明な時に役立ちます。
    )

    def _run(
        self, 
        # dt_now_arg, # エラーになる可能性がある（省略事実引数を使うとエラー回避できる）
        
        dt_now_arg: str, # dt_nowをOutlookSchedule_api.py内で取得する場合はいらない

        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the Schedule Information tool."""
        # return self.api_wrapper.run(location)
        # return self.yahoo_search.run(departure_station, destination_station,     shinkansen, search_results_priority) # オプションの引数ありバージョン
        # return self.outlook_schedule.run()

        
        
        
        
        
        # dt_nowをOutlookSchedule_api.py内で取得する場合に必要
        # set_time = SetTime()
        # dt_now_arg = set_time.run()


        self.outlook_schedule.run(dt_now_arg)
        self.outlook_schedule.MTG_ScheduleItem()
        return self.outlook_schedule.getMeetingContents() # , "Success"
    
    
    

    
    
    # dt_nowをOutlookSchedule_api.py内で取得する場合に必要
    # async def _arun(self) -> str: # オプションの引数ありバージョン
    #     """Use the Schedule Information tool asynchronously."""
    #     raise NotImplementedError("ScheduleQueryRun does not support async")
    async def _arun(self, dt_now_arg: str) -> str: # オプションの引数ありバージョン
        """Use the Schedule Information tool asynchronously."""
        raise NotImplementedError("ScheduleQueryRun does not support async")
