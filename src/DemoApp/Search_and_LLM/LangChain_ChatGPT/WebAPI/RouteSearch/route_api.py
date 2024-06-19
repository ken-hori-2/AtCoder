"""Tool for the OpenWeatherMap API."""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool

# from langchain_community.utilities.openweathermap import OpenWeatherMapAPIWrapper
from RouteSearch.route_search_Lib import RouteSearch

from DateTime.WhatTimeIsItNow import SetTime

class RouteSearchQueryRun(BaseTool):
    """Tool that queries the Route Search API."""

    # api_wrapper: OpenWeatherMapAPIWrapper = Field(
    #     default_factory=OpenWeatherMapAPIWrapper
    # )
    yahoo_search = RouteSearch()

    name: str = "route_search"
    description: str = (
        "This function is useful to get station routing information from a specified station."
        # "The input must be two location strings, one for the origin station and one for the destination station (e.g., Yokohama, Tokyo)."
        
       # オプションの引数ありバージョン
       # <重要> 変数名と同じキーワードを記述するとLLMも認識できる
       # "The input must be two location strings, one for the departure station and one for the destination station, and optionally one character type number for whether or not a shinkansen is needed (e.g., Yokohama, Tokyo, 1)."
       # 明示していなくても要求からパラメータを設定
       "The input must consist of two location strings, one for the departure station and one for the destination station, and optionally one 0 or 1 character-type number for whether a shinkansen is required or not, and one character-type number for the priority of the order in which search results are displayed, either in order of fastest arrival:0 or cheapest fare:1 or fewest number of transfers:2 (e.g. 横浜, 東京, 1, 2)" # Yokohama, Tokyo, 1, 2)"
       "Also, even if you are not directly instructed whether you need the shinkansen, use the shinkansen if you need to arrive earlier."
       # "In addition, if you use the shinkansen, please give priority to fewer transfers."
       "Therefore, a total of four arguments must be specified, three for each of the three claws and the fourth is initialized with the character type number 0 if not specified."
       
       # 出力形式を指定
        #    "When responding, write the final answer at the beginning, followed by the output in list format."
        # "You must include bullet points and transfer stations in your response."
        
        
        
        
        
        
        # ローマ字入力にしないと帰宅時の経路案内が失敗する
        # "Departure and destination must be entered in romaji." # 逆にコメントアウトにしないと出社時は失敗する
        # "Departure and arrival stations must be entered in Japanese."
    )

    # def __init__(self, dt_now):
    #     self.dt_now_arg = dt_now

    def _run(
        self, 

        # dt_now_arg, # エラーになる可能性がある（省略事実引数を使うとエラー回避できる）

        departure_station: str,
        destination_station: str, 
        
        shinkansen: str, # オプションの引数ありバージョン
        search_results_priority: str, # オプションの引数ありバージョン2

        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the Route Search tool."""
        # return self.api_wrapper.run(location)
        # dt_now_arg = self.dt_now_arg
        # return self.yahoo_search.run(dt_now_arg, departure_station, destination_station,     shinkansen, search_results_priority) # オプションの引数ありバージョン

        # """
        # SCO DEMO (6/19)
        # """
        # import datetime
        # YYYY = 2024
        # MM = 6
        # DD = 19
        # # dt_now = datetime.datetime(YYYY, MM, DD, 7, 10)        # 天気情報 (今日より前の日付だとエラーになるかも)
        # dt_now = datetime.datetime(YYYY, MM, DD, 8, 00) # 30)    # 出勤     (stable:楽曲再生[house-music], walking:経路検索)
        # # dt_now = datetime.datetime(YYYY, MM, DD, 10, 55)         # 定例     (stable:何もしない, walk:会議情報)
        # # # dt_now = datetime.datetime(YYYY, MM, DD, 12, 5)        # 昼食     (walk:restaurant, stable:music[relax-music])
        # # dt_now = datetime.datetime(YYYY, MM, DD, 19, 5)          # ジム     (run:up tempo, walk:slow tempo, stable:stop)   # 行動検出と連動モード
        set_time = SetTime()
        dt_now = set_time.run()
        return self.yahoo_search.run(dt_now, departure_station, destination_station,     shinkansen, search_results_priority) # オプションの引数ありバージョン
    
    
    # これがないと動かない
    async def _arun(self, dt_now_arg, departure_station: str, destination_station: str, shinkansen: str, search_results_priority: str) -> str: # オプションの引数ありバージョン
        """Use the Route Search tool asynchronously."""
        raise NotImplementedError("RouteSearchQueryRun does not support async")
