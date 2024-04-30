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
       "The input must consist of two location strings, one for the departure station and one for the destination station, and optionally one 0 or 1 character-type number for whether a shinkansen is required or not, and one character-type number for the priority of the order in which search results are displayed, either in order of fastest arrival:0 or cheapest fare:1 or fewest number of transfers:2 (e.g. Yokohama, Tokyo, 1, 2)"
       "Also, even if you are not directly instructed whether you need the shinkansen, use the shinkansen if you need to arrive earlier."
       # "In addition, if you use the shinkansen, please give priority to fewer transfers."
       "Therefore, a total of four arguments must be specified, three for each of the three claws and the fourth is initialized with the character type number 0 if not specified."
       
       # 出力形式を指定
       "When responding, write the final answer at the beginning, followed by the output in list format."
    )

    def _run(
        self, 
        departure_station: str,
        destination_station: str, 
        
        shinkansen: str, # オプションの引数ありバージョン
        search_results_priority: str, # オプションの引数ありバージョン2

        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the Route Search tool."""
        # return self.api_wrapper.run(location)
        return self.yahoo_search.run(departure_station, destination_station,     shinkansen, search_results_priority) # オプションの引数ありバージョン
    
    
    # これがないと動かない
    async def _arun(self, departure_station: str, destination_station: str, shinkansen: str, search_results_priority: str) -> str: # オプションの引数ありバージョン
        """Use the Route Search tool asynchronously."""
        raise NotImplementedError("RouteSearchQueryRun does not support async")
