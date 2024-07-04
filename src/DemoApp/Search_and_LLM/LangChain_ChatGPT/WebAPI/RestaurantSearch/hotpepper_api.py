"""Tool for the OpenWeatherMap API."""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool

from RestaurantSearch.hotpepper_Lib import RestaurantSearch

class RestaurantSearchQueryRun(BaseTool):
    """Tool that queries the Route Search API."""

    # api_wrapper: OpenWeatherMapAPIWrapper = Field(
    #     default_factory=OpenWeatherMapAPIWrapper
    # )
    restaurant_search = RestaurantSearch()

    name: str = "restaurant_search"
    description: str = (
        "This function is useful for locating restaurants by location."

        # 2024/07/01 追加
        "It is often used for lunch."
        # 2024/07/01 追加


        "The input must be a string of two character type numbers, latitude and longitude, representing the location information, and one keyword about the restaurant. (e.g. 35.4394083, 139.3644221, ramen)."
        "So a total of three must be passed as input."
        "Restaurant selection criteria should be based on the highest rated restaurants that are close to the keyword(s)."
        "If you do not have specific keywords needed to select stores, such as food or genre, please let us know which stores are highly rated."
    )

    def _run(
        self, 
        latitude: str,
        longitude: str,
        keyword: str,

        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the Route Search tool."""
        # return self.api_wrapper.run(location)
        # 2024/07/02 ここでreturnする前に音声ガイダンスしてしまってもいいかも
        return self.restaurant_search.run(latitude, longitude, keyword) # , "Success" # オプションの引数ありバージョン
    
    
    # これがないと動かない
    # async def _arun(self, latitude: str, longitude: str, keyword: str) -> str: # オプションの引数ありバージョン
    #     """Use the Route Search tool asynchronously."""
    #     raise NotImplementedError("RestaurantSearchQueryRun does not support async")
    async def _arun(self, latitude: str, longitude: str, keyword: str) -> str: # オプションの引数ありバージョン
        """Use the Restaurant Search tool asynchronously."""
        raise NotImplementedError("RestaurantSearchQueryRun does not support async")
