# """Tool for the OpenWeatherMap API."""

# from langchain.tools.base import BaseTool
# from langchain.utilities import OpenWeatherMapAPIWrapper

# class OpenWeatherMapQueryRun(BaseTool):
#     """Tool that adds the capability to query using the OpenWeatherMap API."""

#     api_wrapper: OpenWeatherMapAPIWrapper

#     name = "OpenWeatherMap"
#     description = (
#         "A wrapper around OpenWeatherMap API. "
#         "Useful for fetching current weather information for a specified location. "
#         "Input should be a location string (e.g.    'London,GB')."
#     )

#     def __init__(self) -> None:
#         self.api_wrapper = OpenWeatherMapAPIWrapper()
#         return

#     def _run(self, location: str) -> str:
#         """Use the OpenWeatherMap tool."""
#         return self.api_wrapper.run(location)

#     async def _arun(self, location: str) -> str:
#         """Use the OpenWeatherMap tool asynchronously."""
#         raise NotImplementedError("OpenWeatherMapQueryRun does not support async")
"""Tool for the OpenWeatherMap API."""

from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool

from langchain_community.utilities.openweathermap import OpenWeatherMapAPIWrapper


class OpenWeatherMapQueryRun(BaseTool):
    """Tool that queries the OpenWeatherMap API."""

    api_wrapper: OpenWeatherMapAPIWrapper = Field(
        default_factory=OpenWeatherMapAPIWrapper
    )

    name: str = "open_weather_map"
    description: str = (
        "A wrapper around OpenWeatherMap API. "
        "Useful for fetching current weather information for a specified location. "
        "Input should be a location string (e.g. London,GB)."

        "If it is entered in a language other than English, it must be translated into English before it can be entered." # 2024/05/14 追加
    )

    def _run(
        self, location: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the OpenWeatherMap tool."""
        return self.api_wrapper.run(location)
    
    
    # これがないと動かない
    async def _arun(self, location: str) -> str:
        """Use the OpenWeatherMap tool asynchronously."""
        raise NotImplementedError("OpenWeatherMapQueryRun does not support async")
