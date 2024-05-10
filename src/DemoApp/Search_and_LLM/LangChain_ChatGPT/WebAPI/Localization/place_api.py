# import googlemaps
# import pprint # list型やdict型を見やすくprintするライブラリ
# # key = 'Your API' # 上記で作成したAPIキーを入れる
# # 必要モジュールのインポート
# import os
# from dotenv import load_dotenv
# # .envファイルの内容を読み込見込む
# load_dotenv()
# # os.environを用いて環境変数を表示させます
# print(os.environ['GOOGLE_MAPS_API_KEY'])
# key = os.environ['GOOGLE_MAPS_API_KEY']


# # client = googlemaps.Client(key) #インスタンス生成

# # # geocode_result = client.geocode('東京都大森駅') # 位置情報を検索
# # # geocode_result = client.geocode('東京都後楽園駅') # 位置情報を検索
# # geocode_result = client.geocode('本厚木駅') # 位置情報を検索
# # # print(geocode_result)
# # loc = geocode_result[0]['geometry']['location'] # 軽度・緯度の情報のみ取り出す

# # # # place_result = client.places_nearby(location=loc, radius=200, type='food') #半径200m以内のレストランの情報を取得
# # # place_result = client.places_nearby(location=loc, radius=100, type='food') #半径100m以内のレストランの情報を取得
# # # pprint.pprint(place_result)

# # print("loc : ", loc)




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
from Localization.localization_Lib import Localization

class LocalizationQueryRun(BaseTool):
    """Tool that queries the Localization API."""

    # api_wrapper: OpenWeatherMapAPIWrapper = Field(
    #     default_factory=OpenWeatherMapAPIWrapper
    # )
    googlemap_PlaceAPI = Localization()

    name: str = "localization"
    description: str = (
        "This function is useful to get the latitude and longitude from a keyword for a place name."
        "Input must be a string of one keyword related to the place name. (e.g. Hon Atsugi Station)."
    )

    def _run(
        self, 
        location: str,

        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the Route Search tool."""
        return self.googlemap_PlaceAPI.run(location)
    
    
    # これがないと動かない
    async def _arun(self, localization: str) -> str: # オプションの引数ありバージョン
        """Use the Route Search tool asynchronously."""
        raise NotImplementedError("RouteSearchQueryRun does not support async")




# """
# Weather_toolより
# """
# from langchain.prompts import ChatPromptTemplate
# from langchain.chains import LLMChain
# from langchain_openai import ChatOpenAI # 新しいやり方
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# location = "本厚木駅"
# prompt = ChatPromptTemplate.from_messages([
#     ("system", """
#     あなたは地理の専門家です。指定された地域の緯度と経度を
#     latitude:xx.xxx, longitude:xxx.xxx
#     の形式で回答してください。
#     """),
#     ("human", "{location}"),
# ])
# # location = "本厚木駅"
# chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
# r = chain.invoke({"location": location})
# # pos = dict((k, v) for k, v in (item.split(':') for item in r['text'].replace(" ", "").split(',')))
# print(r)