# -*- coding: utf-8 -*-
import requests
import json

import os
from dotenv import load_dotenv
# .envファイルの内容を読み込見込む
load_dotenv()
# from tool_directory import ToolLoader

# HotpepperAPIのAPI KEY
HotPepper_Api_Key = os.environ['HOTPEPPER_API_KEY']

class RestaurantSearch(): # BaseTool): # BaseToolの記述がなくても動く
    
    def run(self, latitude, longitude, keyword): # オプションの引数ありバージョン

        # # GeoJSにリクエストしIPアドレスから現在地の緯度・経度を取得
        # geo_request_url = 'https://get.geojs.io/v1/ip/geo.json'
        # data = requests.get(geo_request_url).json()
        # print("現在位置: ")
        # print(data['latitude'])
        # print(data['longitude'])

        # print(data['country'])
        # # print(data['region'])
        # # print(data['city'])

        # 検索クエリ
        query = {
                'key': HotPepper_Api_Key, # xxxxxxxxxxxxxxxxxx', # APIキー
                'lat': latitude, #  data['latitude'], # 現在地の緯度
                'lng': longitude, # data['longitude'], # 現在地の経度
                'keyword': keyword, # 'ラーメン', # キーワードに「ラーメン」
                'range': '4', # 2000m以内
                'count': 10, # 50, # 取得データ数
                'format': 'json' # データ形式json
                }

        # グルメサーチAPIのリクエストURL        
        url = 'http://webservice.recruit.co.jp/hotpepper/gourmet/v1/'

        # URLとクエリでリクエスト
        responce = requests.get(url, query)

        # 戻り値をjson形式で読み出し、['results']['shop']を抽出
        result = json.loads(responce.text)['results']['shop']

        res = []
        # 店名、住所を表示
        for i in result:
            # print(i['name']+' : '+i['address'])
            res.append(i['name']+' : '+i['address'])
        # return "お店は次の通りです。\n" + res
        return res
        # return "\n\n".join(result)[: 300]
