# import requests
# geo_request_url = 'https://get.geojs.io/v1/ip/geo.json'
# geo_data = requests.get(geo_request_url).json()
# print("現在位置: ")
# print(geo_data['latitude'])
# print(geo_data['longitude'])

# print(geo_data['country'])
# # print(geo_data['region'])
# # print(geo_data['city'])

# # from geopy.distance import geodesic
# # import time

# # # 検知する座標範囲内
# # convenience_store = (store_latitude, store_longitude) 
# # radius_threshold = 0.03  # 30m以内に入った場合に検知

# # # 位置情報を継続的に取得するループ
# # while True:
# #     try:
# #         # iPhoneの位置情報を取得
# #         location = api.devices[0].location()
# #         current_coordinate = (location['latitude'], location['longitude'])
# #         print("座標", current_coordinate)
        
# #         # 現在位置とコンビニ位置の距離を計算
# #         distance = geodesic(current_coordinate, convenience_store).kilometers
# #         print("距離", distance)
        
# #         # 距離が設定した範囲内に入った場合、メール送信
# #         if distance <= radius_threshold:
# #             print("メール送信へ移行")
# #             break
        
# #     except Exception as e:
# #         print("エラー:", e)
    
# #     # 位置情報の取得間隔を設定するために適切な待ち時間を設定
# #     # 例えば、30秒ごとに位置情報を取得する場合
# #     time.sleep(30)


# # from geopy.geocoders import Nominatim

# # loc = Nominatim(user_agent="test1")

# # getlocation = loc.geocode("大阪城")

# # print("住所: ",getlocation.address,"\n")
# # print("緯度: ",getlocation.latitude,"\n")
# # print("経度: ",getlocation.longitude,"\n")
# # print("詳細な情報: ",getlocation.raw)

import requests
import json

import os
from dotenv import load_dotenv
# .envファイルの内容を読み込見込む
load_dotenv()
# from tool_directory import ToolLoader

# HotpepperAPIのAPI KEY
HotPepper_Api_Key = os.environ['HOTPEPPER_API_KEY']

# GeoJSにリクエストしIPアドレスから現在地の緯度・経度を取得
geo_request_url = 'https://get.geojs.io/v1/ip/geo.json'
data = requests.get(geo_request_url).json()
print("現在位置: ")
print(data['latitude'])
print(data['longitude'])

print(data['country'])
# print(data['region'])
# print(data['city'])

# 検索クエリ
query = {
        'key': HotPepper_Api_Key, # xxxxxxxxxxxxxxxxxx', # APIキー
        'lat': data['latitude'], # 現在地の緯度
        'lng': data['longitude'], # 現在地の経度
        'keyword': 'ラーメン', # キーワードに「ラーメン」
        'range': '4', # 2000m以内
        'count': 50, # 取得データ数
        'format': 'json' # データ形式json
        }

# グルメサーチAPIのリクエストURL        
url = 'http://webservice.recruit.co.jp/hotpepper/gourmet/v1/'

# URLとクエリでリクエスト
responce = requests.get(url, query)

# 戻り値をjson形式で読み出し、['results']['shop']を抽出
result = json.loads(responce.text)['results']['shop']

# 店名、住所を表示
for i in result:
    print(i['name']+' : '+i['address'])