# -*- coding: utf-8 -*-
import googlemaps
import pprint # list型やdict型を見やすくprintするライブラリ
# key = 'Your API' # 上記で作成したAPIキーを入れる
# 必要モジュールのインポート
import os


"""
mainファイルで一回読み込んでいるのでいらない
"""
# from dotenv import load_dotenv
# # .envファイルの内容を読み込見込む
# load_dotenv()

# os.environを用いて環境変数を表示させます
print(os.environ['GOOGLE_MAPS_API_KEY'])
key = os.environ['GOOGLE_MAPS_API_KEY']

class Localization(): # BaseTool): # BaseToolの記述がなくても動く
    
    def run(self, location): # オプションの引数ありバージョン

        client = googlemaps.Client(key) #インスタンス生成

        # geocode_result = client.geocode('東京都大森駅') # 位置情報を検索
        # geocode_result = client.geocode('東京都後楽園駅') # 位置情報を検索
        geocode_result = client.geocode(location) # '本厚木駅') # 位置情報を検索
        # print(geocode_result)
        loc = geocode_result[0]['geometry']['location'] # 軽度・緯度の情報のみ取り出す

        # # place_result = client.places_nearby(location=loc, radius=200, type='food') #半径200m以内のレストランの情報を取得
        # place_result = client.places_nearby(location=loc, radius=100, type='food') #半径100m以内のレストランの情報を取得
        # pprint.pprint(place_result)

        print("loc : ", loc)
        # return "\n\n".join(loc)[: 300]
        # return "\n\n緯度と経度は次の通りです。\n".join(loc)[: 300]
        return f"緯度と経度は'{loc}'です。"