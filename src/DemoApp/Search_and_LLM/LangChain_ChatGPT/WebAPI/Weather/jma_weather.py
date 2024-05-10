# -*- coding:utf-8 -*-
import requests
import json

# # 気象庁データの取得
# jma_url = "https://www.jma.go.jp/bosai/forecast/data/forecast/130000.json"
# jma_json = requests.get(jma_url).json()

# # 取得したいデータを選ぶ
# jma_date = jma_json[0]["timeSeries"][0]["timeDefines"][0]
# jma_weather = jma_json[0]["timeSeries"][0]["areas"][0]["weathers"][0]
# jma_rainfall = jma_json["Feature"][0]["Property"]["WeatherList"]["Weather"][0]["Rainfall"]
# # 全角スペースの削除
# jma_weather = jma_weather.replace('　', '')

# print(jma_date)
# print(jma_weather)
# print(jma_rainfall)

# -*- coding:utf-8 -*-
import requests
from datetime import datetime

def get_latest():
    url = "https://www.jma.go.jp/bosai/amedas/data/latest_time.txt"
    with requests.get(url) as response:
        return 
    
# 気象庁から天気予報情報(JSONデータ)を取得
# ファイルパスの「270000」はエリアコード。取得したい地域に応じて適宜変更します。
JSON_URL = "https://www.jma.go.jp/bosai/forecast/data/forecast/130000.json"
jma_json = requests.get(JSON_URL).json()

# 今日の天気
jma_date = jma_json[0]["timeSeries"][0]["timeDefines"][0]
jma_date = datetime.fromisoformat(jma_date).strftime("%Y/%m/%d %H:%M" + "発表")
jma_area = jma_json[0]["timeSeries"][0]["areas"][0]["area"]["name"]
jma_wind = jma_json[0]["timeSeries"][0]["areas"][0]["winds"][0]
jma_wave = jma_json[0]["timeSeries"][0]["areas"][0]["waves"][0]
jma_weather = jma_json[0]["timeSeries"][0]["areas"][0]["weathers"][0]


print("■" + jma_area + "の天気予報" + "(" + jma_date + ")")
print("天気:" + jma_weather)
print("風速:" + jma_wave)
print("風向:" + jma_wind)
