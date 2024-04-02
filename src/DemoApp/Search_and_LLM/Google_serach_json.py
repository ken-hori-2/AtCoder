#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import datetime
import json

from time import sleep
from googleapiclient.discovery import build

from dotenv import load_dotenv
# .envファイルの内容を読み込見込む
load_dotenv()
# os.environを用いて環境変数を表示させます
print(os.environ['Google_Custom_ID'])
google_map_key = os.environ['GoogleMap_API_KEY']
id = os.environ['Google_Custom_ID']

GOOGLE_API_KEY          = google_map_key
CUSTOM_SEARCH_ENGINE_ID = id

DATA_DIR = 'data'

# def makeDir(path):
#     if not os.path.isdir(path):
#         os.mkdir(path)

# def getSearchResponse(keyword):
#     today = datetime.datetime.today().strftime("%Y%m%d")
#     timestamp = datetime.datetime.today().strftime("%Y/%m/%d %H:%M:%S")

#     makeDir(DATA_DIR)

#     service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)

#     page_limit = 1 # 10
#     start_index = 1
#     response = []
#     for n_page in range(0, page_limit):
#         try:
#             sleep(1)
#             response.append(service.cse().list(
#                 q=keyword,
#                 cx=CUSTOM_SEARCH_ENGINE_ID,
#                 lr='lang_ja',
#                 num=10,
#                 start=start_index
#             ).execute())
#             start_index = response[n_page].get("queries").get("nextPage")[0].get("startIndex")
#         except Exception as e:
#             print(e)
#             break

#     # レスポンスをjson形式で保存
#     save_response_dir = os.path.join(DATA_DIR, 'response')
#     makeDir(save_response_dir)
#     out = {'snapshot_ymd': today, 'snapshot_timestamp': timestamp, 'response': []}
#     out['response'] = response
#     jsonstr = json.dumps(out, ensure_ascii=False, indent=4)
#     # with open(os.path.join(save_response_dir, 'response_' + today + '.json'), mode='w', encoding='unicode-escape') as response_file:
#     with open(os.path.join(save_response_dir, 'response_' + today + '.json'), mode='w') as response_file:
#         response_file.write(jsonstr)

# if __name__ == '__main__':

#     target_keyword = 'ダイエット'

#     getSearchResponse(target_keyword)

def makeDir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def getSearchResponse(keyword):
    today = datetime.datetime.today().strftime("%Y%m%d")
    timestamp = datetime.datetime.today().strftime("%Y/%m/%d %H:%M:%S")

    makeDir(DATA_DIR)

    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)

    page_limit = 1 # 0
    start_index = 1
    response = []
    for n_page in range(0, page_limit):
        try:
            sleep(1)
            response.append(service.cse().list(
                q=keyword,
                cx=CUSTOM_SEARCH_ENGINE_ID,
                lr='lang_ja',
                num=10,
                start=start_index
            ).execute())
            start_index = response[n_page].get("queries").get("nextPage")[
                0].get("startIndex")
        except Exception as e:
            print(e)
            break

    # レスポンスをjson形式で保存
    save_response_dir = os.path.join(DATA_DIR, 'response')
    makeDir(save_response_dir)
    out = {'snapshot_ymd': today, 'snapshot_timestamp': timestamp, 'response': []}
    out['response'] = response
    jsonstr = json.dumps(out, ensure_ascii=False, indent=4)
    with open(os.path.join(save_response_dir, 'response_' + today + '.json'), mode='w', encoding='utf-8') as response_file:
        response_file.write(jsonstr)


if __name__ == '__main__':

    target_keyword = '今日の東京の天気'

    getSearchResponse(target_keyword)
