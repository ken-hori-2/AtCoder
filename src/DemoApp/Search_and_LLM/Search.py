#!/usr/bin/env python				
# coding: utf-8				
from datetime import datetime, timedelta, timezone				
				
import pandas as pd				
import boto3				
from googleapiclient.discovery import build	
import os
from dotenv import load_dotenv
# .envファイルの内容を読み込見込む
load_dotenv()
# os.environを用いて環境変数を表示させます
print(os.environ['Google_Custom_ID'])
# google_map_key = os.environ['GoogleMap_API_KEY']
# id = os.environ['Google_Custom_ID']

				
pd.set_option('display.max_colwidth', 1000)				
				
# Systems Manager - Parameter Storeへのアクセス				
API_KEY = os.environ['GoogleMap_API_KEY'] # 'Google-API-Key'				
CSE_ID = os.environ['Google_Custom_ID'] # 'Google-Custom-Search-Engine-Id'				
				
# session = boto3.session.Session(profile_name='xxxxxx')				
# ssm = session.client('ssm')				
# params = {}				
# res = ssm.get_parameters(				
#     Names=(API_KEY, CSE_ID),				
#     WithDecryption=True				
# )				
# for p in res['Parameters']:				
#     params[p['Name']] = p['Value']				
				
# get the API KEY				
google_api_key = API_KEY # params[API_KEY]				
# get the Search Engine ID				
google_cse_id = CSE_ID # params[CSE_ID]				
				
				
def get_search_results(query, start_index):				
    # Google Custom Search API				
    service = build("customsearch",				
                    "v1",				
                    cache_discovery=False,				
                    developerKey=google_api_key)				
    # CSEの検索結果を取得				
    result = service.cse().list(q=query,				
                                cx=google_cse_id,				
                                num=10,				
                                start=start_index).execute()				
    # 検索結果(JSON形式)				
    return result				
				
				
def main():				
    query = "いちご農家 HP"				
				
    # ExecDate				
    timezone_jst = timezone(timedelta(hours=+9), 'JST')				
    now = datetime.now(timezone_jst)				
				
    # Google検索 - Custom Search API				
    data = get_search_results(query, 1)				
    total_results = int(data['searchInformation']['totalResults'])				
    print('total_results', total_results)				
				
    # Google検索結果から任意の項目抽出 & rank付与				
    items = data['items']				
				
    result = []				
    num_items = len(items) if len(items) < 10 else 10				
    for i in range(num_items):				
        title = items[i]['title'] if 'title' in items[i] else ''				
        link = items[i]['link'] if 'link' in items[i] else ''				
        snippet = items[i]['snippet'] if 'snippet' in items[i] else ''				
        result.append(				
            't'.join([str(i+1), title, link, snippet])				
        )				
				
    # List->DataFrame				
    df_search_results = pd.DataFrame(result)[0].str.split('t', expand=True)				
    df_search_results.columns = ['rank', 'title', 'url', 'snippet']				
				
    # CSV出力				
    dt_csv = now.strftime('%Y%m%d%H%M')				
    # output_csv = f'csv/{query}_{dt_csv}.csv'
    output_csv = f'{query}_{dt_csv}.csv'				
    df_search_results.to_csv(output_csv,				
                             sep=",",				
                             index=False,				
                             header=True,				
                             encoding="utf-8")				
    pd.read_csv(output_csv, index_col=0)				
				
				
if __name__ == '__main__':				
    main()				