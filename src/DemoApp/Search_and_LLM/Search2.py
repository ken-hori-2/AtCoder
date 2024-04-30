import urllib
import urllib.request
import json
import sys

url_list = []
title_list = []
snippet_list =[]

API_KEY = "APIのKEYを入れます"
ENGINE_ID="エンジンIDを入れます"
phrase = "検索ワードを入れます"
cnt = 1 # 1:1~10位, 11:11~20位

#リクエストURL
req_url = "https://www.googleapis.com/customsearch/v1?hl=ja&key="+API_KEY+"&cx="+ENGINE_ID+"&alt=json&q="+ phrase +"&start="+ str(cnt)

# #念の為User Agentで
# headers = {"User-Agent": 'Mozilla /5.0 (iPhone; CPU iPhone OS 9_1 like Mac OS X) AppleWebKit/601.1.46 (KHTML, like Gecko) Version/9.0 Mobi    le/13B5110e Safari/601.1'}

#リクエスト
req = urllib.request.Request(req_url)
res = urllib.request.urlopen(req)
dump = json.loads(res.read())
hit = dump["queries"]["request"][0]["totalResults"]
#print(hit)


#検索結果のURL, TITLE, SNIPPET　をappendしてく
for p in range(len(dump["items"])):
    url_list.append(dump['items'][p]['link'])
    print(dump['items'][p]['link'])
    title_list.append(dump['items'][p]['title'])
    print(dump['items'][p]['title'])
    snippet_list.append(dump['items'][p]['snippet'].replace('\n',''))
    print(dump['items'][p]['snippet'])
    print('---------------------------------')
