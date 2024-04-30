import requests
import base64
import json

import os
from dotenv import load_dotenv
# .envファイルの内容を読み込見込む
load_dotenv()
# os.environを用いて環境変数を表示させます
client_id = os.environ['Client_ID'] # ここに自分の client ID'
client_secret = os.environ['Client_Secret'] # ここに自分の client seret'

headers = {'Authorization': 'Basic ' + (base64.b64encode((client_id + ':' + client_secret).encode())).decode()}
data    = {'grant_type': 'client_credentials'}
resp0 = requests.post('https://accounts.spotify.com/api/token', data=data, headers=headers)
resp = json.loads(resp0.content.decode())
token = resp0['access_token']

# search API を叩く
headers = {'Authorization': 'Bearer ' + token}
q = 'q=ドラえもん&type=track'    # 検索クエリ文字列
resp0 = requests.get('https://api.spotify.com/v1/search?' + q, headers=headers)
resp = json.loads(resp0.content.decode())
# 結果を アルバム名: トラック名 形式で出力
for i in resp['tracks']['items']:
    print(i['album']['name'], ':', i['name'])