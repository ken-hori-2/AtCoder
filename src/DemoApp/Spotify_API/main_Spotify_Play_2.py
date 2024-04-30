import spotipy
import spotipy.util as util
import sys
import random
import subprocess
import requests
import json
from spotipy.oauth2 import SpotifyClientCredentials
import pprint

import os
from dotenv import load_dotenv
# .envファイルの内容を読み込見込む
load_dotenv()
# os.environを用いて環境変数を表示させます


# 2024/04/25
# プレイリスト再生、一時停止用のコード






username = os.environ['UserName']
scope = 'user-read-playback-state,playlist-read-private,user-modify-playback-state,playlist-modify-public'


client_id = os.environ['Client_ID'] # ここに自分の client ID'
client_secret = os.environ['Client_Secret'] # ここに自分の client seret'

redirect_uri = 'http://localhost:8888/callback'

search_str = 'YOASOBI' # sys.argv[1] # 実行時の引数でキーワード入力

artist_id_map={}
token = util.prompt_for_user_token(username, scope,     client_id, client_secret, redirect_uri)

sp = spotipy.Spotify(auth=token)
header = {'Authorization': 'Bearer {}'.format(token)}


res = requests.get("https://api.spotify.com/v1/me/player/devices", headers=header)
devices = res.json()
print("device:") # , devices)
pprint.pprint(devices)
device_id = devices["devices"][0]['id']

"プレイリスト生成(キーワードから関連のアーティストでプレイリスト生成)" # J-POP
# playlist = sp.user_playlist_create(username,"NewPlaylist2")
# playlist_id = playlist['id']

# result = sp.search(q='artist:'+search_str, limit=1)
# artist_id = result['tracks']['items'][0]['artists'][0]['id']
# print (result['tracks']['items'][0]['id'])
# artist_related_artists = sp.artist_related_artists(artist_id)
# track_ids = []
# for artist_list in artist_related_artists['artists']:
#     result = sp.search(q='artist:'+artist_list['name'], limit=50)

#     if len(result['tracks']['items']) > 1:
#         track_ids.append(random.choice(result['tracks']['items'])['id'])

# sp.user_playlist_add_tracks(username, playlist_id, track_ids)

# print("playlist id : ", playlist_id)

# 既存のプレイリスト
# add
# playlist_id = 'https://open.spotify.com/playlist/5CzbTLKqtxlXnUOkgd26qx' # ?si=8e1c430b92a147df'

"NewPlaylist"
playlist_id = "0Ud98MOBx8vgwmESw9rCqZ" # 'spotify:artist:64tJ2EAv1R6UaZqc4iOCyj' # "https://open.spotify.com/playlist/0Ud98MOBx8vgwmESw9rCqZ" # ?si=dc3eafc77b90491d"
"House Music"
playlist_id = "5CzbTLKqtxlXnUOkgd26qx"

param = {'device_id':device_id,
         'context_uri':'spotify:playlist:%s' % playlist_id}

"再生"
res = requests.put("https://api.spotify.com/v1/me/player/play", data=json.dumps(param), headers = header)
print(res) # 204なら成功


# # トラック情報の取得
# track = sp.user_playlist_tracks(username, playlist_id)
# sp.start_playback(device_id, playlist_id)
# # print(track)
# pprint.pprint(track)

res = requests.get("https://api.spotify.com/v1/me/playlists?limit=10", headers=header)
playlists = res.json()
print("playlists:") # , devices)
pprint.pprint(playlists['total'])
# pprint.pprint(playlists['id'])

"***** 重要 *****"
# APIのリクエスト回数に上限があるので今はRUNNING=HOUSEMUSIC, WALKING=JPOPにする
# なので、毎回生成や、プレイリストの中身を検索するのは厳しいかも
"マイライブラリ内の全てのプレイリストのIDを取得"
# playlist_urls_list =[]
# playlist_num = playlists['total']
# for i in range(playlist_num):
#     # print(i)
#     playlist_url = playlists['items'][i]['id']
#     playlist_urls_list.append(playlist_url)
# pprint.pprint(playlist_urls_list)
"プレイリストの中を検索"
# # 'href': 'https://api.spotify.com/v1/users/{UserName}/playlists?offset=0&limit=10',
set_tempo = 125 # 結構早め(House Music) # range : 120 ~ 130
set_tempo_range = 5 # +- tempo range
"マイライブラリの全プレイリストで実施"
# ave_bpm_list = []
# # ave_bpm_list_for_select = []
# for i in range(1): # playlist_num):
#     list_data = sp.playlist_tracks(playlist_urls_list[i]) # 全てのプレイリストから選択
#     track_num = list_data['total']
#     ave_bpm = 0
#     bpm_sum = 0
#     # ave_bpm_for_select = 0
#     # bpm_sum_for_select = 0
#     # select_track_count = 0 # BPMの制約を満たした曲の個数
#     if track_num > 100:
#         track_num =100
#     urls_list =[]
#     for i in range(track_num):
#         track_url = list_data['items'][i]['track']['external_urls']['spotify']
#         urls_list.append(track_url)
#     # time.sleep(1) #1sec stop
#     tempo_urls_list =[]
#     for i in range(len(urls_list)):
#         track_url = urls_list[i]
#         track_feature = sp.audio_features(track_url)[0]
#         try:
#             tempo = track_feature['tempo']
#             danceability = track_feature['danceability']
#             # if (set_tempo-set_tempo_range) <= tempo <= (set_tempo + set_tempo_range):
#             #     if danceability > 0.7:
#             #         tempo_urls_list.append(track_url)
#             #         bpm_sum_for_select += tempo # BPMの制約のみで平均を出す場合
#             #         select_track_count += 1
#             #         # print(f"BPM:{tempo}, sum: {bpm_sum}")
#             # else:
#             #     pass
#             "一旦BPMに制約をつけないで平均を加算"
#             bpm_sum += tempo # こうしないとプレイリスト全体のBPMの平均を算出できない

#         except:
#             print("Error!")
#             print("track: ", track_feature)

#     ave_bpm = bpm_sum/track_num # プレイリスト全体の平均を出す場合
#     print("BPM AVERAGE : ", ave_bpm)
#     ave_bpm_list.append(ave_bpm)

# print("BPM AVERAGE LIST : ", ave_bpm_list) # プレイリスト全体の平均を出す場合
# print("MAX AVE BPM : ", max(ave_bpm_list))

# print("このプレイリスト内を検索")
# pprint.pprint(tempo_urls_list)

"一時停止"
# res = requests.put("https://api.spotify.com/v1/me/player/pause", data=json.dumps(param), headers = header)
# print(res) # 204なら成功