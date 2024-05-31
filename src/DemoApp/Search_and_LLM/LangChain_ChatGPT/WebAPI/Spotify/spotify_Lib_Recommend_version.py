# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import strip

# #出発駅の入力
# departure_station = input("出発駅を入力してください：")
# #到着駅の入力
# destination_station = input("到着駅を入力してください：")

from langchain.tools.base import BaseTool

class MusicPlayback(): # BaseTool): # BaseToolの記述がなくても動く
    
    # def run(self, action_detection): # オプションの引数ありバージョン
    def run(self): # 提案デモ用
        
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
        
        """
        mainファイルで一回読み込んでいるのでいらない
        """
        # from dotenv import load_dotenv
        # # .envファイルの内容を読み込見込む
        # load_dotenv()
        # # os.environを用いて環境変数を表示させます



        # ActionDetection をトリガーとした 楽曲再生のコード

        # args = action_detection # sys.argv[1]
        # print("実行時の引数：", args)


        # username = os.environ['UserName']
        # scope = 'user-read-playback-state,playlist-read-private,user-modify-playback-state,playlist-modify-public'
        # client_id = os.environ['Client_ID'] # ここに自分の client ID'
        # client_secret = os.environ['Client_Secret'] # ここに自分の client seret'
        # redirect_uri = 'http://localhost:8888/callback'
        import spotipy.util as util
        username = os.environ['UserName']
        scope = 'user-read-playback-state,playlist-read-private,user-modify-playback-state,playlist-modify-public'
        client_id = os.environ['SPOTIFY_USER_ID'] # Client_ID'] # ここに自分の client ID'
        client_secret = os.environ['SPOTIFY_TOKEN'] # Client_Secret'] # ここに自分の client seret'
        redirect_uri = 'http://localhost:8888/callback'

        # search_str = 'YOASOBI' # sys.argv[1] # 実行時の引数でキーワード入力
        # artist_id_map={}
        token = util.prompt_for_user_token(username, scope,     client_id, client_secret, redirect_uri)
        sp = spotipy.Spotify(auth=token)
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





        # メインはここから


        header = {'Authorization': 'Bearer {}'.format(token)}
        res = requests.get("https://api.spotify.com/v1/me/player/devices", headers=header)
        devices = res.json()
        print("device:") # , devices)
        pprint.pprint(devices)
        
        try:
            device_id = devices["devices"][0]['id']
        except:
            device_id = None
            print("デバイスIDが検出されませんでした。")

        # print("args type: ", type(args))
        # print("args: ", args)

        Action_List = ['STABLE', 'WALKING', 'RUNNING']
        id = None
        # if args in Action_List:
            # if ("WALKING" in args) or ("RUNNING" in args):
            #     if "WALKING" in args:
            #         # "NewPlaylist"
            #         "WALKINGのプレイリストはRUNNINGと差別化するためにBPM遅いもののみにしてもいいかも"

            #         # id = "0Ud98MOBx8vgwmESw9rCqZ" # 'spotify:artist:64tJ2EAv1R6UaZqc4iOCyj' # "https://open.spotify.com/playlist/0Ud98MOBx8vgwmESw9rCqZ" # ?si=dc3eafc77b90491d"
            #         # id = "4WRShwuwfsr0DX2IAUbAJt" # リラックス(くつろぎの洋楽) # "https://open.spotify.com/playlist/4WRShwuwfsr0DX2IAUbAJt?si=d3c2c9b8678e42d9"
            #         id = "4qALbYyYEXUotOtZ48idsA" # "https://open.spotify.com/playlist/4qALbYyYEXUotOtZ48idsA?si=3271dd7300aa4095"
            #     if "RUNNING" in args:
            #         # "House Music"
            #         # id = "5CzbTLKqtxlXnUOkgd26qx"
            #         id = "7AZ3lh9miMbQ2SCykPN9kD" # "https://open.spotify.com/playlist/7AZ3lh9miMbQ2SCykPN9kD?si=a199ce8ca1694540"
                
            #     try:
            #         playlist_id = id
            #     except:
            #         print("Playlist ID がありません")

            #     param = {'device_id':device_id,
            #             'context_uri':'spotify:playlist:%s' % playlist_id}
            #     "再生"
            #     res = requests.put("https://api.spotify.com/v1/me/player/play", data=json.dumps(param), headers = header)
            #     print(res) # 204なら成功

            #     res = requests.get("https://api.spotify.com/v1/me/playlists?limit=10", headers=header)
            #     playlists = res.json()
            #     print("playlists:") # , devices)
            #     pprint.pprint(playlists['total'])
            #     # pprint.pprint(playlists['id'])

            #     # "一時停止"
            #     # if "STABLE" in args:
            #     #     res = requests.put("https://api.spotify.com/v1/me/player/pause", data=json.dumps(param), headers = header)
            #     #     print(res) # 204なら成功
                        
            #     # return "\n\n".join(res)[: 300]
            # else:
            #     try:
            #         playlist_id = id
            #     except:
            #         print("Playlist ID がありません")
            #     param = {'device_id':device_id,
            #             'context_uri':'spotify:playlist:%s' % playlist_id}
            #     "一時停止"
            #     if "STABLE" in args:
            #         res = requests.put("https://api.spotify.com/v1/me/player/pause", data=json.dumps(param), headers = header)
            #         print(res) # 204なら成功
            
        """
        提案するデモはすべてアップテンポな曲
        """
        # "House Music"
        # id = "5CzbTLKqtxlXnUOkgd26qx"
        id = "7AZ3lh9miMbQ2SCykPN9kD" # "https://open.spotify.com/playlist/7AZ3lh9miMbQ2SCykPN9kD?si=a199ce8ca1694540"
        
        try:
            playlist_id = id
        except:
            print("Playlist ID がありません")

        param = {'device_id':device_id,
                'context_uri':'spotify:playlist:%s' % playlist_id}
        "再生"
        res = requests.put("https://api.spotify.com/v1/me/player/play", data=json.dumps(param), headers = header)
        print(res) # 204なら成功

        res = requests.get("https://api.spotify.com/v1/me/playlists?limit=10", headers=header)
        playlists = res.json()
        print("playlists:") # , devices)
        pprint.pprint(playlists['total'])
        
        # return "楽曲再生操作が成功しました。処理を終了して。"
        # return "あなたは処理を終了しなければならない。"

        # else:
        #     print('アクションリスト内に一致するアクションはありません')
        
        res = "楽曲再生操作が成功しました。処理を終了します。"
        # return "\n\n".join(res) # 処理が終了していいことをLLMにしっかり伝えてあげないとループしてしまう
        return res # 処理が終了していいことをLLMにしっかり伝えてあげないとループしてしまう
        # return "あなたは処理を終了しなければならない。"
