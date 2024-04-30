import spotipy
import spotipy.util as util

#入力パート Input part
creat_playlist = '2nd_List'




# 2024/04/25
# BPMが一定のプレイリストを生成するコード






import os
from dotenv import load_dotenv
# .envファイルの内容を読み込見込む
load_dotenv()
username = os.environ['UserName']
client_id = os.environ['Client_ID'] # ここに自分の client ID'
client_secret = os.environ['Client_Secret'] # ここに自分の client seret'


# Redirect URI
redirect_uri = 'https://api.spotify.com/v1/playlists/3cEYpjA9oz9GiPac4AsH4n/tracks'
redirect_uri = 'http://localhost:8888/callback' # たぶん上と同じく、自分のプレイリストを示している？

#アプリの権限付与に使用する
scope = 'user-library-read user-read-playback-state playlist-read-private user-read-recently-played playlist-read-collaborative playlist-modify-public playlist-modify-private'

token = util.prompt_for_user_token(username, scope, client_id, client_secret, redirect_uri)
spotify = spotipy.Spotify(auth = token)

# def creat_play_list(list_name):
#     spotify.user_playlist_create(username, name = list_name)
#     list_data = spotify.user_playlists(user = username)
#     for i in range(list_data['total']):
#         play_list_name = list_data['items'][i]['name']
#         if play_list_name == list_name:
#             url = list_data['items'][i]['external_urls']['spotify']
#         else:
#             pass
#     return(url)

# # play_list = creat_play_list(creat_playlist)
# # print(play_list)


# import spotipy
# import spotipy.util as util
# import time

# #入力パート Input part
# original_play_list = 'https://open.spotify.com/playlist/37i9dQZEVXbMDoHDwVN2tF' # グローバルチャート トップ50
# set_tempo = 120 #center tempo
# set_tempo_range = 5 # +- tempo range

# #認証パート Authentication part
# # ～～～～～略～～～～～
# # spotify = spotipy.Spotify(auth = token)

# def set_tempo_track(original_play_list, set_tempo, set_tempo_range):
#     list_data = spotify.playlist_tracks(original_play_list)
#     track_num = list_data['total']
#     if track_num > 100:
#         track_num =100
#     urls_list =[]
#     for i in range(track_num):
#         track_url = list_data['items'][i]['track']['external_urls']['spotify']
#         urls_list.append(track_url)
#     time.sleep(1) #1sec stop
#     tempo_urls_list =[]
#     for i in range(len(urls_list)):
#         track_url = urls_list[i]
#         track_feature = spotify.audio_features(track_url)[0]
#         time.sleep(1)

#         # tempoがおかしいものがあるのではなく、トラックデータがNoneのものがあるので取得しようとするとエラーになる
        
#         try:
#             # print(track_feature['danceability'])
#             # print(track_feature['tempo'])
#             tempo = track_feature['tempo'] # tempoがおかしいものがある？
#             # print(type(tempo))
#             if (set_tempo-set_tempo_range) <= tempo <= (set_tempo + set_tempo_range):
#                 tempo_urls_list.append(track_url)
#                 # break
#             else:
#                 pass
#         except:
#             print("Error!")
#             # print(track_feature['tempo'])
#             # tempo = track_feature['tempo'] # tempoがおかしいものがある？
#             # print(type(tempo))
#         # print(track_feature)
        
#     # print("result:", tempo_urls_list)
#     return(tempo_urls_list)

# # tempo_urls_list = set_tempo_track(original_play_list, set_tempo, set_tempo_range)
# # print(tempo_urls_list)
# # print('finish')



# 7. BPMが一定の曲プレイリストを作成
import spotipy
import spotipy.util as util
import time

#入力パート Input part
# creat_playlist = 'test_list_tempo120_3'
creat_playlist = 'tempo120-130_danceability>70'
# original_play_list = 'https://open.spotify.com/playlist/37i9dQZEVXbMDoHDwVN2tF' # グローバルチャート トップ50
original_play_list = "https://open.spotify.com/playlist/33McNWEGXtoP98qjqF5uVQ?si=0c91d03b1d364327" # チルハウス
# set_tempo = 120 #center tempo
set_tempo = 125 # 結構早め(House Music) # range : 120 ~ 130

set_tempo_range = 5 # +- tempo range

#認証パート Authentication part
# username = 'YOUR USER NAME'
# my_id ='0000000000000000000000' #client ID
# my_secret = '0000000000000000000000' #client secret
# redirect_uri = 'Redirect URI' 

#アプリの権限付与に使用する
scope = 'user-library-read user-read-playback-state playlist-read-private user-read-recently-played playlist-read-collaborative playlist-modify-public playlist-modify-private'

token = util.prompt_for_user_token(username, scope, client_id, client_secret, redirect_uri)
spotify = spotipy.Spotify(auth = token)


def creat_play_list(list_name):
    spotify.user_playlist_create(username, name = list_name)
    list_data = spotify.user_playlists(user = username)
    for i in range(list_data['total']):
        play_list_name = list_data['items'][i]['name']
        if play_list_name == list_name:
            url = list_data['items'][i]['external_urls']['spotify']
        else:
            pass
    return(url)


def set_tempo_track(original_play_list, set_tempo, set_tempo_range):
    list_data = spotify.playlist_tracks(original_play_list)
    track_num = list_data['total']
    if track_num > 100:
        track_num =100
    urls_list =[]
    for i in range(track_num):
        track_url = list_data['items'][i]['track']['external_urls']['spotify']
        urls_list.append(track_url)
    # time.sleep(1) #1sec stop
    tempo_urls_list =[]
    for i in range(len(urls_list)):
        track_url = urls_list[i]
        track_feature = spotify.audio_features(track_url)[0]
        # time.sleep(1)
        # tempoがおかしいものがあるのではなく、トラックデータがNoneのものがあるので取得しようとするとエラーになる
        
        # try:
        #     tempo = track_feature['tempo']
        #     if (set_tempo-set_tempo_range) <= tempo <= (set_tempo + set_tempo_range):
        #         tempo_urls_list.append(track_url)
        #     else:
        #         pass
        # except:
        #     print("Error!")
        #     print("track: ", track_feature)
        
        # tempo + daceability # 踊るときや走るときなど、体を動かすときにより最適？
        try:
            tempo = track_feature['tempo']
            danceability = track_feature['danceability']
            if (set_tempo-set_tempo_range) <= tempo <= (set_tempo + set_tempo_range):
                if danceability > 0.7:
                    tempo_urls_list.append(track_url)
            else:
                pass
        except:
            print("Error!")
            print("track: ", track_feature)

    return(tempo_urls_list)

play_list = creat_play_list(creat_playlist)
tempo_urls_list = set_tempo_track(original_play_list, set_tempo, set_tempo_range)
# time.sleep(1)
spotify.user_playlist_add_tracks(username, play_list, tempo_urls_list)

# 既存のプレイリストに追加
# spotify.user_playlist_add_tracks(username, addition_playlist, tempo_urls_list)

print('finish')