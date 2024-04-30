
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials

lz_uri = 'spotify:artist:64tJ2EAv1R6UaZqc4iOCyj' # https://open.spotify.com/intl-ja/artist/64tJ2EAv1R6UaZqc4iOCyj?si=NXsclRcYTWWnORJYftAmrg

# 2024/04/25
# プレイリストの曲追加
import os
from dotenv import load_dotenv
# .envファイルの内容を読み込見込む
load_dotenv()
# os.environを用いて環境変数を表示させます

username = os.environ['UserName']
scope = 'user-read-playback-state,playlist-read-private,user-modify-playback-state,playlist-modify-public'


client_id = os.environ['Client_ID'] # ここに自分の client ID'
client_secret = os.environ['Client_Secret'] # ここに自分の client seret'

ccm = SpotifyClientCredentials(client_id = client_id, client_secret = client_secret)
spotify = spotipy.Spotify(client_credentials_manager = ccm)
results = spotify.artist_top_tracks(lz_uri)

#認証パート Authentication part

# Redirect URI
redirect_uri = 'https://api.spotify.com/v1/playlists/3cEYpjA9oz9GiPac4AsH4n/tracks'
redirect_uri = 'http://localhost:8888/callback' # たぶん上と同じく、自分のプレイリストを示している？


# Playlist ID
# https://api.spotify.com/v1/playlists/3cEYpjA9oz9GiPac4AsH4n/tracks


# プレイリスト（ログイン）
#認証パート Authentication part

# redirect_uri = 'http://localhost:8888/callback' # 'Redirect URI' 
# #アプリの権限付与に使用する
# scope = 'user-library-read user-read-playback-state playlist-read-private user-read-recently-played playlist-read-collaborative playlist-modify-public playlist-modify-private'
# token = util.prompt_for_user_token(username, scope, my_id, my_secret, redirect_uri)
# spotify = spotipy.Spotify(auth = token)


CreatePlaylist = False
if CreatePlaylist:
    
    # プレイリスト作成（Spotifyアカウントにプレイリストを追加）

    #入力パート Input part
    creat_playlist = 'My_First_Playlist' # 'test_list_XXX'
    #認証パート Authentication part
    # username = 'YOUR USER NAME'
    # my_id ='0000000000000000000000' #client ID
    # my_secret = '0000000000000000000000' #client secret
    # redirect_uri = 'Redirect URI' 
    #アプリの権限付与に使用する
    scope = 'user-library-read user-read-playback-state playlist-read-private user-read-recently-played playlist-read-collaborative playlist-modify-public playlist-modify-private'
    token = util.prompt_for_user_token(username, scope, client_id, client_secret, redirect_uri)
    spotify = spotipy.Spotify(auth = token)
    spotify.user_playlist_create(user = username, name=creat_playlist)






# プレイリストに曲を追加
AddMusicToPlaylist =False # True
if AddMusicToPlaylist:

    #入力パート Input part
    creat_playlist = 'My_First_Playlist'

    # #認証パート Authentication part
    # username = 'YOUR USER NAME'
    # my_id ='0000000000000000000000' #client ID
    # my_secret = '0000000000000000000000' #client secret
    # redirect_uri = 'Redirect URI' 

    #アプリの権限付与に使用する
    scope = 'user-library-read user-read-playback-state playlist-read-private user-read-recently-played playlist-read-collaborative playlist-modify-public playlist-modify-private'

    token = util.prompt_for_user_token(username, scope, client_id, client_secret, redirect_uri)
    spotify = spotipy.Spotify(auth = token)

    play_list = 'https://open.spotify.com/playlist/5OaxfI3WBCSSkPWUmhf4Dp?si=7dfee32111ac4a56'
    # track_URL = 'https://open.spotify.com/track/6NJQpFNkDCMaRBogIi9sOI'
    track_URL = 'https://open.spotify.com/intl-ja/track/1hAloWiinXLPQUJxrJReb1' # ?si=2a6d3be450e7482a' # YOASOBI/アイドル

    print("プレイリスト名: ", creat_playlist)

    # spotify.user_playlist_add_tracks(username, play_list, track_URL)
    spotify.user_playlist_add_tracks(username, play_list, [track_URL])


#入力パート Input part
creat_playlist = 'My_First_Playlist'

#認証パート Authentication part
# username = 'YOUR USER NAME'
# my_id ='0000000000000000000000' #client ID
# my_secret = '0000000000000000000000' #client secret
# redirect_uri = 'Redirect URI' 

#アプリの権限付与に使用する
scope = 'user-library-read user-read-playback-state playlist-read-private user-read-recently-played playlist-read-collaborative playlist-modify-public playlist-modify-private'

token = util.prompt_for_user_token(username, scope, client_id, client_secret, redirect_uri)
spotify = spotipy.Spotify(auth = token)

play_list = 'https://open.spotify.com/playlist/5OaxfI3WBCSSkPWUmhf4Dp?si=7dfee32111ac4a56' # 'https://open.spotify.com/playlist/My_First_Playlist'
# track_URL = 'https://open.spotify.com/track/6NJQpFNkDCMaRBogIi9sOI'
# track_URL ='https://open.spotify.com/track/3ciqhcLmXP4hVGBD98QlEj' # きらり
track_URL = 'https://open.spotify.com/track/1hAloWiinXLPQUJxrJReb1' # アイドル

spotify.user_playlist_add_tracks(username, play_list, [track_URL])



# import spotipy
# import spotipy.util as util
# import time

# #入力パート Input part
# addition_playlist = 'TRACK_ADDITIONAL_PLAYLIST_URL_' #track adding playlist
# original_play_list = 'SEARCH PLAYLIST_URL' #search playlist
# set_tempo = 120 #center tempo
# set_tempo_range = 5 # +- tempo range

# #認証パート Authentication part
# # username = 'YOUR USER NAME'
# # my_id ='0000000000000000000000' #client ID
# # my_secret = '0000000000000000000000' #client secret
# # redirect_uri = 'Redirect URI' 

# #アプリの権限付与に使用する
# #https://developer.spotify.com/documentation/general/guides/scopes/
# scope = 'user-library-read user-read-playback-state playlist-read-private user-read-recently-played playlist-read-collaborative playlist-modify-public playlist-modify-private'

# token = util.prompt_for_user_token(username, scope, my_id, my_secret, redirect_uri)
# spotify = spotipy.Spotify(auth = token)


# def set_tempo_track(original_play_list, set_tempo, set_tempo_range):
#     list_data = spotify.playlist_tracks(original_play_list)
#     track_num = list_data['total']
#     if track_num > 100:
#         track_num = 100
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
#         tempo = track_feature['tempo']
#         if (set_tempo-set_tempo_range) <= tempo <= (set_tempo + set_tempo_range):
#             tempo_urls_list.append(track_url)
#         else:
#             pass
#     return(tempo_urls_list)

# tempo_urls_list = set_tempo_track(original_play_list, set_tempo, set_tempo_range)
# time.sleep(1)
# spotify.user_playlist_add_tracks(username, addition_playlist, tempo_urls_list)
# print('finish')