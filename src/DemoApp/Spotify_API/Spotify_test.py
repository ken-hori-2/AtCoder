import spotipy
from spotipy.oauth2 import SpotifyClientCredentials



# 2024/04/25
# アーティストの曲検索
# 関連アーティスト検索

# YOASOBIの曲を検索

# lz_uri = 'spotify:artist:36QJpDe2go2KgaRleHCDTp'
lz_uri = 'spotify:artist:64tJ2EAv1R6UaZqc4iOCyj' # https://open.spotify.com/intl-ja/artist/64tJ2EAv1R6UaZqc4iOCyj?si=NXsclRcYTWWnORJYftAmrg

lz_uri = "spotify:playlist:0Ud98MOBx8vgwmESw9rCqZ"


import os
from dotenv import load_dotenv
# .envファイルの内容を読み込見込む
load_dotenv()
# os.environを用いて環境変数を表示させます
client_id = os.environ['Client_ID'] # ここに自分の client ID'
client_secret = os.environ['Client_Secret'] # ここに自分の client seret'

ccm = SpotifyClientCredentials(client_id = client_id, client_secret = client_secret)
spotify = spotipy.Spotify(client_credentials_manager = ccm)
results = spotify.artist_top_tracks(lz_uri)

for track in results['tracks'][:10]:
    print('track    : ' + track['name'])
    print('audio    : ' + track['preview_url'])
    print('cover art: ' + track['album']['images'][0]['url'])
    print()





"*****"
# 関連アーティストを検索

# # # import spotipy
# # # from spotipy.oauth2 import SpotifyClientCredentials
# # #入力パート
# # # artist_url = 'https://open.spotify.com/artist/36QJpDe2go2KgaRleHCDTp'
# # artist_url = lz_uri # ' https://open.spotify.com/intl-ja/artist/64tJ2EAv1R6UaZqc4iOCyj'
# # album_url =''
# # track_url = ''
# # #認証パート
# # # my_id ='0000000000000000000000' #client ID
# # # my_secret = '0000000000000000000000' #client secret
# # ccm = SpotifyClientCredentials(client_id = my_id, client_secret = my_secret)
# # spotify = spotipy.Spotify(client_credentials_manager = ccm)
# # results = spotify.artist_related_artists(artist_url)
# # print(results)


# import spotipy
# from spotipy.oauth2 import SpotifyClientCredentials
# import pandas as pd
# import csv

# #入力パート
# artist_url = lz_uri # 'https://open.spotify.com/artist/36QJpDe2go2KgaRleHCDTp'
# album_url =''
# track_url = ''
# output_filename = 'YOASOBI.csv' # 'zep_related_artist.csv' #.csv形式で名前を入力

# #認証パート
# # my_id ='0000000000000000000000' #client ID
# # my_secret = '0000000000000000000000' #client secret
# ccm = SpotifyClientCredentials(client_id = my_id, client_secret = my_secret)
# spotify = spotipy.Spotify(client_credentials_manager = ccm)

# results = spotify.artist_related_artists(artist_url)
# result = results['artists']
# related_df = pd.DataFrame(index=[], columns=['Name', 'Genres', 'Images_url', 'Popularity', 'URL', 'URI'])
# for i in range(len(result)): #resuktの数をカウントしてfor文を回す
#     related_df= related_df._append({
#         'Name' : result[i]['name'],
#         'Genres' : result[i]['genres'],
#         'Images_url' : result[i]['images'][0]['url'],
#         'Popularity' : result[i]['popularity'],
#         'URL' : result[i]['external_urls']['spotify'], 
#         'URI' : result[i]['uri']}, ignore_index=True)
# #print(related_df)
# related_df.to_csv(output_filename, encoding='utf-8') #csvファイル出力
# with open(output_filename, 'a', newline='') as f:
#     writer = csv.writer(f)


"*****"



import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

#入力パート
artist_url = ''
album_url =''
track_url = 'https://open.spotify.com/track/3ciqhcLmXP4hVGBD98QlEj' # 藤井風「きらり」
track_url = 'https://open.spotify.com/track/2ehGWuN4ONdS8Q5mTW8rC8' # house

#認証パート
# my_id ='0000000000000000000000' #client ID
# my_secret = '0000000000000000000000' #client secret
ccm = SpotifyClientCredentials(client_id = client_id, client_secret = client_secret)
spotify = spotipy.Spotify(client_credentials_manager = ccm)

results = spotify.audio_features(track_url)
result = results[0]
for key, val in result.items():
    print(f'{key} : {val}')


"*****"