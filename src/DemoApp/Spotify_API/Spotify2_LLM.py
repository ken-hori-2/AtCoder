import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

import openai
import os
import re
import pprint

from dotenv import load_dotenv
# .envファイルの内容を読み込見込む
load_dotenv()
# os.environを用いて環境変数を表示させます
print(os.environ['OpenAI_API_KEY'])
key = os.environ['OpenAI_API_KEY']

openai.api_key = os.getenv(key) # "CHATGPT_API_KEY")
sKillReg = re.compile(r"^\d+\. ", re.MULTILINE)




# 2024/04/25
# LLMでプレイリスト生成






# from openai import OpenAI

conversationHistory = []
# user_action = {"role": "user", "content": f"10 words to search for on Spotify when you want to create a '{theme}' themed songs playlist:\n1. "},
# user_action = {"role": "user", "content": text}
# conversationHistory.append(user_action)
def getSearchWords(theme, conversationHistory):

    # client = OpenAI(
    #     api_key = key # os.getenv("OPENAI_API_KEY"),
    # )

    # completion = client.chat.completions.create(
    # res = client.chat.completions.create(
    #     # model = "gpt-3.5-turbo",
    #     # messages = [
    #     #     {"role": "system", "content": "You are a helpful assistant."},
    #     #     {"role": "user", "content": "Hello!"},
    #     # ]
    # )
    # APIリクエストを作成する
    # conversationHistory = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": f"10 words to search for on Spotify when you want to create a '{theme}' themed songs playlist:\n1. "},
    # ]
    

    response = openai.chat.completions.create(
        messages=conversationHistory,
        max_tokens=1024,
        n=1,
        stream=True,
        temperature=0.5,
        stop=None,
        presence_penalty=0.5,
        frequency_penalty=0.5,
        model="gpt-3.5-turbo"
        # model="text-davinci-003"
    )
    # res = openai.Completion.create(
    #     model="text-davinci-003",
    #     prompt=f"10 words to search for on Spotify when you want to create a '{theme}' themed songs playlist:\n1. ",
    #     suffix="",
    #     temperature=1,
    #     max_tokens=256,
    #     top_p=1,
    #     frequency_penalty=0,
    #     presence_penalty=0
    # )
    # return sKillReg.sub("", response["choices"][0]["text"]).split("\n")
    
    # ストリーミングされたテキストを処理する
    fullResponse = ""
    RealTimeResponce = ""   
    for chunk in response:
        text = chunk.choices[0].delta.content # chunk['choices'][0]['delta'].get('content')

        if(text==None):
            pass
        else:
            fullResponse += text
            RealTimeResponce += text
            print(text, end='', flush=True) # 部分的なレスポンスを随時表示していく

            # target_char = ["。", "！", "？", "\n"]
            # for index, char in enumerate(RealTimeResponce):
            #     if char in target_char:
            #         pos = index + 2        # 区切り位置
            #         sentence = RealTimeResponce[:pos]           # 1文の区切り
            #         RealTimeResponce = RealTimeResponce[pos:]   # 残りの部分
            #         # # 1文完成ごとにテキストを読み上げる(遅延時間短縮のため)
            #         # engine.say(sentence)
            #         # engine.runAndWait()
            #         break
            #     else:
            #         pass

    # APIからの完全なレスポンスを返す
    return fullResponse

def chat(conversationHistory):
    # APIリクエストを作成する
    response = openai.chat.completions.create(
        messages=conversationHistory,
        max_tokens=1024,
        n=1,
        stream=True,
        temperature=0.5,
        stop=None,
        presence_penalty=0.5,
        frequency_penalty=0.5,
        model="gpt-3.5-turbo"
    )

    # ストリーミングされたテキストを処理する
    fullResponse = ""
    RealTimeResponce = ""   
    for chunk in response:
        text = chunk.choices[0].delta.content # chunk['choices'][0]['delta'].get('content')

        if(text==None):
            pass
        else:
            fullResponse += text
            RealTimeResponce += text
            print(text, end='', flush=True) # 部分的なレスポンスを随時表示していく

            target_char = ["。", "！", "？", "\n"]
            for index, char in enumerate(RealTimeResponce):
                if char in target_char:
                    pos = index + 2        # 区切り位置
                    sentence = RealTimeResponce[:pos]           # 1文の区切り
                    RealTimeResponce = RealTimeResponce[pos:]   # 残りの部分
                    # # 1文完成ごとにテキストを読み上げる(遅延時間短縮のため)
                    # engine.say(sentence)
                    # engine.runAndWait()
                    break
                else:
                    pass

    # APIからの完全なレスポンスを返す
    return fullResponse

# 10 words to search for on Spotify when you want to create a '{ここにテーマ}' themed songs playlist:
# 1. [Insert]




# SPOTIFY_CLIENT_ID = os.getenv(id) # "SPOTIFY_CLIENT_ID")
# SPOTIFY_CLIENT_SECRET = os.getenv(secret) # "SPOTIFY_CLIENT_SECRET")
# ccm = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
# spotify = spotipy.Spotify(client_credentials_manager=ccm)

import os
from dotenv import load_dotenv
# .envファイルの内容を読み込見込む
load_dotenv()
# os.environを用いて環境変数を表示させます
client_id = os.environ['Client_ID'] # ここに自分の client ID'
client_secret = os.environ['Client_Secret'] # ここに自分の client seret'

ccm = SpotifyClientCredentials(client_id = client_id, client_secret = client_secret)
spotify = spotipy.Spotify(client_credentials_manager = ccm)
# results = spotify.artist_top_tracks(lz_uri)

def searchMusic(words, additional_word, market):
    id_list = []
    meta_list = []
    while len(words) == 1:
        words = words[0]
    for word in words:
        sq = word
        if len(additional_word):
            sq += " " + additional_word
        
        res = spotify.search(sq, limit=10, offset=0, type='track', market=None) # market)

        for track in res['tracks']['items']:
            id_list.append(track['id'])
            meta_list.append({
                "id": track["id"],
                "title": track["name"],
                "artist":",".join([x["name"] for x in track["artists"]]),
                "uri": track["uri"],
            })
    features = spotify.audio_features(id_list)
    c = 0
    for f in features:
        for k in ["tempo", "energy", "instrumentalness", "duration_ms"]:
            meta_list[c][k] = f[k]
        c += 1
    
    print("list : ", meta_list)
    return meta_list


# I am thinking of making a playlist about '{テーマ}'. I just searched Spotify for songs to put in the playlist and found the following {検索して手に入った曲数} songs. Please choose {プレイリストの曲数} songs from these {検索して手に入った曲数} songs to make a playlist. The playlist should be in the form of a Markdown numbered list,  Don't just arrange the songs, rearrange them with the order in mind. Do not include BPM in the result

def createPlayList(theme,meta_list,tracks_length):
    # prompt = f"I am thinking of making a playlist about '{theme}'. I just searched Spotify for songs to put in the playlist and found the following {len(meta_list)} songs. Please choose {tracks_length} songs from these {len(meta_list)} songs to make a playlist. The playlist should be in the form of a Markdown numbered list,  Don't just arrange the songs, rearrange them with the order in mind. Do not include BPM in the result.\n\n"
    prompt = f"{theme}に関するプレイリストを作ろうと思っている。プレイリストに入れる曲をSpotifyで検索したところ、以下の{len(meta_list)}曲が見つかりました。この{len(meta_list)}曲の中から{tracks_length}曲を選んでプレイリストを作ってください。プレイリストはMarkdownの番号付きリストの形式にしてください。ただ曲を並べるのではなく、順番を意識して並べ直してください。結果にはBPMを含めないでください。"
    c = 1
    for i in meta_list:
        prompt += f"No{c}: {i['title']} - {i['artist']} BPM:{i['tempo']}\n"
        c += 1
    # res = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": prompt
    #         }
    #     ]
    # )
    res = openai.chat.completions.create(
        messages=[{"role": "system", "content": prompt}], # conversationHistory, # とりあえず以前の会話は内容は伏せる
        max_tokens=1024,
        n=1,
        stream=True,
        temperature=0.5,
        stop=None,
        presence_penalty=0.5,
        frequency_penalty=0.5,
        model="gpt-3.5-turbo"
        # model="text-davinci-003"
    )
    ressp = sKillReg.sub("", res["choices"][0]["message"]["content"]).split("\n")
    result_ids = []
    for m in ressp:
        sp = m.split(" - ")
        title_match = list(filter(lambda x:x["title"] == sp[0],meta_list))
        if len(title_match):
            title_artist_match =  list(filter(lambda x:x["artist"] == sp[1],title_match))
            if len(title_artist_match):
                result_ids.append(title_artist_match[0])
            else:
                result_ids.append(title_match[0])

    return result_ids #result_idsと言っているが実際はresult　直すのがめんどいです（カス）


def generate(theme,tracks_length,market,additional_word):

    # text = f"10 words to search for on Spotify when you want to create a '{theme}' themed songs playlist:\n1. "
    num = 3 # 10
    # text = f"{num} words to search for on Spotify when you want to create a '{theme}' themed songs playlist:\n" # 1. "
    text = f"{theme}をテーマにした曲のプレイリストを作りたいときにSpotifyで検索すべき{num}個の単語："
    user_action = {"role": "user", "content": text}
    conversationHistory.append(user_action)

    words = getSearchWords(theme, conversationHistory)
    # print("\n***** word: ", words)
    # searchResult = searchMusic(words,additional_word,market)
    # playlist = createPlayList(theme,searchResult,tracks_length)
    # return playlist
    
    return words # add



theme = "気分が上がる曲" # "J-pop"
tracks_length = 3 # 10
market = 10
additional_word = "ドライブ旅行" # "Driving"

# generate(theme,tracks_length,market,additional_word)
words = generate(theme,tracks_length,market,additional_word)


pprint.pprint(words)












# 現在状態からテーマと紐づけ（running検出なら、テーマ=ランニング）
# word + additional word のように、「気分＋検出状態」のように組み合わせてもいいかも

# テーマからいくつかの検索ワード生成
# 検索ワードをもとにSpotifyで曲検索
# 完成したリストからLLMに指定数選ばせる












# # import spotipy
# # from spotipy.oauth2 import SpotifyClientCredentials
# import sys
# import pprint

# if len(sys.argv) > 1:
#         search_str = sys.argv[1]
# else:
#         search_str = words # 'Radiohead'

# print("\nserch str:")
# pprint.pprint(search_str)

# client_credentials_manager = SpotifyClientCredentials(client_id = client_id, client_secret = client_secret) # spotipy.oauth2.SpotifyClientCredentials(my_id, my_secret)
# sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
# # result = sp.search(q='track:Radiohead', limit=10, offset=0, type='track', market=None) #sp.search(search_str)
# result = sp.search(q=search_str, limit=3, offset=0, type='track', market=None) #sp.search(search_str)


# id_list = []
# for track in result['tracks']['items']:
#         id = track['id']
#         id_list.append(id)





# # features = sp.audio_features(id_list)
# # print("\n*****")
# # # pprint.pprint(features)
# id_list = []
# meta_list = []
# # while len(words) == 1:
# #     words = words[0]
# # for word in words:
# #     sq = word
# #     if len(additional_word):
# #         sq += " " + additional_word
# #     print("word : ", sq)
# # 2024/04/23
# for track in result['tracks']['items']:
#             id_list.append(track['id'])
#             meta_list.append({
#                 "id": track["id"],
#                 "title": track["name"],
#                 "artist":",".join([x["name"] for x in track["artists"]]),
#                 "uri": track["uri"],
#                 # "tempo": track["tempo"],
#                 # "danceability": track["danceability"],
#             })

# print("\n*****")
# print("ID List：")
# pprint.pprint(id_list)
# print("\n*****")
# print("Meta List：")
# pprint.pprint(meta_list)

# print("\n*****")
# print("詳細情報：")
# features = sp.audio_features(id_list)
# print("\n*****")
# pprint.pprint(features)

# # c = 0
# # for f in features:
# #     for k in ["tempo", "energy", "instrumentalness", "duration_ms"]:
# #         meta_list[c][k] = f[k]
# #     c += 1

# # print("list : ", meta_list)
