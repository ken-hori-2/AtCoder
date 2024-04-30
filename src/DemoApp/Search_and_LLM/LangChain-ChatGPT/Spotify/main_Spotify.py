from langchain import LLMMathChain, OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType, OpenAIFunctionsAgent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory


"""
Toolの定義
Spotifyに保存した曲リストと、特徴量を取得するツール
"""

from langchain.tools.base import BaseTool
import spotipy
import json
import os

from dotenv import load_dotenv
# .envファイルの内容を読み込見込む
load_dotenv()
# os.environを用いて環境変数を表示させます



# # OpneAIと Spotify APIキー
# %env OPENAI_API_KEY = sk-
# %env SPOTIFY_TOKEN = 
# # https://developer.spotify.com/documentation/web-api/reference/get-current-users-profile
# %env SPOTIFY_USER_ID = 

import spotipy.util as util
username = os.environ['UserName']
scope = 'user-read-playback-state,playlist-read-private,user-modify-playback-state,playlist-modify-public'
client_id = os.environ['SPOTIFY_USER_ID'] # Client_ID'] # ここに自分の client ID'
client_secret = os.environ['SPOTIFY_TOKEN'] # Client_Secret'] # ここに自分の client seret'
redirect_uri = 'http://localhost:8888/callback'

# search_str = 'YOASOBI' # sys.argv[1] # 実行時の引数でキーワード入力
# artist_id_map={}
# token 
my_token = util.prompt_for_user_token(username, scope,     client_id, client_secret, redirect_uri)
# sp
my_sp = spotipy.Spotify(auth=my_token) # token)



class SpotifyTool(BaseTool):
    """Tool that fetches audio features of saved tracks from Spotify."""

    name = "SpotifyTool"
    description = (
        "A tool that fetches audio features of the most recently saved tracks from Spotify."
        "This tool does not require any arguments."
    )

    def _run(self, *args, **kwargs) -> str:
        token = my_token # os.getenv('SPOTIFY_TOKEN')
        if not token:
            raise ValueError("SPOTIFY_TOKEN environment variable is not set.")

        sp = my_sp # spotipy.Spotify(auth=token)
        # result = sp.current_user_saved_tracks(limit=1) # 30)
        """***"""
        import random
        # playlist = sp.user_playlist_create(username,"Selected Langchain")
        # playlist_id = playlist['id']
        search_str = 'YOASOBI'
        result = sp.search(q='artist:'+search_str, limit=1)
        # artist_id = result['tracks']['items'][0]['artists'][0]['id']
        # artist_related_artists = sp.artist_related_artists(artist_id)
        # track_ids = []
        # for artist_list in artist_related_artists['artists']:
        #     result = sp.search(q='artist:'+artist_list['name'], limit=10) # 50)
        #     if len(result['tracks']['items']) > 1:
        #         track_ids.append(random.choice(result['tracks']['items'])['id'])
        """***"""
        # sp.user_playlist_add_tracks(username, playlist_id, track_ids)

        # 仮定: result['items'] はトラックのリスト
        tracks = [item['track']['id'] for item in result['items']]
        # 各トラックのオーディオ特性を取得
        audio_features_list = [sp.audio_features(track)[0] for track in tracks]

        # uriとtrack_hrefを削除
        for features in audio_features_list:
            if 'uri' in features:
                del features['uri']
            if 'track_href' in features:
                del features['track_href']
            if 'analysis_url' in features:
                del features['analysis_url']

        # # JSON形式に変換
        audio_features_json = json.dumps(audio_features_list)
        # audio_features_json = json.dumps(track_ids)

        return audio_features_json

    async def _arun(self, *args, **kwargs) -> str:
        """Use the SpotifyTool asynchronously."""
        return self._run()

"""
Spotifyのプレーリストを作成して、曲を追加するツール
"""

from langchain.tools.base import BaseTool
import spotipy
import os

class SpotifyPlaylistTool(BaseTool):
    """Tool that creates a new playlist and adds tracks to it on Spotify."""

    name = "SpotifyPlaylistTool"
    description = (
        "A tool that creates a new playlist and adds tracks to it on Spotify."
        "This tool requires a list of track IDs (list of strings), a playlist name (string), and a playlist description (string) as keyword arguments."
        "The track IDs should be a list of Spotify track IDs to add to the playlist."
        "The playlist name is the name of the new playlist to be created."
        "The playlist description is a description for the new playlist."
        "The arguments should be passed as keyword arguments like so: tool._run(track_ids=['xxxxxxxxxxx','xxxxxxxxxxx'], playlist_name='the dance music', playlist_description='danceabe music')"
    )

    def _run(self, track_ids, playlist_name, playlist_description) -> str:
        user = client_id # os.getenv('SPOTIFY_USER_ID')
        if not user:
            raise ValueError("SPOTIFY_USER_ID environment variable is not set.")
        
        token = my_token # os.getenv('SPOTIFY_TOKEN')
        if not token:
            raise ValueError("SPOTIFY_TOKEN environment variable is not set.")
        
        sp = my_sp # spotipy.Spotify(auth=token)

        # Create a new playlist
        # user_playlist = sp.user_playlist_create(user, playlist_name, public=False, collaborative=False, description=playlist_description)
        """***"""
        user_playlist = sp.user_playlist_create(user,playlist_name, public=False, collaborative=False, description=playlist_description) # "Playlist by Langchain")
        # playlist_id = playlist['id']
        # user_playlist = sp.user_playlist_add_tracks(username, playlist_id, track_ids)
        """***"""

        # Add tracks to the playlist
        sp.playlist_add_items(user_playlist['id'], items=track_ids, position=None)

        return f"Playlist '{playlist_name}' created with {len(track_ids)} tracks."

    async def _arun(self, track_ids, playlist_name, playlist_description) -> str:
        """Use the SpotifyPlaylistTool asynchronously."""
        return self._run(track_ids, playlist_name, playlist_description)


llm = ChatOpenAI(temperature=0) # , model="gpt-4-0613")

tools = [
    SpotifyTool(),
    # SpotifyPlaylistTool()
]

content = """
            あなたについて:
            あなたはツールとして定義されたspotifyのAPIを操作してUserの要望に答えるAI Agentです。

            実行について:
            あなたはUserの入力に対して、Tool を使った実行計画を立ててレビューを求めてから実行してください

            SpotifyToolの戻り値のパラメータの説明:
            {
                "acousticness": {
                    "description": "Confidence measure of whether the track is acoustic.",
                    "example_value": 0.00242,
                    "range": "0 - 1"
                },
                "danceability": {
                    "description": "How suitable a track is for dancing.",
                    "example_value": 0.585
                },
                "duration_ms": {
                    "description": "Track duration in milliseconds.",
                    "example_value": 237040
                },
                "energy": {
                    "description": "Perceptual measure of intensity and activity.",
                    "example_value": 0.842
                },
                "id": {
                    "description": "Spotify ID for the track.",
                    "example_value": "2takcwOaAZWiXQijPHIx7B"
                },
                "instrumentalness": {
                    "description": "Predicts if a track contains no vocals.",
                    "example_value": 0.00686
                },
                "key": {
                    "description": "The key the track is in.",
                    "example_value": 9,
                    "range": "-1 - 11"
                },
                "liveness": {
                    "description": "Presence of an audience in the recording.",
                    "example_value": 0.0866
                },
                "loudness": {
                    "description": "Overall loudness of a track in decibels (dB).",
                    "example_value": -5.883
                },
                "mode": {
                    "description": "Modality (major or minor) of a track.",
                    "example_value": 0
                },
                "speechiness": {
                    "description": "Presence of spoken words in a track.",
                    "example_value": 0.0556
                },
                "tempo": {
                    "description": "Estimated tempo of a track in BPM.",
                    "example_value": 118.211
                },
                "time_signature": {
                    "description": "Estimated time signature.",
                    "example_value": 4,
                    "range": "3 - 7"
                },
                "type": {
                    "description": "Object type.",
                    "allowed_values": "audio_features"
                },
                "valence": {
                    "description": "Musical positiveness conveyed by a track.",
                    "example_value": 0.428,
                    "range": "0 - 1"
                }
            }
            """

"""
AI Agentの定義
"""
from langchain.schema.messages import (
    SystemMessage,
)

from langchain.prompts import MessagesPlaceholder
agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": SystemMessage(
            content= content
        ),
}
memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
mrkl = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, agent_kwargs=agent_kwargs, memory=memory, verbose=True)

user = ""
while user != "exit":
    user = input("入力してください:")
    print(user)
    ai = mrkl.run(input=user)
    print(ai)