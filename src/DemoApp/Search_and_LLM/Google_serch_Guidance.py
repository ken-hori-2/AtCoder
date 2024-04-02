import json                                 # 返却された検索結果の読み取りにつかう
from googleapiclient.discovery import build # APIへのアクセスにつかう
import pyttsx3
import os
from dotenv import load_dotenv
# .envファイルの内容を読み込見込む
load_dotenv()
# os.environを用いて環境変数を表示させます
print(os.environ['Google_Custom_ID'])
google_map_key = os.environ['GoogleMap_API_KEY']
id = os.environ['Google_Custom_ID']

# カスタム検索エンジンID
CUSTOM_SEARCH_ENGINE_ID = id # "XXXXX(接続先カスタム検索エンジンIDを入力)"
# API キー
API_KEY = google_map_key # "XXXXX(APIキーを入力)"


# APIにアクセスして結果をもらってくるメソッド
def get_search_results(query):
   
   # APIでやりとりするためのリソースを構築
   # 詳細: https://googleapis.github.io/google-api-python-client/docs/epy/googleapiclient.discovery-pysrc.html#build
   search = build(
       "customsearch", 
       "v1", 
       developerKey = API_KEY
   )
   
   # Google Custom Search から結果を取得
   # 詳細: https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list
   result = search.cse().list(
       q = query,
       cx = CUSTOM_SEARCH_ENGINE_ID,
       lr = 'lang_ja',
       num = 10,
       start = 1
   ).execute()

   # 受け取ったjsonをそのまま返却
   return result


# 検索結果の情報をSearchResultに格納してリストで返す
def summarize_search_results(result):

   # 結果のjsonから検索結果の部分を取り出しておく
   result_items_part = result['items']

   # 抽出した検索結果の情報はこのリストにまとめる
   result_items = []
   
   # 今回は (start =) 1 個目の結果から (num =) 10 個の結果を取得した
   for i in range(0, 10):
       # i番目の検索結果の部分
       result_item = result_items_part[i]
       # i番目の検索結果からそれぞれの属性の情報をResultクラスに格納して
       # result_items リストに追加する
       result_items.append(
           SearchResult(
               title = result_item['title'],
               
               url = result_item['link'], # 2024/04/02 コメントアウト
               snippet = result_item['snippet'],
               rank = i + 1
           )
       )

   # 結果を格納したリストを返却
   return result_items

       
# 検索結果の情報を格納するクラス
class SearchResult:
   def __init__(self, title, url, snippet, rank):
       self.title = title
       self.url = url
       self.snippet = snippet
       self.rank = rank

   def __str__(self):
       # コマンドライン上での表示形式はご自由にどうぞ
       # return "[title] " + self.title + "\n\t[url] " + self.url + "\n\t[snippet] " + self.snippet + "\n\t[rank] " + str(self.rank)
       return "[webサイト名称] " + self.title + "\n\t[要約] " + self.snippet # url無しバージョン



#################################
# Pyttsx3でレスポンス内容を読み上げ #
#################################
def text_to_speech(text):
    # テキストを読み上げる
    engine.say(text)
    engine.runAndWait()


# メインプロセス       
if __name__ == '__main__':

    # 検索キーワード
    query = "Note"
    query = '今日の東京の天気'

    # APIから検索結果を取得
    result = get_search_results(query) # result には 返却されたjsonが入る

    # 検索結果情報からタイトル, URL, スニペット, 検索結果の順位を抽出してまとめる
    result_items_list = summarize_search_results(result) # result_items_list には SearchResult のリストが入る

    # コマンドラインに検索結果の情報を出力
    # for i in range(0, 10):
    #     print(result_items_list[i])
    

    
    
    
    
    
    ##################
    # Pyttsx3を初期化 #
    ##################
    engine = pyttsx3.init()
    # 読み上げの速度を設定する
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate)
    # Kyokoさんに喋ってもらう(日本語)
    engine.setProperty('voice', "com.apple.ttsbundle.Kyoko-premium")

    # コマンドラインに検索結果の情報を出力
    for i in range(1): # 3): # 10):
        print(result_items_list[i])
        # 1文完成ごとにテキストを読み上げる(遅延時間短縮のため)
        sentence = result_items_list[i]
        engine.say(sentence)
        engine.runAndWait()