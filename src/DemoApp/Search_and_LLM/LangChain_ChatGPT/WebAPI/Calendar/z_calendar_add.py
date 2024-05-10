import datetime
import googleapiclient.discovery
import google.auth


# ①Google APIの準備をする

SCOPES = ['https://www.googleapis.com/auth/calendar']
calendar_id = 'stmuymte@gmail.com' # '自身のGoogleカレンダーIDを記述'
# Googleの認証情報をファイルから読み込む
gapi_creds = google.auth.load_credentials_from_file('自身の鍵ファイル名.jsonへのパス', SCOPES)[0]
# APIと対話するためのResourceオブジェクトを構築する
service = googleapiclient.discovery.build('calendar', 'v3', credentials=gapi_creds)


# ②予定を書き込む
# 書き込む予定情報を用意する
body = {
    # 予定のタイトル
    'summary': 'ミーティング③',
    # 予定の開始時刻
    'start': {
        'dateTime': datetime.datetime(2022, 2, 6, 10, 30).isoformat(),
        'timeZone': 'Japan'
    },
    # 予定の終了時刻
    'end': {
        'dateTime': datetime.datetime(2022, 2, 6, 12, 00).isoformat(),
        'timeZone': 'Japan'
    },
}
# 用意した予定を登録する
event = service.events().insert(calendarId=calendar_id, body=body).execute()