import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import os
import datetime
from google.oauth2 import service_account
# from google_auth_oauthlib.flow import InstalledAppFlow
import google.auth

from google.auth.transport.requests import Request
from googleapiclient.discovery import build

from Calendar.Pydantic import CalendarAction

class GoogleCalendar():
    def __init__(self, credentials_file, scopes=['https://www.googleapis.com/auth/calendar']):
        self.scopes = scopes
        self.credentials = self._get_credentials(credentials_file)
        if self.credentials is None:
            raise ValueError("Error: 認証情報が見つかりませんでした。")
        self.service = build('calendar', 'v3', credentials=self.credentials)

    def _get_credentials(self, credentials_file):
        # クレデンシャルファイルから認証情報を取得
        gapi_creds = None
        if os.path.exists(credentials_file):
            # flow = InstalledAppFlow.from_client_secrets_file(credentials_file, self.scopes)
            # Googleの認証情報をファイルから読み込む
            gapi_creds = google.auth.load_credentials_from_file('credentials.json', self.scopes)[0]
            # creds = flow.run_local_server(port=0)
            print(gapi_creds)
        return gapi_creds # creds

    def get_events(self, calendar_id='primary', time_min=None, time_max=None):
        # 現在の日時または指定された期間内のイベントを取得
        if time_min is None:
            time_min = datetime.datetime.utcnow().isoformat() + 'Z'
        if time_max is None:
            time_max = (datetime.datetime.utcnow() + datetime.timedelta(weeks=1)).isoformat() + 'Z'

        events_result = self.service.events().list(
            calendarId=calendar_id,
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        events = events_result.get('items', [])
        return events

    def create_event(self, event_data, calendar_id='primary'):
        """
        予定の追加はこのままで大丈夫
        """
        # 新しいイベントを作成
        created_event = self.service.events().insert(
            calendarId='stmuymte@gmail.com', # calendar_id, 
            body=event_data
        ).execute()
        return created_event

        # ②予定を書き込む
        now = datetime.datetime.now(datetime.UTC).isoformat() # 新しいやり方
        # 書き込む予定情報を用意する
        body = {
            # 予定のタイトル
            'summary': 'ミーティング③',
            # 予定の開始時刻
            'start': {
                'dateTime': now, # datetime.datetime(2022, 2, 6, 10, 30).isoformat(),
                'timeZone': 'Japan'
            },
            # 予定の終了時刻
            'end': {
                'dateTime': now, # datetime.datetime(2022, 2, 6, 12, 00).isoformat(),
                'timeZone': 'Japan'
            },
        }
        # 用意した予定を登録する
        create_event = self.service.events().insert(
            calendarId='stmuymte@gmail.com', # calendar_id, 
            body=body
        ).execute()
        return create_event

    def update_event(self, event_id, updated_event_data, calendar_id='primary'):
        # 既存のイベントを更新
        updated_event = self.service.events().patch(
            calendarId='stmuymte@gmail.com', # calendar_id, 
            eventId=event_id, 
            body=updated_event_data
            ).execute()
        return updated_event

    def delete_event(self, event_id, calendar_id='primary'):
        # イベントを削除
        self.service.events().delete(calendarId=calendar_id, eventId=event_id).execute()

    def search_events(self, query, calendar_id='primary'):
        # イベントを検索
        # events_result = self.service.events().list(
        #     calendarId=calendar_id,
        #     q=query,
        #     singleEvents=True,
        #     orderBy='startTime'
        # ).execute()

        # events = events_result.get('items', [])
        # return events
        
        """
        試しに変更 -> できた！！！！！
        ただし、calendar_idをしっかり入れないと自分のメアドの予定は表示されない
        """
        import googleapiclient.discovery
        # service = self.service
        now = datetime.datetime.now(datetime.UTC).isoformat() # 新しいやり方
        # 直近3件のイベントを取得する
        event_list = self.service.events().list(
            calendarId='stmuymte@gmail.com', timeMin=now,
            maxResults=3, singleEvents=True,
            orderBy='startTime').execute()


        # ③イベントの開始時刻、終了時刻、概要を取得する
        events = event_list.get('items', [])
        formatted_events = [(event['start'].get('dateTime', event['start'].get('date')), # start time or day
            event['end'].get('dateTime', event['end'].get('date')), # end time or day
            event['summary']) for event in events]
        """
        試しに変更
        """
        return events
    




    def process_json_data(self, json_data: str):
        data = CalendarAction.parse_raw(json_data)

        if data.action == "get":
            events = self.get_events()
            for event in events:
                print(event['summary'], event['start']['dateTime'])
            return events

        elif data.action == "create":
            event_data = data.event_data.dict()
            created_event = self.create_event(event_data)
            print("Event created:", created_event['id'])
            return "Finish."

        elif data.action == "search":
            query = data.event_data.query
            events = self.search_events(query)
            for event in events:
                print(event['summary'], event['start']['dateTime'])
            return events

        elif data.action == "update":
            query = data.event_data.query
            updated_data = data.event_data.updated_data

            today = datetime.datetime.utcnow().date()
            time_min = datetime.datetime.combine(today, datetime.time.min).isoformat() + 'Z'
            time_max = datetime.datetime.combine(today, datetime.time.max).isoformat() + 'Z'

            events = self.search_events(query, time_min=time_min, time_max=time_max)
            for event in events[:20]:
                updated_event = self.update_event(event['id'], updated_data)
                print("Event updated:", updated_event['id'])
            return "Finish."

        elif data.action == "delete":
            query = data.event_data.query

            # today = datetime.datetime.utcnow().date()
            today = datetime.datetime.now(datetime.UTC).date()
            time_min = datetime.datetime.combine(today, datetime.time.min).isoformat() + 'Z'
            time_max = datetime.datetime.combine(today, datetime.time.max).isoformat() + 'Z'

            events = self.search_events(query) # , time_min=time_min, time_max=time_max)
            for event in events[:20]:
                self.delete_event(event['id'])
                print("Event deleted:", event['id'])
            return "Finish."

        else:
            print("Invalid action")
            return "Finish with Error!"