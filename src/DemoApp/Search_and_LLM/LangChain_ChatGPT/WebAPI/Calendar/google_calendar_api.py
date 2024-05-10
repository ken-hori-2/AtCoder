import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from langchain.tools import BaseTool
from Calendar.google_calendar_client import GoogleCalendarClient
from datetime import datetime, timedelta

class GoogleCalendarTool(BaseTool):
    name = 'GoogleCalendarSearch'
    description = (
        'カレンダーの予定を探す場合に有用です。'
        'インプットは JSON で、以下の2つのキーを持っています: "date" と "n"'
        'date は、予定を検索する日付です。yyyy-mm-dd という形式か、「今日」、「明日」、「明後日」のいずれかです。'
        'n は、予定を検索する数です。'
    )
    # credentials_file: str

    def _run(self, query: dict[str, str]) -> str:
        # client = GoogleCalendarClient() # self.credentials_file) # 今はクラス化していない普通の関数なので初期化はしない


        date_query = query.get('date')

        if date_query == '今日':
            date = datetime.today()
        elif date_query == '明日':
            date = datetime.today() + timedelta(days=1)
        elif date_query == '明後日':
            date = datetime.today() + timedelta(days=2)
        elif date_query:
            date = datetime.strptime(query.get('date'), '%Y-%m-%d')
        else:
            return "I don't know what to do."

        n = query.get('n')
        # response = client.get_events(date, max_results=int(n))
        response = GoogleCalendarClient() # date, max_results=int(n))

        if len(response) == 0:
            return "予定はありませんでした。"
        else:
            events = [f"- 予定: {event['summary']}" for event in response]
            return f"{date_query} の予定は以下の通りです。\n\n" + "\n".join(events)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("does not support async")