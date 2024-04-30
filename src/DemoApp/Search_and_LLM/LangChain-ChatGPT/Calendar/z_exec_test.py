# from z_GoogleCalendarLib import GoogleCalendar

# def main():
#     # GoogleカレンダーAPI認証情報ファイルのパスを指定
#     credentials_file = "credentials.json"

#     # GoogleCalendarクラスのインスタンスを作成
#     gcal = GoogleCalendar(credentials_file)

#     # イベントを取得
#     print("取得したイベント:")
#     events = gcal.get_events()
#     for event in events:
#         start = event['start'].get('dateTime', event['start'].get('date'))
#         print(f"{start} - {event['summary']}")

#     # 新しいイベントを作成
#     event_data = {
#         'summary': 'Test Event',
#         'start': {
#             'dateTime': '2023-04-01T10:00:00',
#             'timeZone': 'Asia/Tokyo',
#         },
#         'end': {
#             'dateTime': '2023-04-01T12:00:00',
#             'timeZone': 'Asia/Tokyo',
#         },
#     }
#     created_event = gcal.create_event(event_data)
#     print(f"作成したイベント: {created_event['id']} - {created_event['summary']}")

#     # イベントを検索
#     print("検索結果:")
#     search_results = gcal.search_events(query='Test Event')
#     for event in search_results:
#         print(f"{event['id']} - {event['summary']}")

#     # イベントを更新
#     updated_event_data = {
#         'summary': 'Updated Test Event',
#     }
#     updated_event = gcal.update_event(created_event['id'], updated_event_data)
#     print(f"更新したイベント: {updated_event['id']} - {updated_event['summary']}")

#     # イベントを削除
#     gcal.delete_event(created_event['id'])
#     print(f"削除したイベント: {created_event['id']}")

# if __name__ == "__main__":
#     main()

credentials_file = "credentials.json"
from GCalTool import GoogleCalendarTool
from langchain_openai import ChatOpenAI # 新しいやり方
from dotenv import load_dotenv
# .envファイルの内容を読み込見込む
load_dotenv()

# モデル作成
llm = ChatOpenAI(temperature=0)

calendar_tool = GoogleCalendarTool(credentials_file, llm=ChatOpenAI(temperature=0), memory=None)
# calendar_tool.run('明日の12時にデートの約束を入れて')
# calendar_tool.run("直近イベント3件")
calendar_tool.run("今日の予定を削除して。") # 教えて。")