from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


"""
これが重要
action 文字列によって、 Google Calendar で呼び出すエンドポイントを決定しています。FieldのDescription にそれぞれの内容を説明させます。
"""

class CalendarAction(BaseModel):
    action: str = Field(description="The action to be performed. Supported actions are 'get', 'create', 'search', 'update', and 'delete'.")

    class EventData(BaseModel):
        summary: Optional[str] = Field(None, description="The summary of the event.")
        start: Optional[Dict[str, str]] = Field(None, description="The start time of the event. It contains 'dateTime' and 'timeZone' keys.")
        end: Optional[Dict[str, str]] = Field(None, description="The end time of the event. It contains 'dateTime' and 'timeZone' keys.")
        query: Optional[str] = Field(None, description="The search query to find matching events.")
        updated_data: Optional[Dict[str, Any]] = Field(None, description="The updated data for the event. It contains keys to update such as 'summary', 'start', and 'end'.")

    event_data: Optional[EventData] = Field(None, description="The data for the event or query. Contains fields such as 'summary', 'start', 'end', 'query', and 'updated_data'.")