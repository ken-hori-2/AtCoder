from langchain.agents import tool

@tool
def get_current_weather(location: str) -> str:
    """The city and state, e.g. San Francisco, CA."""
    # return "{'weather':'rain'}"
    # return {'weather':'rain'}
    weather = "sunny"
    target_date = "2024/05/03"
    try:
        return {'weather':weather, 'date':target_date, 'location':location}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'weather':'unknown', 'date':target_date, 'location':location}