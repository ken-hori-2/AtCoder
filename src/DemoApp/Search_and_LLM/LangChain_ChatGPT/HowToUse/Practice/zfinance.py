from langchain.agents import tool
from datetime import datetime, timedelta
import yfinance as yf
from mypy_extensions import Arg, KwArg
from typing import Any, Dict, List, Optional, Callable, Tuple
from langchain_core.tools import BaseTool

# =================================================================================
# 関数作成
# =================================================================================

@tool
def current_stock_price(ticker:str) -> dict:
    """株式銘柄（米）の最新株価取得する関数""" # description
    data   = yf.Ticker(ticker).history(period="1d")
    result = {"price": data.iloc[0]["Close"], "currency": "USD"}
    return result

# @tool
# def stock_performance(ticker:str, days:int) -> dict:
#     """ 過去days間で変動した株価[%] """ # description
#     start_date    = datetime.today() - timedelta(days=days)
#     ticker_data   = yf.Ticker(ticker)
#     history       = ticker_data.history(start=start_date)
#     old_price     = history.iloc[0]["Close"]
#     current_price = history.iloc[-1]["Close"]
#     result        = {"percent_change": ((current_price - old_price) / old_price) * 100}
#     return result

# @tool
# # def music(**kwargs: Any) -> BaseTool:
# def music(): #  -> BaseTool:
#     """useful for when you need to answer questions about current events"""
#     return "'All I Want For Christmas Is You' by Mariah Carey."


# =================================================================================
# ツール定義
# =================================================================================

# # tools = [current_stock_price, stock_performance]
# # 関数実行
# current_stock_price("AAPL")
# # 出力イメージ
# # {'price': 195.7100067138672, 'currency': 'USD'}


# # 関数実行
# stock_performance("GOOG", 30)
# # 出力イメージ
# # {'percent_change': 3.7588251624831}