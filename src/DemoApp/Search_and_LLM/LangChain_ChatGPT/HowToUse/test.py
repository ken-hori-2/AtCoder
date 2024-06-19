import datetime
from datetime import timedelta
import time

# for i in range(10):
    
#     dt_now = datetime.datetime.now()
#     print(dt_now)
#     time.sleep(1)


dt_now = datetime.datetime.now() # 現在時刻
# print(dt_now)
start_date = datetime.date(dt_now.year, dt_now.month, dt_now.day)
print(start_date)
dt_now += datetime.timedelta(days=1)
# dt_now += timedelta(days=1)
end_date = datetime.date(dt_now.year, dt_now.month, dt_now.day) # 月末だと、同じ月の次の日がないのでエラーになる
print(end_date)


# dt_now = datetime.datetime(2024, 5, 24, 8, 00)
# print(type(dt_now))
# start_date = datetime.date(dt_now.year, dt_now.month, dt_now.day)
# print(type(start_date))