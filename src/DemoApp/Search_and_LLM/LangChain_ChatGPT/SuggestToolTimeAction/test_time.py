import datetime

margin = 5
margin = datetime.timedelta(minutes=margin)


import datetime
dt_now = datetime.datetime(2024, 5, 24, 11, 55) # 昼食(walk:restaurant, stable:music)
# dt_now_for_time_action = datetime.timedelta(hours=8, minutes=36) # 経路案内
print(int(dt_now.hour))
print(int(dt_now.minute))
dt_now_for_time_action = datetime.timedelta(hours=dt_now.hour, minutes=dt_now.minute)
print(dt_now_for_time_action)