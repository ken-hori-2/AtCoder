"""
SCO DEMO (6/19)
"""
import datetime

class SetTime():

    def run(self):
        # # SCOデモ
        # YYYY = 2024
        # MM = 6
        # DD = 19
        # # お披露目会デモ
        # YYYY = 2024
        # MM = 7
        # DD = 1

        # 2024/07/07
        YYYY = 2024
        MM = 7
        DD = 17

          # dt_now = datetime.datetime(YYYY, MM, DD, 7, 10)        # 天気情報 (今日より前の日付だとエラーになるかも)
        dt_now = datetime.datetime(YYYY, MM, DD, 8, 00) # 30)    # 出勤     (stable:楽曲再生[house-music], walking:経路検索) # 8:00, 8:30の場合、TimeActionのdbだと、stableの場合は天気情報になる
        dt_now = datetime.datetime(YYYY, MM, DD, 10, 58) # 55)         # 定例     (stable:何もしない, walk:会議情報)
        # dt_now = datetime.datetime(YYYY, MM, DD, 12, 5)        # 昼食     (walk:restaurant, stable:music[relax-music])
        # dt_now = datetime.datetime(YYYY, MM, DD, 19, 5)          # ジム     (run:up tempo, walk:slow tempo, stable:stop)   # 行動検出と連動モード

        return dt_now