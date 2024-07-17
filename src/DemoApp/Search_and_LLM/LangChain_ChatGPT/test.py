def run(received_value):
    # hexValueを対応する文字列に置き換える
    activity_map = {
        "00": "STABLE",
        "01": "WALKING",
        "02": "RUNNING"
    }
    activity = activity_map.get(received_value, "unknown")

    # ここでは例として、activityをログに出力します
    print(f"Activity: {activity}")

if __name__ == "__main__":
    run("00")
    run("01")
    run("02")