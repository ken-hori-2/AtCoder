# import pyautogui
import pyautogui
pyautogui.PAUSE = 1        # 関数呼出直後に何秒停止するかを指定
pyautogui.FAILSAFE = True  # フェールセーフ機能

# スクリーンショットを撮る
image = pyautogui.screenshot()

# スクリーンショットを保存
image.save("screenshot.png")




pyautogui.FAILSAFE = True  # フェールセーフ機能

# # 指定の場所をマウスでクリック
# pyautogui.click(500,             # X軸座標位置
#                 600,             # Y軸座標位置
#                 button = "left", # 左クリック
#                 )

# キーボードから文字列を送信
# pyautogui.typewrite("Hello")

# pyautogui.hotkey("ctrl","c")
import pyautogui  # PyAutoGUIライブラリをインポート
import pyperclip  # pyperclipライブラリをインポート（クリップボード操作用）
import time  # timeモジュールをインポート（時間待機用）
time.sleep(5)

# pyautogui.hotkey("ctrlleft", "c") # "ctrl","c")
pyautogui.hotkey("ctrl","c")

# PyAutoGUIで指定可能なキーを確認する
# pyautogui.KEYBOARD_KEYS