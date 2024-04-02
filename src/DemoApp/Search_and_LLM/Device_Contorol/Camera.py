# from selenium import webdriver

# driver = webdriver.Edge(executable_path="C:/Users/0107409377/Desktop/msedgedriver.exe")  
# # driver.get("http://google.com") 
# driver.get("https://www.bing.com/chat?q=Bing+AI&FORM=hpcodx")

import pyautogui  # PyAutoGUIライブラリをインポート
import pyperclip  # pyperclipライブラリをインポート（クリップボード操作用）
import time  # timeモジュールをインポート（時間待機用）

# ユーザーに検索クエリを入力するように要求
# input_string = input("explorer search:")
# Open_LLM = "https://www.bing.com/chat?q=Bing+AI&FORM=hpcodx" # input("https://www.bing.com/chat?q=Bing+AI&FORM=hpcodx")





# 1コマンドでChromeで検索




# いずれ音声入力にする
# input_string = "現在地周辺の居酒屋" # input("explorer search:")
input_time = 10

# ウィンドウズキー（Winキー）と「s」を組み合わせて、Windowsの検索ボックスを開く
pyautogui.hotkey('winleft', 's')
time.sleep(2)

# "explorer" というテキストを検索ボックスに入力
Boot_Camera = "Camera" # "カメラ"
pyautogui.typewrite(Boot_Camera) # 'Microsoft Edge') # explorer')

# Enterキーを押して、検索を実行
pyautogui.hotkey('enter')

# 5秒間待機（検索結果が表示されるのを待つため）
time.sleep(input_time) # 8) # 5)

# Ctrl+Vを使ってクリップボードの内容（ユーザーが入力した検索クエリ）を貼り付け
pyautogui.hotkey('space')