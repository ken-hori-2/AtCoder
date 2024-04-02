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
input_string = "現在地周辺の居酒屋" # input("explorer search:")

# 入力されたクエリをクリップボードにコピー
# pyperclip.copy(input_string)
# pyperclip.copy(Open_LLM)
pyperclip.copy(input_string)

# ウィンドウズキー（Winキー）と「s」を組み合わせて、Windowsの検索ボックスを開く
pyautogui.hotkey('winleft', 's')

# "explorer" というテキストを検索ボックスに入力
Serach_Chrome = "Chrome"
pyautogui.typewrite(Serach_Chrome) # 'Microsoft Edge') # explorer')

# Enterキーを押して、検索を実行
pyautogui.hotkey('enter')

# 5秒間待機（検索結果が表示されるのを待つため）
time.sleep(5)

# Ctrl+Vを使ってクリップボードの内容（ユーザーが入力した検索クエリ）を貼り付け
pyautogui.hotkey('ctrl', 'v')

# Enterキーを押して、クエリを実行
pyautogui.hotkey('enter')

# # 2秒間待機
# # time.sleep(2)現在地周辺の居酒屋

# time.sleep(2) # 5)

# # 2024/03/28 追加
# # 入力されたクエリをクリップボードにコピー
# input_string = "天気教えて"
# pyperclip.copy(input_string)
# # 2秒間待機
# time.sleep(2)
# # Ctrl+Vを使ってクリップボードの内容（ユーザーが入力した検索クエリ）を貼り付け
# pyautogui.hotkey('ctrl', 'v')
# # Enterキーを押して、クエリを実行
# pyautogui.hotkey('enter')

# # 2秒間待機
# time.sleep(2)

# # Tabキーを2回押して、最初の検索結果にフォーカスを当て、Enterキーを押して選択
# pyautogui.hotkey('tab', 'tab')
# pyautogui.hotkey('enter')

