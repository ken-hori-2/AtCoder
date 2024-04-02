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
# input_string = "現在地周辺の居酒屋" # input("explorer search:")






# GoogleMapを1コマンドで画面に表示

# memo
start = "本厚木駅"
goal = "みなとみらい駅"
# loc_x = 35.4792251
# loc_y = 139.3353584
# input_string = "https://www.google.com/maps/dir/"+ start +"/" + goal +"/@" + loc_x + "," + loc_y + ",11z/data=!4m2!4m1!3e0?entry=ttu"
input_string = "https://www.google.com/maps/dir/"+ start +"/" + goal

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
time.sleep(7) # 5)

# Ctrl+Vを使ってクリップボードの内容（ユーザーが入力した検索クエリ）を貼り付け
pyautogui.hotkey('ctrl', 'v')

# Enterキーを押して、クエリを実行
pyautogui.hotkey('enter')