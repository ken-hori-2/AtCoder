import os
# カレントディレクトリの取得のときに区切り文字を「/」に置き換え
current_dir = os.getcwd().replace(os.sep,'\\\\')
print(current_dir)
# C:/testDir