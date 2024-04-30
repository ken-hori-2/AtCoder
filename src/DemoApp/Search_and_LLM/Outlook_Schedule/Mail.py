import win32com.client

# outlook = win32com.client.Dispatch("Outlook.Application")

# mail = outlook.CreateItem(0)

# mail.to = 'Kenji.B.Horiuchi@sony.com; hrucknj@icloud.com'
# mail.cc = 'stmuymte@gmail.com'
# # mail.bcc = 'momoko@mahodo.com'
# mail.subject = '1級試験の件'
# mail.bodyFormat = 1
# mail.body = '''お疲れ様です。どれみです。

# なんか今年の試験はコロナの影響でリモート面接になったらしいです。
# さっきマジョリカから聞きました。

# びっくりしたので取り急ぎご連絡です。

# よろしくお願いいたします。
# '''

# mail.display(True)




import win32com.client as win32

triggar = True

def send_mail_based_on_condition(condition):
    try:
        if condition:
            outlook = win32.Dispatch("Outlook.Application")
            mail = outlook.CreateItem(0)
            mail.Subject = "2024/04/26 条件を満たしたため、メールを送信します"
            mail.Body = "これは条件に基づいて送信されたメールです。\nThis report was generated and sent by chatGPT."
            # mail.Body += "\nReport Powered by ChatGPT"
            # mail.To = "test@example.com"
            mail.to = 'Kenji.B.Horiuchi@sony.com; hrucknj@icloud.com'
            mail.cc = 'stmuymte@gmail.com'
            # mail.bcc = 'momoko@mahodo.com'

            # 添付ファイル
            # attachment = "C:/Users/0107409377/Desktop/data/test_mail.txt" # "C:\\path\\to\\your\\file.txt"
            attachment = "C:/Users/0107409377/Desktop/data/sample.pptx" # "C:\\path\\to\\your\\file.txt"
            mail.Attachments.Add(attachment)
            mail.Send()

            print(f"メール送信に成功しました") # : {str(e)}")
    except Exception as e:
        print(f"メール送信に失敗しました: {str(e)}")

send_mail_based_on_condition(triggar) # True)
