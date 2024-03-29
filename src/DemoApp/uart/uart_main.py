import serialstream

"""
COMポートの取得とSerialStreamの生成。
ただしCOMポートが明らかになっている場合はこれを使う必要はなく、
その場合は port="COM4" のように指定する。

※checkPortはCOMポートが取得できない場合にFalseを返す。

SerialStreamはもちろんopenとcloseでの利用も可能なので場合によって使い分ける。
"""

import serial.tools.list_ports as serialtool
import json
import os

def getSetting():
    """
    Read the data from json.
    """
    path = "setting.json"
    if os.path.exists(path):
        with open(path, "r") as fr:
            dat = json.load(fr)
        return dat
    else:
        return False


def checkPort(port=None):
    """
    Check for existing serial ports.
    If there is only one serial port, it will be returned,
    but if there are multiple, you will be prompted to select it.

    If there is a default among multiple existing,
    you can specify it in setting.json.
    """
    portlist = []
    ports = serialtool.comports()
    setting = getSetting()
    if setting:
        for p in ports:
            portlist.append(p)
            if p[1].find(setting["name"]) != -1 and port == None:
                return(p[0])
            elif p[1].find(port) != -1:
                return(p[0])

    """
    Returns False if the number of portlists is 0
    """
    if len(portlist) == 0:
        return False

    """
    If there is only one port list, return it as a serial port number
    """
    if len(portlist) == 1:
        return portlist[0][0]

    """
    If there are multiple port lists, the selected port is returned.
    (*) Currently, we are making a simple selection with the CLI,
    but since it also supports GUI etc.,
    it may be better to return the list itself.
    """
    while True:
        for idx, p in enumerate(portlist):
            print("%d: %s" % (idx, p))

        sel = input("select port :")
        if len(portlist) >= int(sel):
            return portlist[int(sel)][0]

        # return portlist


port = "COM8" # checkPort()
if port:
    with serialstream.SerialStream(port=port, monitor=True) as ser:

        """
        文字列を待つ。
        ※ただしwaitForを連続した場合は正しい動きが期待できないので
        必ず何かしらをwriteしてから次のwaitForを実行する。
        """
        ser.waitFor("login")

        """
        文字列を送る、改行文字も忘れずに。
        どうしても煩わしい場合はSerialStreamのwriteを修正すれば良い。
        業務にて改行が必要ないケースが多々あったためこのような仕様。
        """
        ser.write("admin\n")


        ser.waitFor("Password")
        ser.write("admin\n")


        """
        複数パターンの文字列が期待される場合はタプルで渡す。
        ヒットした文字列が返ってくるので処理を分ける。
        """
        if ser.waitFor((">", "Login incorrect")) == ">":

            """
            タイムアウトを指定して文字列を待つ場合このように処理
            """
            #if ser.waitFor("hoge", timeout=3) == False:
            #    print("hogeきません！")
            ser.write("terminal length 0\n")
            ser.waitFor(">")

            """
            好きなタイミングでログファイルへの書き込みを開始・終了できる。
            """
            ser.startLog(logfile="log.txt")
            ser.write("show run\n")
            ser.waitFor(">")
            ser.closeLog()

            """
            waitForでヒットした行に含まれる文字列が欲しい場合取り出すことが出来る
            （個体シリアルやデバイスナンバーを取得したい場合など）
            ※しかし、リアルタイム性が必要ないのであれば書き出したログから取得するほうが
            スマートなのでそちらを推奨。
            """
            print(ser.getBuffer())