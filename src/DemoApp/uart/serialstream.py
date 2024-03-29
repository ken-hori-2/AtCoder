import serial
import serial.tools.list_ports as serialtool
import json
import os
import sys
import time

# def getSetting():
#     """
#     Read the data from json.
#     """
#     path = "setting.json"
#     if os.path.exists(path):
#         with open(path, "r") as fr:
#             dat = json.load(fr)
#         return dat
#     else:
#         return False


# def checkPort(port=None):
#     """
#     Check for existing serial ports.
#     If there is only one serial port, it will be returned,
#     but if there are multiple, you will be prompted to select it.

#     If there is a default among multiple existing,
#     you can specify it in setting.json.
#     """
#     portlist = []
#     ports = serialtool.comports()
#     setting = getSetting()
#     if setting:
#         for p in ports:
#             portlist.append(p)
#             if p[1].find(setting["name"]) != -1 and port == None:
#                 return(p[0])
#             elif p[1].find(port) != -1:
#                 return(p[0])

#     """
#     Returns False if the number of portlists is 0
#     """
#     if len(portlist) == 0:
#         return False

#     """
#     If there is only one port list, return it as a serial port number
#     """
#     if len(portlist) == 1:
#         return portlist[0][0]

#     """
#     If there are multiple port lists, the selected port is returned.
#     (*) Currently, we are making a simple selection with the CLI,
#     but since it also supports GUI etc.,
#     it may be better to return the list itself.
#     """
#     while True:
#         for idx, p in enumerate(portlist):
#             print("%d: %s" % (idx, p))

#         sel = input("select port :")
#         if len(portlist) >= int(sel):
#             return portlist[int(sel)][0]

#         # return portlist


class SerialStream:

    def __init__(self, **kw):
        self._serial = None
        self._buf = ""
        self._logfile = None
        self._serialParams = {
            "port": False,
            "baudrate": 9600,
            "bytesize": 8,
            "parity": "N",
            "stopbits": 1,
            "timeout": 1,
            "xonxoff": False,
            "rtscts": False,
            "dsrdtr": False,
        }
        self._params = {
            "logfile": "",
            "monitor": False
        }
        self.setParams(**kw)

    def setParams(self, **kw):
        """
        Register the key / value combination corresponding to
        the dict key in dict.
        """
        keys = kw.keys()
        keylist = self._serialParams.keys()
        for key in keylist:
            if key in keys:
                self[key] = kw[key]

        keylist = self._params.keys()
        for key in keylist:
            if key in keys:
                self[key] = kw[key]

    def open(self, **kw):
        """
        Open the serial. The default parameter set in the class is used,
        but at least port must be specified.
        """
        self.setParams(**kw)
        if self["port"] and self._serial == None:
            try:
                self._serial = serial.Serial(
                    port = self["port"],
                    baudrate = self["baudrate"],
                    timeout = self["timeout"],
                    bytesize = self["bytesize"],
                    parity = self["parity"],
                    stopbits = self["stopbits"],
                    xonxoff = self["xonxoff"],
                    rtscts = self["rtscts"],
                    dsrdtr = self["dsrdtr"]
                    )
            except serial.serialutil.SerialException:
                # serial open error
                raise

    def close(self):
        """
        Discard the serial if it is open.
        """
        if self._serial != None:
            self._serial.close()
            self._serial = None

    def _monitor(self, buf):
        if self["monitor"]:
            print(buf, end="", flush=True)

    def _read(self):
        """
        It is not a clear idea to call it directly because it is mainly used internally.
        However, in some cases calling directly can help your project.
        """
        if self._serial.inWaiting() > 0:
            buf = self._serial.read()
            buf = buf.decode()
            self._writeLog(buf)
            self._monitor(buf)
            """
            If the result string of strip does not exist,
            the contents of the retained buffer are cleared.
            Sometimes this is an inappropriate behavior.
            In that case, it should be modified so that the buffer is not cleared.
            """
            if len(buf.strip()) > 0:
                self._buf += buf
            else:
                self._buf = ""

    def waitFor(self, target, timeout=False):
        """
        Checks if the buffer received from the serial matches the target string,
        and if so, returns the string.
        If the target string is passed as a tuple, it verifies that it matches
        each content in the tuple and returns the matching string.
        If a timeout (seconds) is specified, False is returned if there is no
        reaction for the specified time.
        """
        starttime = time.time()
        buflen = 0
        while True:
            self._read()
            if timeout != False:
                if buflen == len(self._buf):
                    if (time.time() - starttime) > timeout:
                        return False
                else:
                    starttime = time.time()
                    buflen = len(self._buf)
            if isinstance(target, tuple):
                for t in target:
                    if self._buf.find(t) != -1:
                        return t
            else:
                if self._buf.find(target) != -1:
                    return target

    def write(self, msg):
        """
        When writing serially, the buffer it holds is cleared.
        Therefore, if you need to work on the buffer,
        you need to do it before writing.
        """
        self._buf = ""
        time.sleep(0.5)
        self._serial.write(msg.encode())

    def sleep(self, sleepTime):
        """
        It does not block serial reading (it is stored in the buffer)
        and waits for the specified time.
        Note that if the line you are waiting for contains the character
        you want to search for, it will be ignored.
        """
        ti = time.time()
        while True:
            self._read()
            if time.time() - ti > sleepTime:
                return
            time.sleep(0.02)

    def sendBreak(self):
        self._serial.sendBreak()

    def getBuffer(self):
        """
        If you want the contents of the buffer,
        you can get the line just before waitFor.
        """
        return self._buf

    def startLog(self, **kw):
        """
        If you specify logfile, the file is opened and ready to write
        out the contents of the serial read.
        """
        self.setParams(**kw)
        if self["logfile"] != "":
            self._logfile = open(self["logfile"], "w")

    def closeLog(self):
        """
        Be sure to close after use after startLog.
        """
        if self["logfile"] != "":
            self._logfile.close()
            self._logfile = None

    def _writeLog(self, buf):
        """
        Called internally.
        """
        if self._logfile != None:
            self._logfile.write(buf)

    def __setitem__(self, key, value):
        if key in self._serialParams.keys():
            self._serialParams[key] = value
        if key in self._params.keys():
            self._params[key] = value

    def __getitem__(self, key):
        if key in self._serialParams.keys():
            return self._serialParams[key]
        if key in self._params.keys():
            return self._params[key]

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exctype, excvalue, traceback):
        self.close()