import socket

import numpy as np
import json
import time
import re
# import subprocess
import win32con
from PIL import ImageGrab
import cv2
from mss import mss
import win32gui
import win32ui
from PIL import Image

# alt u hides hud ingmae
def captureScreen(): # BeamNG.drive - 0.20.2.0.10611 - RELEASE - x64
    printscreen = np.array(ImageGrab.grab(bbox=(62,40,960,540)))
    # cv2.imshow('window', cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
    return printscreen

def is_json(myjson):
  try:
    json_object = json.loads(myjson)
  except ValueError as e:
    return False
  return True

def main():
    TCP_IP = '127.0.0.1'
    TCP_PORT = 4343
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_IP, TCP_PORT))
    s.listen(1)
    conn, addr = s.accept()
    while 1:
        data = conn.recv(128)
        # data = re.search('\n(.*)\n', str(data, 'utf-8') )
        data = str(data, 'utf-8')
        for line in data.splitlines():
            if(is_json(line)):
                data = json.loads(line)
        if not isinstance(data, str):
            # data = json.loads(data)
            img = captureScreen()
            # data['steering_input'] = 1
            print("received data: ", data)  # steering_input
            toSend = {}

        # if (data['speed'] < 16.6):
        #     toSend = {'throttle_input': 1}

        conn.send((json.dumps(toSend) + '\n\r').encode('utf-8'))  # echo
    conn.close()


if __name__ == '__main__':
    main()
