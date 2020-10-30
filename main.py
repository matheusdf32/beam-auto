import socket

import numpy as np
import json
import time
import re
# import subprocess
# import win32con
from PIL import ImageGrab
import cv2
# import mss
import d3dshot  # https://pypi.org/project/d3dshot/
# import win32gui
# import win32ui
import asyncio
import threading
from PIL import Image

d = d3dshot.create(capture_output="numpy")
data = {'throttle_input': 0, 'steering_input': 0, 'speed': 0}


# alt u hides hud ingame
def captureScreen():  # BeamNG.drive - 0.20.2.0.10611 - RELEASE - x64
    printscreen = d.screenshot(region=(62, 40, 960, 540))
    return printscreen


def imageShow(img):
    cv2.imshow('window', img)
    cv2.waitKey(1)


def is_json(myjson):
    try:
        json_object = json.loads(myjson)
    except ValueError as e:
        return False
    return True


def getData(conn):
    while 1:
        global data
        data = conn.recv(128)
        # data = re.search('\n(.+?)\n', data)
        data = json.loads(data)
        # print(data)


def main():
    loop = asyncio.get_event_loop()
    TCP_IP = '127.0.0.1'
    TCP_PORT = 4343
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_IP, TCP_PORT))
    s.listen(1)
    conn, addr = s.accept()

    b = threading.Thread(name='getData', target=getData, args=[conn])
    b.start()
    last_time = time.time()
    frameQueue = 0

    width, height = 960, 540
    pts1 = np.float32([[323, 238], [581, 219], [89, 390], [835, 394]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)


    while 1:
        frameQueue += 1
        img = d.screenshot(region=(62, 40, 960, 540))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # imageShow(img)
        imgOut = cv2.warpPerspective(img, matrix, (width, height))
        imageShow(imgOut)

        print("throttle_input: " + '{0:,.2f}'.format(data["throttle_input"])
              + " steering_input: " + '{0:,.2f}'.format(data["steering_input"])
              + " speed: " + '{0:,.2f}'.format(data["speed"] * 3.6) + "km/h"
              + ' running at {0:,.2f} fps'.format(1 / (time.time() - last_time)))

        # conn.send((json.dumps(toSend) + '\n\r').encode('utf-8'))  # echo
        last_time = time.time()
    conn.close()


if __name__ == '__main__':
    main()
