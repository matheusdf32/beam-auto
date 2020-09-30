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
import d3dshot # https://pypi.org/project/d3dshot/
# import win32gui
# import win32ui
import asyncio
import threading
from PIL import Image

d = d3dshot.create(capture_output="numpy")
data = {'throttle_input': 0, 'steering_input': 0, 'speed': 0}

# alt u hides hud ingmae
def captureScreen():  # BeamNG.drive - 0.20.2.0.10611 - RELEASE - x64
    # printscreen = np.array(ImageGrab.grab(bbox=(62, 40, 960, 540)))
    printscreen = d.screenshot(region=(62, 40, 960, 540))
    # cv2.imshow('window', cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
    return printscreen

def imageShow(img):
    cv2.imshow('window', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
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
    # loop = asyncio.get_event_loop()
    # loop.subprocess_exec(getData(conn))
    b = threading.Thread(name='getData', target=getData, args=[conn])
    b.start()
    last_time = time.time()
    frameQueue = 0
    # while 1:
    #     toSend = {}
    #     frameQueue += 1
    #     # data = conn.recv(128)
    #     # data = str(data, 'utf-8')
    #     # print("received data: ", data)  # steering_input
    #     # conn.send((json.dumps(toSend) + '\n\r').encode('utf-8'))
    #     img = captureScreen()
    #     data = conn.recv(128)
    #     if not img.any(): pass
    #     data = str(data, 'utf-8')
    #     data = re.search('\n(.+?)\n', data)
    #     if (data):
    #         data = json.loads(data.group(1))
    #     if not isinstance(data, str):
    #         # data['steering_input'] = 1
    #         print("received data: ", data)  # steering_input
    #         print('running at {} fps'.format(1 / (time.time() - last_time)) + ' queue: ' + str(frameQueue))
    #         frameQueue -= 1
    #
    #     # conn.send((json.dumps(toSend) + '\n\r').encode('utf-8'))  # echo
    #     last_time = time.time()

    # img = captureScreen()
    # print(img.all())
    # return


    while 1:
        frameQueue += 1

        # data = conn.recv(128)
        # data = str(data, 'utf-8')
        # data = re.search('\n(.+?)\n', data)
        # # if data:
        # #     print("received data: ", data)
        # #     data = json.loads(data.group(1))
        # if not isinstance(data, str) and data:
        #     data = json.loads(data.group(1))
        # else:
        #     pass
        # print("throttle_input: ", data["throttle_input"])
        print("throttle_input: ", data["throttle_input"])
        img = d.screenshot(region=(62, 40, 960, 540))
        imageShow(img)
        # print(img[0][0], data)

        # print(img[0][0], "adsfas")
        # if img.any():
        #     pass
        # data = str(data, 'utf-8')
        # data = re.search('\n(.+?)\n', data)
        # if (data):
        #     data = json.loads(data.group(1))
        # if not isinstance(data, str):
        #     # data['steering_input'] = 1
        #     print("received data: ", data)  # steering_input
        #     print('running at {} fps'.format(1 / (time.time() - last_time)) + ' queue: ' + str(frameQueue))
        #     frameQueue -= 1
        print('running at {} fps'.format(1 / (time.time() - last_time)) + ' queue: ' + str(frameQueue))

        # conn.send((json.dumps(toSend) + '\n\r').encode('utf-8'))  # echo
        last_time = time.time()
    conn.close()


if __name__ == '__main__':
    main()
