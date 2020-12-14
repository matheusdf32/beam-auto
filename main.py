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
from PIL import ImageFilter
from PIL import Image
from beamngDecisionComponent import BeamNgDecisionComponent

d = d3dshot.create(capture_output="numpy")
data = {'throttle_input': 0, 'steering_input': 0, 'speed': 0}
toSend = {}

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
        global toSend
        data = conn.recv(128)
        # data = re.search('\n(.+?)\n', data)
        data = json.loads(data)
        conn.send((json.dumps(toSend) + '\n\r').encode('utf-8'))
        # print(data)

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return line_image


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
    # pts1 = np.float32([[350, 240], [550, 240], [240, 400], [800, 400]])
    pts1 = np.float32([[0, 220], [900, 220], [100, 380], [900, 380]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    decisionComponent = BeamNgDecisionComponent(0.5, 3, 3, 0.9)
    while 1:
        frameQueue += 1
        img = d.screenshot(region=(62, 40, 960, 540))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        blur = cv2.GaussianBlur(rgb, (5, 5), 0)

        grad = np.asarray(Image.fromarray(blur)
                          # .filter(ImageFilter.SMOOTH_MORE)
                          # .filter(ImageFilter.SMOOTH_MORE)
                          .filter(ImageFilter.FIND_EDGES)
                          .filter(ImageFilter.SMOOTH_MORE)
                          .filter(ImageFilter.SMOOTH_MORE)
                          .filter(ImageFilter.FIND_EDGES)
                          .filter(ImageFilter.EDGE_ENHANCE_MORE)
                          # .filter(ImageFilter.SMOOTH_MORE)
                          )
        gray = cv2.cvtColor(grad, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        img = cv2.warpPerspective(gray, matrix, (width, height))
        finalFrame = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)[1]

        # print(decisionComponent.decide(finalFrame)*-1)
        steering_input = decisionComponent.decide(finalFrame)*-1

        toSend['throttle_input'] = 0.5 if data['speed'] < 40 else 0
        toSend['steering_input'] = steering_input

        imageShow(finalFrame)



        # print("throttle_input: " + '{0:,.2f}'.format(data["throttle_input"])
        #       + " steering_input: " + '{0:,.2f}'.format(data["steering_input"])
        #       + " speed: " + '{0:,.2f}'.format(data["speed"] * 3.6) + "km/h"
        #       + ' running at {0:,.2f} fps'.format(1 / (time.time() - last_time)))

        # conn.send((json.dumps(toSend) + '\n\r').encode('utf-8'))  # echo
        last_time = time.time()
    conn.close()


if __name__ == '__main__':
    main()
