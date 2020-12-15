import socket

import numpy as np
import json
import time
import math
import statistics
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
        # data = re.search('\n(.+?)\n', data)
        try:
            # data = conn.recv(128)
            data = json.loads(conn.recv(128))
            conn.send((json.dumps(toSend) + '\n\r').encode('utf-8'))
        except:
            print('conexao interrompida')
            pass


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return line_image


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        # tangle
        # [(200, height), (700, height), (450, 250)]
        [(400, height - 200), (550, height - 200), (550, 250), (400, 250), ]
    ])
    # mask = np.full_like(image, (255, 255, 255), dtype=np.uint8)
    # cv2.fillPoly(mask, polygons, (0, 0, 0))
    mask = np.full_like(image, 255)
    cv2.fillPoly(mask, polygons, 0)
    masked_img = cv2.bitwise_and(image, mask)
    return masked_img

def main():
    # loop = asyncio.get_event_loop()
    TCP_IP = '127.0.0.1'
    TCP_PORT = 4343
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_IP, TCP_PORT))
    s.listen(1)
    conn, addr = s.accept()
    if (conn):
        print('conexao estabelecida')

    b = threading.Thread(name='getData', target=getData, args=[conn])
    b.start()

    width, height = 960, 540
    pts1 = np.float32([[0, 240], [900, 240], [100, 380], [900, 380]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    decisionComponent = BeamNgDecisionComponent(1, 3, 6, 1.1) # sweetspot: (1, 3, 6, 1.1)
    speedLimit = 90 # sweetspot: 10
    while 1:
        img = d.screenshot(region=(62, 40, 960, 540))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        blur = cv2.GaussianBlur(rgb, (5, 5), 0)

        grad = np.asarray(Image.fromarray(blur)
                          .filter(ImageFilter.FIND_EDGES)
                          .filter(ImageFilter.EDGE_ENHANCE_MORE)
                          )
        gray = cv2.cvtColor(grad, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        img2 = cv2.warpPerspective(gray, matrix, (width, height))
        finalFrame = cv2.threshold(img2, 80, 255, cv2.THRESH_BINARY)[1]
        steering_input = decisionComponent.decide(finalFrame) * -1
        toSend['throttle_input'] = 1.0 if data['speed'] < speedLimit else 0.0
        # toSend['brake_input'] = 1 if data['speed'] < speedLimit + 1 else 0
        toSend['steering_input'] = steering_input

        imageShow(finalFrame)

    conn.close()


if __name__ == '__main__':
    main()
