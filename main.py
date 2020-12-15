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


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        # tangle
        # [(200, height), (700, height), (450, 250)]
        [(400, height-200), (550, height-200), (550, 250), (400, 250), ]
    ])
    # mask = np.full_like(image, (255, 255, 255), dtype=np.uint8)
    # cv2.fillPoly(mask, polygons, (0, 0, 0))
    mask = np.full_like(image, 255)
    cv2.fillPoly(mask, polygons, 0)
    masked_img = cv2.bitwise_and(image, mask)
    return masked_img

def shadowRemove(img):
    blue, green, red = cv2.split(img)

    blue[blue == 0] = 1
    green[green == 0] = 1
    red[red == 0] = 1

    div = np.multiply(np.multiply(blue, green), red) ** (1.0 / 3)

    a = np.log1p((blue / div) - 1)
    b = np.log1p((green / div) - 1)
    c = np.log1p((red / div) - 1)

    a1 = np.atleast_3d(a)
    b1 = np.atleast_3d(b)
    c1 = np.atleast_3d(c)
    rho = np.concatenate((c1, b1, a1), axis=2)  # log chromaticity on a plane

    U = [[1 / math.sqrt(2), -1 / math.sqrt(2), 0], [1 / math.sqrt(6), 1 / math.sqrt(6), -2 / math.sqrt(6)]]
    U = np.array(U)  # eigens

    X = np.dot(rho, U.T)  # 2D points on a plane orthogonal to [1,1,1]

    d1, d2, d3 = img.shape

    e_t = np.zeros((2, 181))
    for j in range(181):
        e_t[0][j] = math.cos(j * math.pi / 180.0)
        e_t[1][j] = math.sin(j * math.pi / 180.0)

    Y = np.dot(X, e_t)
    nel = img.shape[0] * img.shape[1]

    bw = np.zeros((1, 181))

    for i in range(181):
        bw[0][i] = (3.5 * np.std(Y[:, :, i])) * ((nel) ** (-1.0 / 3))

    entropy = []
    for i in range(181):
        temp = []
        comp1 = np.mean(Y[:, :, i]) - 3 * np.std(Y[:, :, i])
        comp2 = np.mean(Y[:, :, i]) + 3 * np.std(Y[:, :, i])
        for j in range(Y.shape[0]):
            for k in range(Y.shape[1]):
                if Y[j][k][i] > comp1 and Y[j][k][i] < comp2:
                    temp.append(Y[j][k][i])
        nbins = round((max(temp) - min(temp)) / bw[0][i])
        (hist, waste) = np.histogram(temp, bins=nbins)
        hist = filter(lambda var1: var1 != 0, hist)
        hist1 = np.array([float(var) for var in hist])
        hist1 = hist1 / sum(hist1)
        entropy.append(-1 * sum(np.multiply(hist1, np.log2(hist1))))

    angle = entropy.index(min(entropy))

    e_t = np.array([math.cos(angle * math.pi / 180.0), math.sin(angle * math.pi / 180.0)])
    e = np.array([-1 * math.sin(angle * math.pi / 180.0), math.cos(angle * math.pi / 180.0)])

    I1D = np.exp(np.dot(X, e_t))  # mat2gray to be done

    p_th = np.dot(e_t.T, e_t)
    X_th = X * p_th
    mX = np.dot(X, e.T)
    mX_th = np.dot(X_th, e.T)

    mX = np.atleast_3d(mX)
    mX_th = np.atleast_3d(mX_th)

    theta = (math.pi * float(angle)) / 180.0
    theta = np.array([[math.cos(theta), math.sin(theta)], [-1 * math.sin(theta), math.cos(theta)]])
    alpha = theta[0, :]
    alpha = np.atleast_2d(alpha)
    beta = theta[1, :]
    beta = np.atleast_2d(beta)

    # Finding the top 1% of mX
    mX1 = mX.reshape(mX.shape[0] * mX.shape[1])
    mX1sort = np.argsort(mX1)[::-1]
    mX1sort = mX1sort + 1
    mX1sort1 = np.remainder(mX1sort, mX.shape[1])
    mX1sort1 = mX1sort1 - 1
    mX1sort2 = np.divide(mX1sort, mX.shape[1])
    mX_index = [[x, y, 0] for x, y in zip(list(mX1sort2), list(mX1sort1))]
    mX_top = [mX[x[0], x[1], x[2]] for x in mX_index[:int(0.01 * mX.shape[0] * mX.shape[1])]]
    mX_th_top = [mX_th[x[0], x[1], x[2]] for x in mX_index[:int(0.01 * mX_th.shape[0] * mX_th.shape[1])]]
    X_E = (statistics.median(mX_top) - statistics.median(mX_th_top)) * beta.T
    X_E = X_E.T

    for i in range(X_th.shape[0]):
        for j in range(X_th.shape[1]):
            X_th[i, j, :] = X_th[i, j, :] + X_E

    rho_ti = np.dot(X_th, U)
    c_ti = np.exp(rho_ti)
    sum_ti = np.sum(c_ti, axis=2)
    sum_ti = sum_ti.reshape(c_ti.shape[0], c_ti.shape[1], 1)
    r_ti = c_ti / sum_ti

    r_ti2 = 255 * r_ti
    return r_ti2

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
    decisionComponent = BeamNgDecisionComponent(1, 3, 6, 0.9)
    while 1:
        frameQueue += 1
        img = d.screenshot(region=(62, 40, 960, 540))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # rgb = shadowRemove(rgb)

        # rgb = region_of_interest(rgb)

        blur = cv2.GaussianBlur(rgb, (5, 5), 0)

        grad = np.asarray(Image.fromarray(blur)
                          .filter(ImageFilter.FIND_EDGES)
                          .filter(ImageFilter.EDGE_ENHANCE_MORE)
                          )
        gray = cv2.cvtColor(grad, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # gray = region_of_interest(gray)

        img2 = cv2.warpPerspective(gray, matrix, (width, height))
        finalFrame = cv2.threshold(img2, 80, 255, cv2.THRESH_BINARY)[1]

        # print(decisionComponent.decide(finalFrame)*-1)
        steering_input = decisionComponent.decide(finalFrame) * -1

        toSend['throttle_input'] = 0.5 if data['speed'] < 20 else 0
        toSend['steering_input'] = steering_input

        # imageShow(region_of_interest(gray))
        # imageShow(region_of_interest(rgb))
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
