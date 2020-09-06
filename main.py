import socket

import numpy as np
import json
# import time
# import subprocess
import win32con
# from PIL import ImageGrab
import cv2
import win32gui
import win32ui
from PIL import Image

def captureScreen():
    hwnd = win32gui.FindWindow(None, 'Calculator')
    left, top, right, bot = win32gui.GetWindowRect(hwnd)
    w = right - left
    h = bot - top
    wDC = win32gui.GetWindowDC(hwnd)
    dcObj = win32ui.CreateDCFromHandle(wDC)
    cDC = dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
    bmpinfo = dataBitMap.GetInfo()
    # cDC.SelectObject(dataBitMap)
    # cDC.BitBlt((0, 0), (w, h), dcObj, (0, 0), win32con.SRCCOPY)
    # dataBitMap.SaveBitmapFile(cDC, 'game_ss')
    signedIntsArray = dataBitMap.GetBitmapBits(True)
    # img = np.fromstring(signedIntsArray, dtype='uint8')
    img = np.float32(Image.frombuffer(
        'RGB',
        (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
        signedIntsArray, 'raw', 'BGRX', 0, 1))
    # cv2.imshow('window', img)
    print(wDC) 
    cv2.imshow('window', cv2.cvtColor(img, cv2.COLOR_RGBA2RGB))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Free Resources
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())
    return img



def main():
    # start_time = time.time()
    # x = 1  # displays the frame rate every 1 second
    # counter = 0

    TCP_IP = '127.0.0.1'
    TCP_PORT = 4343
    # BUFFER_SIZE = 20  # Normally 1024, but we want fast response

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # s.setsockopt(socket.SOL_TCP, 23, 5)
    s.bind((TCP_IP, TCP_PORT))
    s.listen(0)

    conn, addr = s.accept()
    print('Connection address: ', addr)
    while 1:
        # img = np.float32(captureScreen())
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        #     break
        img = captureScreen()
        if not img.any(): break
        # data = conn.recv(128)
        # if not data: break
        # # subprocess.call('cls', shell=True)
        # data = json.loads(data)
        #
        # # data['steering_input'] = 1
        # print("received data: ", data)  # steering_input
        # toSend = {}
        #
        # if (data['speed'] < 16.6):
        #     toSend = {'throttle_input': 1}
        #
        # conn.send((json.dumps(toSend) + '\n\r').encode('utf-8'))  # echo

        # counter += 1
        # if (time.time() - start_time) > x:
        #     print("FPS: ", counter / (time.time() - start_time))
        #     counter = 0
        #     start_time = time.time()
    conn.close()


if __name__ == '__main__':
    main()
