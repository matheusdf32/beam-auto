import json
import socket
import threading
from typing import Dict

from cv2 import cv2
import d3dshot  # https://pypi.org/project/d3dshot/
import numpy as np
from beamngDecisionComponent import BeamNgDecisionComponent
from PIL import Image, ImageFilter

SHOOTER = d3dshot.create(capture_output="numpy")
DRIVING_DATA = {'throttle_input': 0, 'steering_input': 0, 'speed': 0}
DRIVING_INSTRUCTIONS: Dict[str, int] = {}


def image_show(img):
    cv2.imshow('window', img)
    cv2.waitKey(1)


def get_data(conn):
    while 1:
        global DRIVING_DATA
        global DRIVING_INSTRUCTIONS
        try:
            DRIVING_DATA = json.loads(conn.recv(128))
            conn.send((json.dumps(DRIVING_INSTRUCTIONS) + '\n\r'
                       ).encode('utf-8'))
        except Exception:
            print('conexao interrompida')


def main():  # alt u hides hud ingame
    TCP_IP = '127.0.0.1'
    TCP_PORT = 4343
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_IP, TCP_PORT))
    s.listen(1)
    conn, _addr = s.accept()
    if (conn):
        print('conexao estabelecida')

    b = threading.Thread(name='getData', target=get_data, args=[conn])
    b.start()

    width, height = 960, 540
    pts1 = np.float32([[0, 240], [900, 240], [100, 380], [900, 380]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    decision_component = BeamNgDecisionComponent(
        1, 3, 6, 1.1)  # sweetspot: (1, 3, 6, 1.1)
    speed_limit = 90  # sweetspot: 10
    while 1:
        img = SHOOTER.screenshot(region=(62, 40, 960, 540))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        blur = cv2.GaussianBlur(rgb, (5, 5), 0)

        grad = np.asarray(Image.fromarray(blur)
                          .filter(ImageFilter.FIND_EDGES)
                          .filter(ImageFilter.EDGE_ENHANCE_MORE)
                          )
        gray = cv2.cvtColor(grad, cv2.COLOR_RGB2GRAY)
        soft_gray = cv2.GaussianBlur(gray, (3, 3), 0)

        warped = cv2.warpPerspective(soft_gray, matrix, (width, height))
        final_frame = cv2.threshold(warped, 80, 255, cv2.THRESH_BINARY)[1]
        steering_input = decision_component.decide(final_frame) * -1
        DRIVING_INSTRUCTIONS['throttle_input'] = (
            1.0 if DRIVING_DATA['speed'] < speed_limit else 0.0
        )
        DRIVING_INSTRUCTIONS['steering_input'] = steering_input

        image_show(final_frame)

    conn.close()


if __name__ == '__main__':
    main()
