
from cv2 import cv2
import d3dshot  # https://pypi.org/project/d3dshot/
import numpy as np
from beamngDecisionComponent import BeamNgDecisionComponent
from PIL import Image, ImageFilter

SHOOTER = d3dshot.create(capture_output="numpy")


def image_show(img):
    cv2.imshow('window', img)
    cv2.waitKey(0)


width, height = 960, 540
pts1 = np.float32([[0, 240], [900, 240], [100, 380], [900, 380]])
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
img = SHOOTER.screenshot(region=(62, 40, 960, 540))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite('00_original.png', img)

img = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imwrite('01_first_blur.png', img)

img = np.asarray(Image.fromarray(img).filter(ImageFilter.FIND_EDGES).filter(ImageFilter.EDGE_ENHANCE_MORE))
cv2.imwrite('02_FindEdges_EdgeEnhanceMore.png', img)

img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imwrite('03_RGBtoGRAY.png', img)

img = cv2.GaussianBlur(img, (3, 3), 0)
cv2.imwrite('04_second_blur.png', img)

img = cv2.warpPerspective(img, matrix, (width, height))
cv2.imwrite('05_warped.png', img)

img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite('06_THRESH_BINARY.png', img)
# img = np.asarray(Image.fromarray(img).convert('L'))
# img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
# image_show(img)

