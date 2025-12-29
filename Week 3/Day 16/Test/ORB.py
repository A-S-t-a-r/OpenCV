import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
img = cv.imread('Data/feature_map.png', cv.IMREAD_GRAYSCALE)
 
# 初始化 ORB 检测器
orb = cv.ORB_create()
 
# 使用ORB找到关键点
kp = orb.detect(img,None)
 
# 使用ORB计算描述符
kp, des = orb.compute(img, kp)
 
# 仅绘制关键点的位置，不绘制大小和方向
img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)

plt.imshow(img2), plt.show()