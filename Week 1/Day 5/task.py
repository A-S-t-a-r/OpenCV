import cv2 as cv
import numpy as np

# 读取图像
img = cv.imread('Data/Elegant_person.jpg',0)

# 定义结构元素
kernel = np.ones((5,5),np.uint8)

# 开运算去除噪点
opened = cv.morphologyEx(img,cv.MORPH_OPEN,kernel)

# 闭运算填充空洞
closed = cv.morphologyEx(img,cv.MORPH_CLOSE,kernel)

cv.imshow('Original', img)
cv.imshow('Opened', opened)
cv.imshow('Closed', closed)
cv.waitKey(0)