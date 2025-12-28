import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while True:
    # 获得每一帧
    ret, frame = cap.read()
    if not ret:
        break
    # 转换到HSV空间
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)

    # 定义HSV中蓝色的范围
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # 根据阈值构建掩膜
    mask = cv.inRange(hsv,lower_blue,upper_blue)

    # 对原图像和掩膜进行位运算
    res = cv.bitwise_and(frame,frame,mask=mask)

    cv.imshow('frame',frame)
    cv.imshow('mask',mask) 
    cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()