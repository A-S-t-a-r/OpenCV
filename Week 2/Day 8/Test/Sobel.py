import cv2 as cv
import numpy as np

# 读取图像
img = cv.imread('Data/Kapibala.jpeg',0)

# 计算 x 方向梯度
sobel_x = cv.Sobel(img,cv.CV_64F,1,0,ksize=3)

# 计算 y 方向梯度
sobel_y = cv.Sobel(img,cv.CV_64F,0,1,ksize=3)

# 计算梯度幅度
sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)

# 显示结果
cv.imshow('Sobel X', sobel_x)
cv.imshow('Sobel Y', sobel_y)
cv.imshow('Sobel Combined', sobel_combined)
cv.waitKey(0)
cv.destroyAllWindows()