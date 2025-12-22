import cv2 as cv
import numpy as np

# 读取图像
img = cv.imread('Data/Kapibala.jpeg', cv.IMREAD_GRAYSCALE)

# 应用 Laplacian 算子
laplacian = cv.Laplacian(img, cv.CV_64F)

# 显示结果
cv.imshow('Laplacian', laplacian)
cv.waitKey(0)
cv.destroyAllWindows()