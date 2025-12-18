import cv2 as cv

# 读取图像
img = cv.imread('Data/Dinosaur.jpg',0)

# Otsu's 二值化
ret,thresh3=cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

# 显示结果
cv.imshow('Otsu\'s Threshold', thresh3)
cv.waitKey(0)
cv.destroyAllWindows