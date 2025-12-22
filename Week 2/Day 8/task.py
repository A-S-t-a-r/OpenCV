import cv2 as cv

# 读取图像
img = cv.imread('Data/Kapibala.jpeg',cv.IMREAD_GRAYSCALE)

# Sobel
sobel_x = cv.Sobel(img,cv.CV_64F,1,0,ksize=3)
sobel_y = cv.Sobel(img,cv.CV_64F,0,1,ksize=3)

# Canny
edges = cv.Canny(img,50,100)

cv.imshow('Sobel X', cv.convertScaleAbs(sobel_x))
cv.imshow('Sobel Y', cv.convertScaleAbs(sobel_y))
cv.imshow('Canny', edges)
cv.waitKey(0)